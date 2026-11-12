"""Experimental nearby-first tracking with a cadence-aware room hold."""

from __future__ import annotations

from collections import deque
from math import inf
from statistics import quantiles
from typing import TYPE_CHECKING, Any

from .room_tracking import TemporalTracker

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence


class AdaptiveRoomTracker(TemporalTracker):
    """Combine a tentative native startup fix with learned room confirmation."""

    def __init__(self, decay_intervals: float, travel_seconds: float) -> None:
        super().__init__(adjacent_travel_ms=travel_seconds * 1000)
        self.decay_intervals = decay_intervals
        self._intervals: deque[float] = deque(maxlen=24)
        self._last_packet = -inf
        self._provisional_room: str | None = None
        self._provisional_at = -inf

    def _observe_packet(self, now: float, evidence: Mapping[str, float], eligible: Collection[str]) -> float | None:
        packet = max(
            (stamp for key in eligible if (stamp := evidence.get(key, -inf)) <= now + 500),
            default=-inf,
        )
        if packet <= self._last_packet:
            return None
        interval = packet - self._last_packet
        self._last_packet = packet
        # Long outages do not teach the tracker a slower cadence.
        if 500 <= interval <= 15_000:
            self._intervals.append(interval)
        return min(now, packet) if now - packet <= self.PACKET_MAX_AGE_MS else None

    def _cadence_ms(self) -> float:
        if len(self._intervals) < 4:
            return 4000
        upper_quartile = quantiles(self._intervals, n=4, method="inclusive")[2]
        return min(10_000, max(1000, upper_quartile))

    @staticmethod
    def _nearby(previous: str | None, room: str, connections: Sequence[Sequence[str]]) -> bool:
        return previous == room or any(previous in edge and room in edge for edge in connections)

    def _native_startup_allowed(
        self,
        room: str,
        candidates: Sequence[dict[str, Any]],
        now: float,
        evidence: Mapping[str, float],
        eligible: Collection[str],
    ) -> bool:
        """Use the learned shortlist when two current receiver readings support it."""
        if not candidates or any(candidate["roomId"] == room for candidate in candidates[:2]):
            return True
        return (
            sum(
                key in eligible and -500 <= now - evidence.get(key, -inf) <= self.PACKET_MAX_AGE_MS
                for key in candidates[0]["supporting"]
            )
            < 2
        )

    def _nearby_estimate(
        self,
        estimate: dict[str, Any],
        now: float,
        evidence: Mapping[str, float],
        eligible: Collection[str],
        connections: Sequence[Sequence[str]],
    ) -> dict[str, Any]:
        local = [
            candidate
            for candidate in estimate.get("allRooms", estimate["candidates"])
            if self._nearby(self.confirmed, candidate["roomId"], connections)
            and candidate["common"] >= 2
            and candidate["quality"] >= 2
            and not candidate["contradictions"]
            and candidate["score"] <= 2.2
            and sum(
                key in eligible and -5000 <= now - evidence.get(key, -inf) <= self.PACKET_MAX_AGE_MS
                for key in candidate["supporting"]
            )
            >= 2
        ]
        # Keep the original global match flag. Adjacency by itself cannot
        # turn a weak fingerprint into a strong vote.
        return {**estimate, "candidates": local[:2]} if local else estimate

    def update(
        self,
        estimate: dict[str, Any],
        now: float,
        evidence: Mapping[str, float],
        model: dict[str, Any],
        eligible: Collection[str] | None = None,
        *,
        native_room: str | None = None,
    ) -> dict[str, Any]:
        """Update using recorded packet stamps and an optional native UKF result."""
        if eligible is None:
            eligible = evidence.keys()
        packet = self._observe_packet(now, evidence, eligible)
        cadence = self._cadence_ms()
        self.HOLD_MS = min(45_000, cadence * (1 + 3 * self.decay_intervals))
        self.CONTEXT_MS = self.HOLD_MS
        connections = model.get("connections", [])
        if self.confirmed_at is not None and now - self.confirmed_at <= self.CONTEXT_MS:
            estimate = self._nearby_estimate(estimate, now, evidence, eligible, connections)
        result = super().update(estimate, now, evidence, model, eligible)
        if self.confirmed is None:
            if now - self._provisional_at > self.HOLD_MS:
                self._provisional_room = None
            if native_room is not None and packet is not None:
                allowed = (
                    self._native_startup_allowed(native_room, estimate["candidates"], now, evidence, eligible)
                    if self._provisional_room is None
                    else self._nearby(self._provisional_room, native_room, connections)
                )
                if allowed:
                    self._provisional_room = native_room
                    self._provisional_at = packet
            if self._provisional_room is not None:
                result["provisionalLocation"] = self._provisional_room
                result["reason"] = "Tentative room from the native fingerprint estimator"
                age = now - self._provisional_at
            else:
                return result
        elif result.get("heldLocation"):
            age = result["lastSeenAge"]
        else:
            return result
        result["placementFreshness"] = round(2 ** (-max(0, age - cadence) / (cadence * self.decay_intervals)), 3)
        return result
