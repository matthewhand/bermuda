"""
Experimental room estimators and elapsed-time tracking, independent of HA and geometry.

Signal prototypes may represent different positions and body orientations in one room.
Scores are robust comparison losses, not calibrated probabilities or metre distances.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from statistics import median
from typing import Any, NamedTuple, Protocol


class RoomEstimator(Protocol):
    """Interface for room estimators; physical coordinate modes may be added later."""

    def estimate(self, current: dict, model: dict, signature: str, unavailable: set[str] | None = None) -> dict:
        """Return candidate rooms, quality and the reason for uncertainty."""
        ...


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _huber(value: float) -> float:
    return value * value if abs(value) <= 2 else 4 * abs(value) - 4


def _strength(value: float) -> float:
    return max(0.15, min(1.4, (value + 100) / 25))


class FingerprintEstimator:
    """Compare complete weighted signal patterns against multiple labelled prototypes."""

    def estimate(self, current: dict, model: dict, signature: str, unavailable: set[str] | None = None) -> dict:
        rooms = {room["id"] for room in model["rooms"]}
        samples = [
            sample for sample in model["samples"] if sample["roomId"] in rooms and sample["signature"] == signature
        ]
        trained = {sample["roomId"] for sample in samples}
        offline = unavailable or set()
        manual = {receiver["key"].lower(): receiver.get("weight", 1) for receiver in model["config"]["receivers"]}
        keys = manual.keys() - offline
        if not any(_finite(current.get(key)) for key in keys):
            return {"kind": "unknown", "reason": "No fresh phone readings", "candidates": []}
        if len(trained) < 2:
            reason = "Record another room to compare" if trained else "No room samples yet"
            return {"kind": "learning", "reason": reason, "candidates": []}
        noise = {}
        weights = {}
        for key in keys:
            values = [sample["signals"][key] for sample in samples if sample["signals"].get(key)]
            deviations = [1.4826 * value["mad"] for value in values]
            noise[key] = max(4, min(8, median(deviations) if deviations else 4))
            centres = []
            for room in trained:
                room_values = [
                    sample["signals"][key]["median"]
                    for sample in samples
                    if sample["roomId"] == room and sample["signals"].get(key)
                ]
                if room_values:
                    centres.append(median(room_values))
            mean = sum(centres) / len(centres) if centres else 0
            spread = math.sqrt(sum((value - mean) ** 2 for value in centres) / len(centres)) if centres else 0
            weights[key] = (0.5 + min(1.5, spread / (noise[key] + 2))) * max(0.1, min(5, manual[key]))
        denominator = sum(weights.values()) or 1
        by_room = defaultdict(list)
        for sample in samples:
            loss = 0
            common = quality = contradictions = 0
            supporting = []
            for key in keys:
                reference = sample["signals"].get(key)
                live = _finite(current.get(key))
                if reference and live:
                    loss += weights[key] * _huber((current[key] - reference["median"]) / noise[key])
                    common += 1
                    supporting.append(key)
                    quality += reference["count"] >= 2
                    if (
                        reference["median"] > -75
                        and current[key] < reference["median"] - 18
                        and reference["count"] >= 2
                    ):
                        contradictions += 1
                elif reference:
                    reliability = min(1, 0.5 + 0.2 * reference["count"])
                    loss += weights[key] * 3.5 * _strength(reference["median"]) * reliability
                    if reference["median"] > -75 and reference["count"] >= 2:
                        contradictions += 1
                elif live:
                    loss += weights[key] * 2 * _strength(current[key])
            if common:
                by_room[sample["roomId"]].append(
                    {
                        "roomId": sample["roomId"],
                        "score": math.sqrt(loss / denominator) + 1.5 * contradictions,
                        "common": common,
                        "quality": quality,
                        "contradictions": contradictions,
                        "supporting": sorted(supporting),
                    }
                )
        candidates = sorted(
            (min(items, key=lambda item: item["score"]) for items in by_room.values()), key=lambda item: item["score"]
        )
        if not candidates:
            return {"kind": "unknown", "reason": "No comparable samples", "candidates": []}
        if any(candidate["contradictions"] == 0 for candidate in candidates):
            candidates = [candidate for candidate in candidates if candidate["contradictions"] == 0]
        first = candidates[0]
        if first["score"] > 3.25:
            return {
                "kind": "unknown",
                "reason": "No similar recorded signal pattern",
                "candidates": [],
                "weights": weights,
            }
        result = {"candidates": candidates[:2], "weights": weights}
        if first["common"] < 2 or (len(candidates) > 1 and candidates[1]["score"] - first["score"] < 0.75):
            return {
                **result,
                "kind": "uncertain",
                "reason": "Limited receiver coverage" if first["common"] < 2 else "Room patterns overlap",
            }
        return {
            **result,
            "kind": "match",
            "roomId": first["roomId"],
            "confidence": "good match"
            if first["common"] >= 3 and first["quality"] >= 2 and first["score"] < 1
            else "possible match",
        }


def shortest_path(start: str, target: str, edges: list) -> float:
    """Return graph hops without interpreting drawing coordinates as distances."""
    if start == target:
        return 0
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        room, distance = queue.popleft()
        for edge in edges:
            if room not in edge:
                continue
            other = edge[1] if edge[0] == room else edge[0]
            if other == target:
                return distance + 1
            if other not in visited:
                visited.add(other)
                queue.append((other, distance + 1))
    return math.inf


class RoomVote(NamedTuple):
    """One independent packet observation supporting a room."""

    stamp: float
    room: str
    margin: float
    strong: bool


class TemporalTracker:
    """Confirm rooms from fresh receiver observations and retain short gaps."""

    HOLD_MS = 8000
    CONTEXT_MS = 30000
    VOTE_WINDOW_MS = 12000
    VOTE_SPACING_MS = 800
    PACKET_MAX_AGE_MS = 4000
    ADJACENT_WINDOW_MS = 8000
    ADJACENT_DWELL_MS = 1700
    MIN_TRAVEL_MS = 4000

    def __init__(self) -> None:
        self.confirmed = None
        self.confirmed_at = None
        self.receiver_stamps: dict[str, float] = {}
        self.votes: deque[RoomVote] = deque()
        self.adjacent_votes: dict[str, deque[tuple[float, bool]]] = defaultdict(deque)
        self.was_placed = False
        self.unplaced_since = -math.inf

    def update(
        self, estimate: dict, now: float, evidence: dict[str, float], model: dict, eligible: set[str] | None = None
    ) -> dict:
        """Advance from receipt times. Cached and excluded packets never count twice."""
        recent = set()
        fresh = {}
        for key, stamp in evidence.items():
            if not _finite(stamp) or stamp > now + 5000:
                continue
            if (eligible is None or key in eligible) and now - stamp <= self.PACKET_MAX_AGE_MS:
                recent.add(key)
            if stamp <= self.receiver_stamps.get(key, -math.inf):
                continue
            self.receiver_stamps[key] = stamp
            if key in recent:
                fresh[key] = min(now, stamp)

        elapsed = math.inf if self.confirmed_at is None else now - self.confirmed_at
        previous = self.confirmed if elapsed <= self.CONTEXT_MS else None
        candidates = estimate.get("candidates", [])
        base = {
            "lastLocation": self.confirmed,
            "lastSeenAge": None if self.confirmed_at is None else elapsed,
            "candidates": candidates,
            "reason": estimate.get("reason"),
        }
        if elapsed <= self.HOLD_MS:
            base["heldLocation"] = self.confirmed

        while self.votes and now - self.votes[0].stamp > self.VOTE_WINDOW_MS:
            self.votes.popleft()
        edges = model.get("connections", [])
        proposed = None
        if candidates:
            ranked = self._rank(candidates, previous, elapsed, edges)
            cost, distance, leader = ranked[0]
            room = leader["roomId"]
            gap = min(1.0, ranked[1][0] - cost) if len(ranked) > 1 else 1.0
            strong = estimate.get("roomId") == room and estimate["kind"] == "match"
            corroborated = (
                distance <= 1
                and leader["common"] >= 2
                and leader.get("quality", 0) >= 2
                and not leader.get("contradictions")
                and leader["score"] <= 2.2
                and gap >= 0.10
            )
            supporting = leader["supporting"]
            packet = max((fresh[key] for key in supporting if key in fresh), default=None)
            if (strong or corroborated) and packet is not None:
                if not self.votes or packet - self.votes[-1].stamp >= self.VOTE_SPACING_MS:
                    self.votes.append(RoomVote(packet, room, gap, strong))
                    can_acquire = previous is not None or (strong and len(recent.intersection(supporting)) >= 2)
                    if (
                        can_acquire
                        and self._move_allowed(
                            leader, candidates, recent, distance=distance, previous=previous, packet=packet, now=now
                        )
                        and self._can_confirm(room, leader["common"], distance, previous, packet=packet, now=now)
                    ):
                        return self._confirm(base, room, packet, now, "Consistent BLE evidence")
            if room != previous and (strong or corroborated):
                proposed = room

        if previous and candidates and (fix := self._adjacent_fix(candidates, fresh, now, edges, previous)):
            room, packet = fix
            return self._confirm(base, room, packet, now, "Repeated adjacent BLE evidence")

        placed = bool(base.get("heldLocation"))
        if self.was_placed and not placed:
            self.unplaced_since = now
        self.was_placed = placed
        if proposed is not None:
            return {
                **base,
                "kind": "transition",
                "location": None,
                "proposed": proposed,
                "reason": "Waiting for consistent readings and plausible travel time",
            }

        kind = estimate["kind"] if estimate["kind"] in {"learning", "unknown"} else "uncertain"
        if previous and elapsed < 15000:
            base["candidates"] = [
                candidate for candidate in candidates if shortest_path(previous, candidate["roomId"], edges) <= 1
            ]
        return {**base, "kind": kind, "location": None}

    def _confirm(self, base, room, packet, now, reason):
        if self.confirmed is None:
            placement = "initial"
        elif not self.was_placed:
            placement = "reacquired"
        elif room != self.confirmed:
            placement = "move"
        else:
            placement = "continued"
        if room != self.confirmed:
            self.votes.clear()
        if placement != "continued":
            self.adjacent_votes.clear()
        self.confirmed = room
        self.confirmed_at = packet
        self.was_placed = True
        return {
            **base,
            "kind": "stable",
            "placement": placement,
            "location": room,
            "heldLocation": None,
            "lastLocation": room,
            "lastSeenAge": now - packet,
            "reason": reason,
        }

    def _adjacent_fix(self, candidates, fresh, now, edges, previous):
        neighbors = {right if left == previous else left for left, right in edges if previous in (left, right)}
        leader = candidates[0]
        for candidate in candidates:
            room = candidate["roomId"]
            if room not in neighbors or candidate["common"] < 3 or candidate["quality"] < 2:
                continue
            if candidate["contradictions"] or candidate["score"] > 2.2:
                continue
            leads = candidate is leader
            if not leads and (
                leader["roomId"] in neighbors
                or leader["roomId"] == previous
                or candidate["common"] <= leader["common"]
                or candidate["score"] - leader["score"] > 1.1
            ):
                continue
            packet = max((fresh[key] for key in candidate["supporting"] if key in fresh), default=None)
            if packet is None:
                continue
            votes = self.adjacent_votes[room]
            while votes and now - votes[0][0] > self.ADJACENT_WINDOW_MS:
                votes.popleft()
            if votes and packet - votes[-1][0] < self.VOTE_SPACING_MS:
                continue
            votes.append((packet, leads))
            # A runner-up can sustain recent evidence, but must have led at least once.
            if (
                len(votes) >= 3
                and packet - votes[0][0] >= self.ADJACENT_DWELL_MS
                and any(led for _, led in votes)
                and packet - self.confirmed_at >= self.MIN_TRAVEL_MS
            ):
                return room, packet
        return None

    @staticmethod
    def _rank(candidates, previous, elapsed, edges):
        ranked = []
        for candidate in candidates:
            distance = shortest_path(previous, candidate["roomId"], edges) if previous else 0
            # Prefer nearby rooms only for similar fingerprints. Evidence can still
            # confirm a distant room after enough travel time.
            penalty = min(2, max(0, distance - 1)) * 0.35 * max(0.35, 1 - elapsed / 40000)
            ranked.append((candidate["score"] + penalty, distance, candidate))
        return sorted(ranked, key=lambda item: item[0])

    def _move_allowed(self, candidate, candidates, recent_receivers, *, distance, previous, packet, now):
        room = candidate["roomId"]
        overlap = candidate["common"]
        if (self.was_placed and room != self.confirmed and (previous is None or distance > 1)) or (
            self.confirmed and previous is None and packet <= self.unplaced_since
        ):
            return False
        if room != self.confirmed and distance > 1:
            support = [vote for vote in self.votes if vote.room == room and vote.stamp > self.unplaced_since]
            if len(support) < 2 or packet - support[0].stamp < 10000:
                return False
        if overlap == 2 and self.confirmed is None:
            return sum(vote.room == room and vote.strong for vote in self.votes) >= 3
        if overlap == 2 and previous and room != previous:
            raw_gap = candidates[1]["score"] - candidates[0]["score"] if len(candidates) > 1 else 1.0
            if (
                candidate is not candidates[0]
                or raw_gap < 0.25
                or len(recent_receivers.intersection(candidate["supporting"])) < 2
            ):
                return False
            clear = sum(
                vote.room == room and now - vote.stamp <= self.ADJACENT_WINDOW_MS and vote.margin >= 0.30
                for vote in self.votes
            )
            return clear >= 2 and self.votes[-1].margin >= 0.30
        return True

    def _can_confirm(self, room, overlap, distance, previous, *, packet, now):
        if room == previous and self.votes[-1].strong:
            return True
        travel = (max(4000, distance * 3000) if math.isfinite(distance) else 20000) if previous else 0
        supporting = [vote for vote in self.votes if vote.room == room]
        span = packet - supporting[0].stamp
        if previous and room != previous and max(packet - self.confirmed_at, span) < travel:
            return False
        if len(supporting) >= 2 and all(vote.room == room and vote.strong for vote in self.votes):
            dwell = (6000 if distance <= 1 else 10000) if previous else 3000
            if span >= dwell:
                return True
        if distance > 1:
            return False
        sparse = previous is None and overlap < 3
        recent = self.votes if sparse else [vote for vote in self.votes if now - vote.stamp <= 8000]
        supporting = [vote for vote in recent if vote.room == room]
        opposition = sum(vote.margin for vote in recent if vote.room != room)
        strength = sum(vote.margin for vote in supporting)
        dwell = 8000 if sparse else 4000 if previous is None else 2400
        return (
            len(supporting) >= (5 if sparse else 3)
            and packet - supporting[0].stamp >= dwell
            and strength >= max(0.70, opposition + 0.22)
            and (not sparse or opposition == 0)
        )


ESTIMATORS = {"fingerprint": FingerprintEstimator}
