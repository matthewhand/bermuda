"""
Experimental room estimators and elapsed-time tracking, independent of HA and geometry.

Signal prototypes may represent different positions and body orientations in one room.
Scores are robust comparison losses, not calibrated probabilities or metre distances.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from statistics import median
from typing import Any, Protocol


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
            for key in keys:
                reference = sample["signals"].get(key)
                live = _finite(current.get(key))
                if reference and live:
                    loss += weights[key] * _huber((current[key] - reference["median"]) / noise[key])
                    common += 1
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


class TemporalTracker:
    """Require fresh evidence, elapsed dwell time and plausible graph travel time."""

    def __init__(self) -> None:
        self.confirmed = None
        self.confirmed_at = None
        self.pending = None
        self.pending_since = 0
        self.count = 0
        self.receiver_stamps: dict[str, float] = {}

    def update(
        self, estimate: dict, now: float, evidence: dict[str, float], model: dict, eligible: set[str] | None = None
    ) -> dict:
        """Track receipt times in monotonic milliseconds, counting only eligible receivers."""
        newest = max(
            (
                stamp
                for key, stamp in evidence.items()
                if (eligible is None or key in eligible) and stamp > self.receiver_stamps.get(key, -math.inf)
            ),
            default=None,
        )
        # Remember excluded observations too, so going back online cannot replay them.
        for key, stamp in evidence.items():
            self.receiver_stamps[key] = max(stamp, self.receiver_stamps.get(key, -math.inf))
        edges = model.get("connections", [])
        target = estimate.get("roomId") if estimate["kind"] == "match" else None

        elapsed = math.inf if self.confirmed_at is None else now - self.confirmed_at
        previous = self.confirmed if elapsed <= 30000 else None
        base = {
            "lastLocation": self.confirmed,
            "lastSeenAge": None if self.confirmed_at is None else elapsed,
            "candidates": estimate.get("candidates", []),
            "reason": estimate.get("reason"),
        }
        if target is None:
            self.pending = None
            self.count = 0
            nearby = base["candidates"]
            if previous and elapsed < 15000:
                nearby = [candidate for candidate in nearby if shortest_path(previous, candidate["roomId"], edges) <= 1]
            kind = estimate["kind"] if estimate["kind"] in {"learning", "unknown"} else "uncertain"
            return {**base, "kind": kind, "location": None, "candidates": nearby}
        if target == previous:
            if newest is not None:
                self.confirmed_at = now
            self.pending = None
            return {
                **base,
                "kind": "stable",
                "location": target,
                "reason": estimate.get("confidence"),
            }
        if self.pending != target:
            self.pending = target
            self.pending_since = now
            self.count = 0
        if newest is not None:
            self.count += 1
        distance = shortest_path(previous, target, edges) if previous else 0
        dwell = (6000 if distance <= 1 else 10000) if previous else 3000
        travel = (max(4000, distance * 3000) if math.isfinite(distance) else 20000) if previous else 0
        if (
            newest is not None
            and self.count >= 2
            and min(now, newest) - self.pending_since >= dwell
            and (previous is None or min(now, newest) - self.confirmed_at >= travel)
        ):
            self.confirmed = target
            self.confirmed_at = now
            self.pending = None
            return {
                **base,
                "kind": "stable",
                "location": target,
                "reason": estimate.get("confidence"),
            }
        return {
            **base,
            "kind": "transition",
            "location": None,
            "proposed": target,
            "remainingMs": max(0, dwell - (now - self.pending_since), travel - elapsed),
            "reason": "Waiting for consistent readings and plausible travel time",
        }


ESTIMATORS = {"fingerprint": FingerprintEstimator}
