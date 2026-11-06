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
        if not any(_finite(value) for key, value in current.items() if key not in offline):
            return {"kind": "unknown", "reason": "No fresh phone readings", "candidates": []}
        if len(trained) < 2:
            reason = "Record another room to compare" if trained else "No room samples yet"
            return {"kind": "learning", "reason": reason, "candidates": []}
        keys = set(current)
        for sample in samples:
            keys.update(sample["signals"])
        noise = {}
        weights = {}
        manual = {receiver["key"]: receiver.get("weight", 1) for receiver in model["config"]["receivers"]}
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
            weights[key] = (0.5 + min(1.5, spread / (noise[key] + 2))) * max(0.1, min(5, manual.get(key, 1)))
        denominator = sum(weight for key, weight in weights.items() if key not in offline) or 1
        by_room = defaultdict(list)
        for sample in samples:
            loss = 0
            common = quality = contradictions = 0
            for key in keys - offline:
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
        self.evidence = None

    def update(self, estimate: dict, now: float, evidence: tuple, model: dict) -> dict:
        """Update using monotonic milliseconds; never label the last room as current."""
        groups = model.get("roomGroups", [])
        edges = model.get("connections", [])
        candidate_ids = [candidate["roomId"] for candidate in estimate.get("candidates", [])]
        target = estimate.get("roomId") if estimate["kind"] == "match" else None
        if target is None and estimate["kind"] == "uncertain" and len(candidate_ids) >= 2:
            group = next((group for group in groups if set(candidate_ids) <= set(group["rooms"])), None)
            if group:
                target = "group:" + group["id"]

        def members(location):
            if not location:
                return []
            if location.startswith("group:"):
                return next((group["rooms"] for group in groups if "group:" + group["id"] == location), [])
            return [location]

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
                nearby = [
                    candidate
                    for candidate in nearby
                    if any(shortest_path(room, candidate["roomId"], edges) <= 1 for room in members(previous))
                ]
            kind = estimate["kind"] if estimate["kind"] in {"learning", "unknown"} else "uncertain"
            return {**base, "kind": kind, "location": None, "candidates": nearby}
        if target == previous:
            if evidence != self.evidence:
                self.confirmed_at = now
                self.evidence = evidence
            self.pending = None
            return {
                **base,
                "kind": "stable",
                "location": target,
                "members": members(target),
                "reason": "Area known; zone uncertain" if target.startswith("group:") else estimate.get("confidence"),
            }
        if self.pending != target:
            self.pending = target
            self.pending_since = now
            self.count = 0
            self.evidence = None
        if evidence != self.evidence:
            self.evidence = evidence
            self.count += 1
        distance = (
            min((shortest_path(a, b, edges) for a in members(previous) for b in members(target)), default=math.inf)
            if previous
            else 0
        )
        dwell = (6000 if distance <= 1 else 10000) if previous else 3000
        travel = (max(4000, distance * 3000) if math.isfinite(distance) else 20000) if previous else 0
        if self.count >= 2 and now - self.pending_since >= dwell and elapsed >= travel:
            self.confirmed = target
            self.confirmed_at = now
            self.pending = None
            return {
                **base,
                "kind": "stable",
                "location": target,
                "members": members(target),
                "reason": "Area known; zone uncertain" if target.startswith("group:") else estimate.get("confidence"),
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
