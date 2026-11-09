"""Behavioural tests for robust fingerprints, body poses and non-teleporting state."""

from copy import deepcopy

import pytest

from custom_components.bermuda.room_tracking import FingerprintEstimator, TemporalTracker, shortest_path
from custom_components.bermuda.room_tracking_hybrid import AdaptiveRoomTracker

SIG = "beacon|high|a,b,c|layout:1"


def sample(room, a=None, b=None, c=None):
    return {
        "roomId": room,
        "signature": SIG,
        "signals": {
            key: None if value is None else {"median": value, "mad": 1, "count": 5}
            for key, value in zip(("a", "b", "c"), (a, b, c), strict=True)
        },
    }


@pytest.fixture
def model():
    return make_model()


def make_model():
    return {
        "version": 1,
        "revision": 0,
        "rooms": [{"id": room, "label": room.title()} for room in ("east", "west", "corridor")],
        "config": {
            "device": {"id": "beacon", "transmitterEntity": "sensor.test_ble"},
            "receivers": [{"key": key, "name": key} for key in ("a", "b", "c")],
        },
        "receiverLayoutVersion": 1,
        "connections": [["east", "corridor"], ["corridor", "west"]],
        "samples": [sample("east", -65, -85), sample("west", None, -65, -85), sample("corridor", -85, -85, -65)],
        "enabled": True,
    }


def test_complete_signal_pattern_beats_partial_match(model):
    result = FingerprintEstimator().estimate({"a": -65, "b": -85, "c": None}, model, SIG)
    assert result["kind"] == "match"
    assert result["roomId"] == "east"


def test_missing_strong_receiver_counts_against_room_not_as_fake_rssi(model):
    model["samples"] = [sample("east", -65, None, -80), sample("west", None, -78, -73)]
    result = FingerprintEstimator().estimate({"a": None, "b": -78, "c": -73}, model, SIG)
    assert result["kind"] == "match"
    assert result["roomId"] == "west"
    assert all(item["roomId"] != "east" for item in result["candidates"])


def test_known_offline_receiver_is_not_negative_room_evidence(model):
    model["samples"] = [sample("east", -65, None, -80), sample("west", None, -78, -73)]
    result = FingerprintEstimator().estimate({"a": None, "b": None, "c": -80}, model, SIG, {"a"})
    assert any(item["roomId"] == "east" for item in result["candidates"])
    assert result["kind"] == "uncertain"


def test_no_readings_never_produces_a_room(model):
    result = FingerprintEstimator().estimate({"a": None, "b": None, "c": None}, model, SIG)
    assert result["kind"] == "unknown"


def test_identical_room_patterns_remain_uncertain(model):
    model["samples"] = [sample("east", -65, -85, -75), sample("west", -65, -85, -75)]
    assert FingerprintEstimator().estimate({"a": -65, "b": -85, "c": -75}, model, SIG)["kind"] == "uncertain"


def test_one_receiver_does_not_make_a_confident_room_fix(model):
    assert FingerprintEstimator().estimate({"a": -65, "b": None, "c": None}, model, SIG)["kind"] != "match"


def test_body_orientation_prototypes_are_not_averaged_away(model):
    model["samples"].append(sample("corridor", -92, -84, -75))
    for vector in ({"a": -85, "b": -85, "c": -65}, {"a": -92, "b": -84, "c": -75}):
        result = FingerprintEstimator().estimate(vector, model, SIG)
        assert result["roomId"] == "corridor"
        assert result["candidates"][0]["score"] == 0


def test_another_receiver_layout_cannot_reuse_old_training(model):
    assert FingerprintEstimator().estimate({"a": -65}, model, SIG + "-moved")["kind"] == "learning"


def fix(room):
    return {
        "kind": "match",
        "roomId": room,
        "confidence": "possible match",
        "candidates": [
            {"roomId": room, "score": 0, "common": 3, "quality": 0, "contradictions": 0, "supporting": ["a", "b"]}
        ],
    }


def establish(tracker, model):
    assert tracker.update(fix("east"), 0, {"a": 0}, model)["kind"] == "transition"
    assert tracker.update(fix("east"), 3000, {"a": 3000, "b": 2800}, model)["location"] == "east"


def test_hybrid_startup_is_provisional_and_cached_packets_do_not_refresh_it(model):
    tracker = AdaptiveRoomTracker(decay_intervals=2, travel_seconds=4)
    unknown = {"kind": "unknown", "candidates": []}
    result = tracker.update(unknown, 0, {"a": 0}, model, native_room="east")
    assert result["provisionalLocation"] == "east"
    assert result["location"] is None
    assert result["placementFreshness"] == 1
    result = tracker.update(unknown, 12000, {"a": 0}, model, native_room="west")
    assert result["provisionalLocation"] == "east"
    assert result["placementFreshness"] == 0.5
    assert "provisionalLocation" not in tracker.update(unknown, 3_600_000, {"a": 0}, model, native_room="east")


def test_hybrid_decay_tracks_observed_packet_cadence(model):
    tracker = AdaptiveRoomTracker(decay_intervals=2, travel_seconds=4)
    unknown = {"kind": "unknown", "candidates": []}
    for stamp in (0, 2000, 4000, 6000, 8000):
        tracker.update(unknown, stamp, {"a": stamp}, model, native_room="east")
    assert tracker.update(unknown, 14000, {"a": 8000}, model)["placementFreshness"] == 0.5
    assert "provisionalLocation" not in tracker.update(unknown, 23000, {}, model)


@pytest.mark.parametrize(("travel_seconds", "expected"), [(2.5, "corridor"), (4, None)])
def test_hybrid_adjacent_travel_setting_is_independent_of_decay(model, travel_seconds, expected):
    tracker = AdaptiveRoomTracker(decay_intervals=4, travel_seconds=travel_seconds)
    establish(tracker, model)
    adjacent = fix("corridor")
    adjacent["candidates"][0]["quality"] = 3
    for stamp in (3500, 4500, 5500):
        result = tracker.update(adjacent, stamp, {"a": stamp, "b": stamp}, model)
    assert result["location"] == expected


def test_hybrid_considers_a_qualified_neighbor_outside_the_global_top_two(model):
    tracker = AdaptiveRoomTracker(decay_intervals=4, travel_seconds=4)
    establish(tracker, model)
    distant = fix("west")
    nearby = {**fix("corridor")["candidates"][0], "score": 1.5, "quality": 2}
    estimate = {**distant, "allRooms": [*distant["candidates"], nearby]}
    result = tracker.update(estimate, 4000, {"a": 4000, "b": 4000}, model)
    assert result["proposed"] == "corridor"
    assert result["location"] is None


def test_far_room_cannot_teleport_after_burst_of_observations(model):
    tracker = TemporalTracker()
    establish(tracker, model)
    for index in range(100):
        result = tracker.update(fix("west"), 3100 + index, {"a": 3100 + index}, model)
        assert result["kind"] == "transition"
        assert result["location"] is None
        assert result["lastLocation"] == "east"
    gap = tracker.update(fix("west"), 13100, {"a": 13100}, model)
    assert gap["location"] is None and "heldLocation" not in gap
    assert tracker.update(fix("west"), 14000, {"a": 14000}, model)["location"] is None
    result = tracker.update(fix("west"), 24000, {"a": 24000}, model)
    assert result["location"] == "west"
    assert result["placement"] == "reacquired"


def test_repeated_render_of_one_observation_does_not_confirm_location(model):
    tracker = TemporalTracker()
    for now in range(0, 10000, 1000):
        assert tracker.update(fix("east"), now, {"a": 1}, model)["kind"] == "transition"


def test_loss_of_reception_preserves_only_last_known_location(model):
    tracker = TemporalTracker()
    establish(tracker, model)
    result = tracker.update(
        {"kind": "unknown", "reason": "No fresh phone readings", "candidates": []}, 35000, {}, model
    )
    assert result["kind"] == "unknown"
    assert result["location"] is None
    assert result["lastLocation"] == "east"


def test_drawing_coordinates_do_not_affect_fingerprints(model):
    before = FingerprintEstimator().estimate({"a": -65, "b": -85, "c": None}, model, SIG)
    changed = deepcopy(model)
    for room in changed["rooms"]:
        room.update(x=9999, y=-9999, w=1, h=10000)
    assert FingerprintEstimator().estimate({"a": -65, "b": -85, "c": None}, changed, SIG) == before


def test_disconnected_rooms_are_not_treated_as_adjacent(model):
    assert shortest_path("east", "west", model["connections"]) == 2
    assert shortest_path("east", "unknown", model["connections"]) == float("inf")


def test_packet_must_corroborate_the_candidate_after_dwell(model):
    tracker = TemporalTracker()
    establish(tracker, model)
    assert tracker.update(fix("corridor"), 3100, {"a": 3100}, model)["location"] is None
    assert tracker.update(fix("corridor"), 3200, {"a": 3200}, model)["location"] is None
    assert tracker.update(fix("corridor"), 9100, {"a": 3200}, model)["location"] is None
    assert tracker.update(fix("corridor"), 10000, {"a": 9099}, model)["location"] is None
    assert tracker.update(fix("corridor"), 11000, {"a": 11000}, model)["location"] == "corridor"


def test_disconnected_room_still_needs_twenty_seconds_of_travel(model):
    tracker = TemporalTracker()
    model["connections"] = []
    establish(tracker, model)
    for now in (3100, 11500, 12400, 22400):
        assert tracker.update(fix("west"), now, {"a": now}, model)["location"] is None
    assert tracker.update(fix("west"), 23400, {"a": 23400}, model)["location"] == "west"


def test_unconfigured_receiver_keys_do_not_dilute_loss(model):
    model["samples"] = [sample("east", -80, -81, -82), sample("west", -65, -65, -65)]
    current = {"a": -95, "b": -96, "c": -97}
    estimator = FingerprintEstimator()
    baseline = estimator.estimate(current, model, SIG)
    assert baseline["kind"] == "unknown"
    for index in range(35):
        prototype = deepcopy(model["samples"][0])
        prototype["signals"].update({f"other_{index}_{key}": None for key in range(13)})
        model["samples"].append(prototype)
    assert estimator.estimate(current, model, SIG) == baseline


def test_short_gap_holds_last_room_without_renewing_from_cached_packets(model):
    tracker = TemporalTracker()
    establish(tracker, model)
    missing = {"kind": "unknown", "candidates": []}

    gap = tracker.update(missing, 5000, {}, model)
    assert gap["location"] is None
    assert gap["heldLocation"] == "east"
    cached = tracker.update(fix("east"), 8500, {"a": 3000, "b": 2800}, model)
    assert cached["heldLocation"] == "east"
    expired = tracker.update(missing, 11001, {}, model)
    assert expired["location"] is None
    assert "heldLocation" not in expired
    assert expired["lastLocation"] == "east"


def test_connected_room_accumulates_support_through_ambiguous_frames(model):
    tracker = TemporalTracker()
    establish(tracker, model)
    corridor = {"roomId": "corridor", "score": 0.6, "common": 2, "quality": 2, "supporting": ["a", "b"]}
    rival = {"roomId": "east", "score": 0.95, "common": 2, "quality": 2, "supporting": ["a", "b"]}
    narrow_lead = {"kind": "uncertain", "candidates": [corridor, rival]}
    tie = {"kind": "uncertain", "candidates": [corridor, {**rival, "score": 0.6}]}

    assert tracker.update(narrow_lead, 4000, {"a": 4000}, model)["heldLocation"] == "east"
    tracker.update(tie, 4500, {"a": 4500}, model)
    # A new packet from a receiver outside the fingerprint contributes no vote.
    tracker.update(narrow_lead, 5000, {"a": 4500, "c": 5000}, model)
    assert tracker.update(narrow_lead, 6000, {"a": 6000}, model)["location"] is None
    tracker.update(tie, 6500, {"a": 6500}, model)
    assert tracker.update(narrow_lead, 7000, {"a": 6500, "c": 7000}, model)["location"] is None
    assert tracker.update(narrow_lead, 8000, {"a": 8000, "b": 2800}, model)["location"] is None
    result = tracker.update(narrow_lead, 9000, {"a": 9000, "b": 9000}, model)
    assert result["location"] == "corridor"
    assert result["heldLocation"] is None


def test_three_receivers_sustain_adjacent_room_through_competing_lead(model):
    tracker = TemporalTracker()
    establish(tracker, model)
    corridor = {
        "roomId": "corridor",
        "score": 0.6,
        "common": 3,
        "quality": 3,
        "contradictions": 0,
        "supporting": ["a", "b", "c"],
    }
    west = {"roomId": "west", "score": 0.4, "common": 2, "quality": 2, "contradictions": 0, "supporting": ["a", "b"]}
    first = {"kind": "uncertain", "candidates": [corridor, {**west, "score": 0.65}]}
    later = {"kind": "uncertain", "candidates": [west, corridor]}
    assert tracker.update(first, 5000, {"a": 5000}, model)["location"] is None
    assert tracker.update(later, 6000, {"c": 6000}, model)["location"] is None
    result = tracker.update(later, 7000, {"c": 7000}, model)
    assert result["location"] == "corridor"
    assert result["placement"] == "move"


def test_cold_fix_needs_recent_packets_from_two_matching_receivers(model):
    tracker = TemporalTracker()
    for now in (5000, 8000, 9000):
        result = tracker.update(fix("east"), now, {"a": now, "b": 0, "c": now}, model)
        assert result["location"] is None
    assert tracker.update(fix("east"), 10000, {"a": 10000, "b": 9950}, model)["location"] == "east"
