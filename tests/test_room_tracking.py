"""Behavioural tests for robust fingerprints, body poses and non-teleporting state."""

from copy import deepcopy

import pytest

from custom_components.bermuda.room_tracking import FingerprintEstimator, TemporalTracker, shortest_path

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
        "candidates": [{"roomId": room, "score": 0, "common": 3}],
    }


def establish(tracker, model):
    assert tracker.update(fix("east"), 0, (1,), model)["kind"] == "transition"
    assert tracker.update(fix("east"), 3000, (2,), model)["location"] == "east"


def test_far_room_cannot_teleport_after_burst_of_observations(model):
    tracker = TemporalTracker()
    establish(tracker, model)
    for index in range(100):
        result = tracker.update(fix("west"), 3100 + index, (index + 3,), model)
        assert result["kind"] == "transition"
        assert result["location"] is None
        assert result["lastLocation"] == "east"
    assert tracker.update(fix("west"), 13100, (200,), model)["location"] == "west"


def test_repeated_render_of_one_observation_does_not_confirm_location(model):
    tracker = TemporalTracker()
    for now in range(0, 10000, 1000):
        assert tracker.update(fix("east"), now, (1,), model)["kind"] == "transition"


def test_loss_of_reception_preserves_only_last_known_location(model):
    tracker = TemporalTracker()
    establish(tracker, model)
    result = tracker.update(
        {"kind": "unknown", "reason": "No fresh phone readings", "candidates": []}, 35000, (None,), model
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
