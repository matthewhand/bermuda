"""Walk recorder persistence, evidence identity and calibration isolation."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError, Unauthorized

from custom_components.bermuda.room_tracking import TemporalTracker
from custom_components.bermuda.walk_test import WalkTestRecorder
from tests.test_room_tracking import make_model, SIG


def test_walk_records_independently_and_survives_load(tmp_path, monkeypatch):
    async def scenario():
        clock = [100.0]
        monkeypatch.setattr("custom_components.bermuda.walk_test.monotonic_time_coarse", lambda: clock[0])
        hass = HomeAssistant(str(tmp_path))
        entry = SimpleNamespace(entry_id="walk_contract", async_on_unload=lambda _: None)
        manager = SimpleNamespace(model=make_model(), tracker=TemporalTracker(), last_signature=SIG)
        before = deepcopy(manager.model)
        recorder = WalkTestRecorder(hass, entry, manager)
        recorder.async_register_services()

        async def call(action, **fields):
            return await hass.services.async_call(
                "bermuda", "walk_test", {"action": action, **fields}, blocking=True, return_response=True
            )

        try:
            response = await call("start", label="Trial", setup_confirmed=True)
            identifier = response["walk"]["id"]
            with pytest.raises(HomeAssistantError):
                await call("start")
            for step in range(5):
                # Stable timestamp remains the same evidence across multiple frames.
                recorder.capture(
                    100 + step,
                    {"a": -65, "b": -85},
                    [("a", 100), ("b", 100)],
                    SIG,
                    set(),
                    {"kind": "transition", "location": None},
                )
            response = await call("stop", walk_id=identifier)
            assert len(response["walk"]["frames"]) == 5
            assert response["walk"]["frames"][0]["evidence"] == response["walk"]["frames"][-1]["evidence"]
            assert response["walk"]["model"] == before == manager.model
            response = await call(
                "annotate", walk_id=identifier, expected_revision=0, from_ms=1000, to_ms=3000, room_id="east"
            )
            assert response["walk"]["revision"] == 1
            with pytest.raises(HomeAssistantError):
                await call(
                    "annotate", walk_id=identifier, expected_revision=0, from_ms=3000, to_ms=4000, room_id="west"
                )
            with pytest.raises(HomeAssistantError):
                await call(
                    "annotate", walk_id=identifier, expected_revision=1, from_ms=2000, to_ms=4000, room_id="west"
                )
            restored = WalkTestRecorder(hass, entry, manager)
            await restored.async_load()
            assert restored.walks == recorder.walks
            assert manager.model == before
            await call("delete", walk_id=identifier)
            assert not recorder.walks
        finally:
            await hass.async_stop()

    asyncio.run(scenario())


def test_walk_limits_interruptions_and_permissions(tmp_path, monkeypatch):
    async def scenario():
        monkeypatch.setattr("custom_components.bermuda.walk_test.monotonic_time_coarse", lambda: 100)
        hass = HomeAssistant(str(tmp_path))
        entry = SimpleNamespace(entry_id="walk_limits", async_on_unload=lambda _: None)
        manager = SimpleNamespace(model=make_model(), tracker=TemporalTracker(), last_signature=SIG)
        recorder = WalkTestRecorder(hass, entry, manager)
        recorder.async_register_services()

        async def call(action, context=None, **fields):
            return await hass.services.async_call(
                "bermuda",
                "walk_test",
                {"action": action, **fields},
                context=context,
                blocking=True,
                return_response=True,
            )

        try:

            async def viewer(_):
                return SimpleNamespace(is_admin=False)

            hass.auth = SimpleNamespace(async_get_user=viewer)
            with pytest.raises(Unauthorized):
                await call("start", context=Context(user_id="viewer"))
            await call("start")
            recorder.capture(100, {"a": -65}, [("a", 100)], SIG, set(), {"kind": "unknown"})
            restored = WalkTestRecorder(hass, entry, manager)
            await recorder.store.async_save(recorder._data())
            await restored.async_load()
            assert restored.walks[0]["status"] == "interrupted"
            assert restored.active is None
            recorder.capture(701, {"a": -65}, [("a", 701)], SIG, set(), {"kind": "unknown"})
            assert recorder.active is None
            assert recorder.walks[0]["status"] == "complete"
            await call("start")
            manager.model["revision"] += 1
            recorder.capture(102, {}, [], SIG, set(), {"kind": "unknown"})
            assert recorder.active is None
            assert recorder.walks[-1]["status"] == "configuration_changed"
        finally:
            await hass.async_stop()

    asyncio.run(scenario())
