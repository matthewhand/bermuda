"""Persistence, permission and live-observation contract tests using real HA services."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError, Unauthorized

from custom_components.bermuda.room_tracking_runtime import RoomTrackingManager, validate_room_model
from tests.test_room_tracking import make_model


def setup_manager(hass, clock):
    hass.states.async_set("sensor.test_ble", "Transmitting", {"Transmitting power": "high"})
    scanners = {
        key: SimpleNamespace(last_seen=clock[0], _hascanner=SimpleNamespace(scanning=True)) for key in ("a", "b", "c")
    }
    adverts = {
        key: SimpleNamespace(
            scanner_address=key, stamp=clock[0], rssi=rssi, rssi_distance=1 if rssi is not None else None
        )
        for key, rssi in [("a", -65), ("b", -85), ("c", None)]
    }
    phone = SimpleNamespace(adverts=adverts, area_name="Original scanner area")
    coordinator = SimpleNamespace(
        _get_device=lambda key: phone if key == "beacon" else scanners.get(key), async_update_listeners=lambda: None
    )
    callbacks = []
    entry = SimpleNamespace(entry_id="room_tracking_test", async_on_unload=callbacks.append)
    manager = RoomTrackingManager(hass, entry, coordinator)
    manager.async_register_services()
    return manager, phone, scanners, callbacks


async def save(hass, model, revision=0, context=None):
    return await hass.services.async_call(
        "bermuda",
        "save_room_model",
        {"model": model, "expected_revision": revision},
        blocking=True,
        return_response=True,
        context=context,
    )


def test_store_roundtrip_and_service_cleanup(tmp_path):
    async def scenario():
        hass = HomeAssistant(str(tmp_path))
        manager, _, _, callbacks = setup_manager(hass, [100])
        try:
            result = await save(hass, make_model())
            assert result["model"]["revision"] == 1
            other = RoomTrackingManager(hass, manager.entry, manager.coordinator)
            await other.async_load()
            assert other.model == manager.model
            for callback in callbacks:
                callback()
            assert not hass.services.has_service("bermuda", "save_room_model")
        finally:
            await hass.async_stop()

    asyncio.run(scenario())


def test_concurrent_saves_require_matching_revision(tmp_path):
    async def scenario():
        hass = HomeAssistant(str(tmp_path))
        manager, _, _, _ = setup_manager(hass, [100])
        original_save = manager.store.async_save

        async def slow_save(value):
            await asyncio.sleep(0.02)
            await original_save(value)

        manager.store.async_save = slow_save
        try:
            results = await asyncio.gather(save(hass, make_model()), save(hass, make_model()), return_exceptions=True)
            assert sum(isinstance(result, dict) for result in results) == 1
            assert sum(isinstance(result, HomeAssistantError) for result in results) == 1
            assert manager.model["revision"] == 1
        finally:
            await hass.async_stop()

    asyncio.run(scenario())


def test_non_admin_cannot_save_model(tmp_path):
    async def scenario():
        hass = HomeAssistant(str(tmp_path))
        manager, _, _, _ = setup_manager(hass, [100])

        async def get_user(_):
            return SimpleNamespace(is_admin=False)

        hass.auth = SimpleNamespace(async_get_user=get_user)
        try:
            with pytest.raises(Unauthorized):
                await save(hass, make_model(), context=Context(user_id="viewer"))
            assert manager.model is None
        finally:
            await hass.async_stop()

    asyncio.run(scenario())


def test_manager_updates_do_not_change_existing_area(tmp_path, monkeypatch):
    async def scenario():
        clock = [100.0]
        monkeypatch.setattr("custom_components.bermuda.room_tracking_runtime.monotonic_time_coarse", lambda: clock[0])
        hass = HomeAssistant(str(tmp_path))
        manager, phone, _, _ = setup_manager(hass, clock)
        try:
            await save(hass, make_model())
            clock[0] = 104
            for advert in phone.adverts.values():
                advert.stamp = 104
            manager.update()
            assert manager.result["location"] == "east"
            assert phone.area_name == "Original scanner area"
            clock[0] = 150
            manager.update()
            assert manager.result["kind"] == "unknown"
            assert manager.result["location"] is None
            assert manager.result["lastLocation"] == "east"
            assert phone.area_name == "Original scanner area"
        finally:
            await hass.async_stop()

    asyncio.run(scenario())


def test_invalid_save_preserves_existing_model(tmp_path):
    async def scenario():
        hass = HomeAssistant(str(tmp_path))
        manager, _, _, _ = setup_manager(hass, [100])
        try:
            await save(hass, make_model())
            before = deepcopy(manager.model)
            invalid = make_model()
            invalid["samples"][0]["signals"]["a"]["median"] = float("nan")
            with pytest.raises(HomeAssistantError):
                await save(hass, invalid, revision=1)
            assert manager.model == before
        finally:
            await hass.async_stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(config=[]),
        lambda m: m["config"].update(receivers=["bad"]),
        lambda m: m["config"]["receivers"].append(dict(m["config"]["receivers"][0])),
        lambda m: m.update(estimator="unimplemented"),
        lambda m: m.update(roomGroups=[{"id": "bad", "rooms": ["missing"]}]),
    ],
)
def test_model_contract_rejects_invalid_input(mutation):
    value = make_model()
    mutation(value)
    with pytest.raises((ValueError, TypeError)):
        validate_room_model(value)
