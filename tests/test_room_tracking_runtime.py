"""Persistence, permission and live-observation contract tests using real HA services."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError, Unauthorized

from custom_components.bermuda.bermuda_device import BermudaDevice
from custom_components.bermuda.room_tracking_runtime import RoomTrackingManager, validate_room_model
from tests.test_room_tracking import make_model, sample


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
    # Device observations live in attributes, while the inherited dict is empty.
    # Exercise the real object's false truthiness, rather than a namespace stub.
    phone = BermudaDevice.__new__(BermudaDevice)
    phone.adverts = adverts
    phone.area_name = "Original scanner area"
    assert not phone
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


def test_non_admin_cannot_read_or_save_model(tmp_path):
    async def scenario():
        hass = HomeAssistant(str(tmp_path))
        manager, _, _, _ = setup_manager(hass, [100])

        async def get_user(_):
            return SimpleNamespace(is_admin=False)

        hass.auth = SimpleNamespace(async_get_user=get_user)
        try:
            context = Context(user_id="viewer")
            with pytest.raises(Unauthorized):
                await hass.services.async_call(
                    "bermuda", "get_room_model", {}, blocking=True, return_response=True, context=context
                )
            with pytest.raises(Unauthorized):
                await save(hass, make_model(), context=context)
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
    ],
)
def test_model_contract_rejects_invalid_input(mutation):
    value = make_model()
    mutation(value)
    with pytest.raises((ValueError, TypeError)):
        validate_room_model(value)


def test_failed_setup_does_not_register_room_model_services(tmp_path, monkeypatch):
    from homeassistant.exceptions import ConfigEntryNotReady

    from custom_components import bermuda

    async def scenario():
        hass = HomeAssistant(str(tmp_path))
        entry = SimpleNamespace()
        calls = []

        async def load():
            calls.append("load")

        async def refresh():
            calls.append("refresh")

        manager = SimpleNamespace(async_load=load, async_register_services=lambda: calls.append("register"))
        coordinator = SimpleNamespace(room_tracking=manager, async_refresh=refresh, last_update_success=False)
        monkeypatch.setattr(bermuda, "BermudaDataUpdateCoordinator", lambda *_: coordinator)
        try:
            with pytest.raises(ConfigEntryNotReady):
                await bermuda.async_setup_entry(hass, entry)
            assert calls == ["load", "refresh"]
        finally:
            await hass.async_stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("offline", [False, True])
def test_expired_or_offline_receivers_cannot_confirm_room(tmp_path, monkeypatch, offline):
    async def scenario():
        clock = [100.0]
        monkeypatch.setattr("custom_components.bermuda.room_tracking_runtime.monotonic_time_coarse", lambda: clock[0])
        hass = HomeAssistant(str(tmp_path))
        manager, phone, scanners, _ = setup_manager(hass, clock)
        model = make_model()
        model["samples"] = [sample("east", -65, -85, -100), sample("west", -95, -65, -60)]
        advert = phone.adverts["c"]
        advert.rssi, advert.rssi_distance, advert.stamp = -100, 1, 100 if offline else 71
        scanners["c"]._hascanner.scanning = not offline
        try:
            assert (await save(hass, model))["result"]["kind"] == "transition"
            clock[0] = 104
            if offline:
                advert.stamp = 104
            manager.update()
            assert manager.result["kind"] == "transition"
            if offline:
                scanners["c"]._hascanner.scanning = True
                manager.update()
                assert manager.result["kind"] == "transition"
            phone.adverts["a"].stamp = 104
            manager.update()
            assert manager.result["location"] == "east"
        finally:
            await hass.async_stop()

    asyncio.run(scenario())


def test_only_current_layout_rejects_unconfigured_recording_sources():
    model = make_model()
    recording = model["samples"][0]
    recording["signals"]["unconfigured"] = None
    with pytest.raises(ValueError, match="configured layout"):
        validate_room_model(model)
    model["receiverLayoutVersion"] += 1
    assert "unconfigured" in validate_room_model(model)["samples"][0]["signals"]


def test_receiver_reassignment_requires_new_layout_version(tmp_path, monkeypatch):
    async def scenario():
        clock = [100.0]
        monkeypatch.setattr("custom_components.bermuda.room_tracking_runtime.monotonic_time_coarse", lambda: clock[0])
        hass = HomeAssistant(str(tmp_path))
        manager, _, _, _ = setup_manager(hass, clock)
        model = make_model()
        try:
            assert (await save(hass, model))["result"]["kind"] == "transition"
            model["config"]["receivers"][0]["roomId"] = "west"
            with pytest.raises(HomeAssistantError, match="Increase the receiver layout"):
                await save(hass, model, revision=1)
            assert manager.model["revision"] == 1
            model["receiverLayoutVersion"] += 1
            updated = await save(hass, model, revision=1)
            assert updated["result"]["kind"] == "learning"
            assert updated["model"]["samples"] == model["samples"]
            model["receiverLayoutVersion"] -= 1
            with pytest.raises(HomeAssistantError, match="Increase the receiver layout"):
                await save(hass, model, revision=2)
        finally:
            await hass.async_stop()

    asyncio.run(scenario())
