"""Exercise room model services through Core's WebSocket and config-entry lifecycle."""

import asyncio
import json

import pytest

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import Context
from homeassistant.exceptions import HomeAssistantError

from tests.test_room_tracking import make_model
from tests.test_room_tracking_runtime import save


async def test_ws_and_lifecycle(hass, setup_bermuda_entry, hass_ws_client, hass_read_only_access_token):
    entry = setup_bermuda_entry
    manager = entry.runtime_data.coordinator.room_tracking
    assert manager.model is None and manager.result["kind"] == "disabled"
    admin = await hass_ws_client()
    viewer = await hass_ws_client(access_token=hass_read_only_access_token)

    async def service(ws, request_id, name, data=None, **extra):
        await ws.send_json(
            {
                "id": request_id,
                "type": "call_service",
                "domain": "bermuda",
                "service": name,
                "service_data": data or {},
                "return_response": True,
                **extra,
            }
        )
        return await ws.receive_json()

    def denied(response):
        assert not response["success"] and response["error"]["message"] == "Unauthorized"

    denied(await service(viewer, 1, "get_room_model"))
    model = make_model()
    model["enabled"] = False
    write = {"model": model, "expected_revision": 0}
    denied(await service(viewer, 2, "save_room_model", write))
    forged = await service(viewer, 3, "save_room_model", write, context={"user_id": None})
    assert not forged["success"]
    assert manager.model is None
    saved = await service(admin, 1, "save_room_model", write)
    assert saved["success"], saved
    assert saved["result"]["response"]["model"]["revision"] == 1
    admin_model = await service(admin, 2, "get_room_model")
    assert admin_model["result"]["response"]["model"]["revision"] == 1
    internal = await hass.services.async_call(
        "bermuda", "get_room_model", {}, blocking=True, return_response=True, context=Context()
    )
    assert internal["model"]["revision"] == 1

    await viewer.send_json({"id": 4, "type": "get_states"})
    states = await viewer.receive_json()
    learned = next(state for state in states["result"] if "learned_room" in state["entity_id"])
    assert learned["state"] == "disabled"
    sensitive = {"tracked_device", "tracking_result", "unavailable_sources"}
    assert not sensitive.intersection(learned["attributes"])

    enabled = make_model()
    await save(hass, enabled, revision=1, context=Context())
    await viewer.send_json({"id": 5, "type": "get_states"})
    states = await viewer.receive_json()
    learned = next(state for state in states["result"] if "learned_room" in state["entity_id"])
    assert learned["state"] != "disabled"
    assert not sensitive.intersection(learned["attributes"])

    manager.result = {"kind": "transition", "location": None, "heldLocation": "east", "proposed": "corridor"}
    manager.coordinator.async_update_listeners()
    await hass.async_block_till_done()
    held = hass.states.get(learned["entity_id"])
    assert held.state == "East"
    assert held.attributes["held"] is True
    assert not sensitive.intersection(held.attributes)

    manager.result = {"kind": "unknown", "location": None, "lastLocation": "east"}
    manager.coordinator.async_update_listeners()
    await hass.async_block_till_done()
    expired = hass.states.get(learned["entity_id"])
    assert expired.state == "unknown"
    assert expired.attributes["held"] is False
    assert await hass.config_entries.async_reload(entry.entry_id)
    current = entry.runtime_data.coordinator.room_tracking
    assert current is not manager and current.model["revision"] == 2
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert entry.state is ConfigEntryState.NOT_LOADED
    assert not hass.services.has_service("bermuda", "get_room_model")
    assert not hass.services.has_service("bermuda", "save_room_model")


@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_reload_drains_active_save(hass, setup_bermuda_entry, monkeypatch, cancel_caller):
    entry = setup_bermuda_entry
    original = make_model()
    assert (await save(hass, original))["model"]["revision"] == 1
    old = entry.runtime_data.coordinator.room_tracking
    persisted = old.store.async_save
    close = old.async_close
    entered, resume, unloading = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def delayed(model):
        entered.set()
        await resume.wait()
        await persisted(model)

    async def closing():
        unloading.set()
        await close()

    monkeypatch.setattr(old.store, "async_save", delayed)
    monkeypatch.setattr(old, "async_close", closing)
    first = make_model()
    first["rooms"][0]["label"] = "First editor"
    pending = asyncio.create_task(save(hass, first, revision=1))
    await asyncio.wait_for(entered.wait(), 5)
    if cancel_caller:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    reload = asyncio.create_task(hass.config_entries.async_reload(entry.entry_id))
    try:
        await asyncio.wait_for(unloading.wait(), 5)
        assert not reload.done()
    finally:
        resume.set()
        if not cancel_caller:
            assert (await asyncio.wait_for(pending, 5))["model"]["revision"] == 2
        assert await asyncio.wait_for(reload, 5)
    new = entry.runtime_data.coordinator.room_tracking
    assert new is not old
    assert new.model["revision"] == 2
    assert new.model["rooms"][0]["label"] == "First editor"
    with pytest.raises(HomeAssistantError, match="Room model changed"):
        await save(hass, make_model(), revision=1)


async def test_deeply_nested_store_disables_only_room_tracking(hass, setup_bermuda_entry, hass_storage):
    entry = setup_bermuda_entry
    manager = entry.runtime_data.coordinator.room_tracking
    await save(hass, make_model())
    nested = "{}"
    for _ in range(550):
        nested = '{"nested":' + nested + "}"

    hass_storage[manager.store.key]["data"]["metadata"] = json.loads(nested)
    assert await hass.config_entries.async_reload(entry.entry_id)
    assert entry.state is ConfigEntryState.LOADED
    coordinator = entry.runtime_data.coordinator
    assert coordinator.last_update_success
    assert coordinator.room_tracking.model is None
    assert coordinator.room_tracking.result["kind"] == "disabled"
