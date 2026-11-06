"""Bounded server-side walk records, independent of calibration storage."""

from __future__ import annotations

import asyncio
import copy
import uuid
from datetime import UTC, datetime

import voluptuous as vol
from bluetooth_data_tools import monotonic_time_coarse
from homeassistant.core import SupportsResponse
from homeassistant.exceptions import HomeAssistantError, Unauthorized
from homeassistant.helpers.storage import Store

from .const import DOMAIN


def _fail(message):
    raise HomeAssistantError(message)


MAX_WALKS = 6
MAX_FRAMES = 1201


class WalkTestRecorder:
    """Collect normal coordinator observations without new radio subscriptions."""

    def __init__(self, hass, entry, manager) -> None:
        self.hass, self.entry, self.manager = hass, entry, manager
        self.store = Store(hass, 1, f"bermuda_walk_tests_{entry.entry_id}")
        self.walks = []
        self.active = None
        self.started = 0
        self.last_save = 0
        self.lock = asyncio.Lock()

    async def async_load(self):
        data = await self.store.async_load()
        if data is not None and (not isinstance(data, dict) or not isinstance(data.get("walks"), list)):
            _fail("Invalid walk-test history")
        self.walks = (data or {}).get("walks", [])[-MAX_WALKS:]
        for walk in self.walks:
            if (
                not isinstance(walk, dict)
                or walk.get("format") != "bermuda-walk"
                or not isinstance(walk.get("frames"), list)
            ):
                _fail("Invalid saved walk")
            if walk["status"] == "recording":
                walk["status"] = "interrupted"
                walk["durationMs"] = walk["frames"][-1]["t_ms"] if walk["frames"] else 0

    def _data(self):
        return {"walks": copy.deepcopy(self.walks)}

    @staticmethod
    def summary(walk):
        return {
            key: copy.deepcopy(walk[key])
            for key in ("id", "label", "startedAt", "status", "durationMs", "revision", "setupConfirmed")
        } | {"frames": len(walk["frames"])}

    def status(self):
        return {
            "active": self.summary(self.active) if self.active is not None else None,
            "walks": [self.summary(w) for w in reversed(self.walks)],
        }

    def interrupt(self):
        if self.active is not None:
            self.active["status"] = "configuration_changed"
            self.active = None
            self.store.async_delay_save(self._data, 0)

    def capture(self, now, vector, evidence, signature, unavailable, result):
        walk = self.active
        if walk is None:
            return
        elapsed = round((now - self.started) * 1000)
        if walk["frames"] and elapsed - walk["frames"][-1]["t_ms"] < 950:
            return
        changed = signature != walk["signature"] or self.manager.model["revision"] != walk["model"]["revision"]
        if changed or elapsed > walk["limitMs"] or len(walk["frames"]) >= MAX_FRAMES:
            walk["status"] = "configuration_changed" if changed else "complete"
            self.active = None
            self.store.async_delay_save(self._data, 0)
            return
        walk["frames"].append(
            {
                "t_ms": elapsed,
                "vector": dict(vector),
                "evidence": [list(item) for item in evidence],
                "signature": signature,
                "unavailableSources": sorted(unavailable),
                "recorded": copy.deepcopy(result),
            }
        )
        walk["durationMs"] = elapsed
        if now - self.last_save >= 15:
            self.last_save = now
            self.store.async_delay_save(self._data, 1)

    async def command(self, call):
        action = call.data.get("action", "status")
        if action in {"status", "get"}:
            if action == "status":
                return self.status()
            walk = next((w for w in self.walks if w["id"] == call.data.get("walk_id")), None)
            if walk is None:
                _fail("Walk not found")
            return {"walk": copy.deepcopy(walk)}
        if user_id := call.context.user_id:
            user = await self.hass.auth.async_get_user(user_id)
            if user is None or not user.is_admin:
                raise Unauthorized
        async with self.lock:
            if action == "start":
                if self.active is not None:
                    _fail("A walk is already recording")
                if len(self.walks) >= MAX_WALKS:
                    _fail("Download and remove an old walk before starting another (six-walk limit)")
                model = self.manager.model
                if not model or not model.get("enabled"):
                    _fail("Enable the learned-room model before recording a walk")
                if not self.manager.last_signature:
                    _fail("Wait for the first tracking update before starting a walk")
                self.started = monotonic_time_coarse()
                self.last_save = self.started
                walk = {
                    "format": "bermuda-walk",
                    "version": 1,
                    "id": uuid.uuid4().hex,
                    "label": call.data.get("label", "Walk test")[:80],
                    "startedAt": datetime.now(UTC).isoformat(),
                    "status": "recording",
                    "revision": 0,
                    "durationMs": 0,
                    "limitMs": 600000,
                    "model": copy.deepcopy(model),
                    "signature": self.manager.last_signature,
                    "setupConfirmed": call.data.get("setup_confirmed", False),
                    "frames": [],
                    "markers": [],
                    "annotations": [],
                    "engine": "fingerprint-v1",
                }
                tracker = self.manager.tracker
                walk["initialTracker"] = {
                    "confirmed": tracker.confirmed,
                    "confirmedAt": tracker.confirmed_at - self.started * 1000
                    if tracker.confirmed_at is not None
                    else None,
                    "pending": tracker.pending,
                    "pendingSince": tracker.pending_since - self.started * 1000,
                    "count": tracker.count,
                    "evidence": [list(item) for item in tracker.evidence] if tracker.evidence is not None else None,
                }
                self.walks.append(walk)
                self.active = walk
            else:
                walk = next((w for w in self.walks if w["id"] == call.data.get("walk_id")), None)
                if walk is None:
                    _fail("Walk not found")
                if action == "stop":
                    if self.active is not walk:
                        _fail("This walk is not recording")
                    walk["status"] = "complete"
                    self.active = None
                elif action == "delete":
                    if self.active is walk:
                        _fail("Stop the walk before removing it")
                    self.walks.remove(walk)
                elif action == "annotate":
                    if self.active is walk:
                        _fail("Stop the walk before adding interval labels")
                    if call.data.get("expected_revision") != walk["revision"]:
                        _fail("Walk labels changed; reopen the walk before editing")
                    start, end = call.data.get("from_ms", -1), call.data.get("to_ms", -1)
                    room = call.data.get("room_id")
                    if (
                        room not in {r["id"] for r in walk["model"]["rooms"]}
                        or not 0 <= start < end <= walk["durationMs"]
                        or len(walk["annotations"]) >= 100
                    ):
                        _fail("Choose a room and a valid interval within this walk")
                    if any(start < a["to_ms"] and end > a["from_ms"] for a in walk["annotations"]):
                        _fail("Room label intervals must not overlap")
                    walk["annotations"].append({"from_ms": start, "to_ms": end, "room_id": room})
                    walk["revision"] += 1
                else:
                    _fail("Unknown walk action")
            await self.store.async_save(self._data())
            return {**self.status(), "walk": copy.deepcopy(walk)}

    def async_register_services(self):
        schema = vol.Schema(
            {
                vol.Optional("action", default="status"): vol.In(
                    ["status", "start", "stop", "get", "annotate", "delete"]
                ),
                vol.Optional("walk_id"): str,
                vol.Optional("label"): str,
                vol.Optional("setup_confirmed", default=False): bool,
                vol.Optional("expected_revision"): vol.All(int, vol.Range(min=0)),
                vol.Optional("from_ms"): vol.All(int, vol.Range(min=0)),
                vol.Optional("to_ms"): vol.All(int, vol.Range(min=0)),
                vol.Optional("room_id"): str,
            }
        )
        self.hass.services.async_register(
            DOMAIN, "walk_test", self.command, schema, supports_response=SupportsResponse.ONLY
        )
        self.entry.async_on_unload(lambda: self.hass.services.async_remove(DOMAIN, "walk_test"))
