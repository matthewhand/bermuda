"""
Opt-in persistent room models and a separate learned-room estimate.

The existing nearest-scanner Area sensor, distance calibration and radio settings
are deliberately left independent of this experimental estimator.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import re

import voluptuous as vol
from bluetooth_data_tools import monotonic_time_coarse
from homeassistant.core import SupportsResponse
from homeassistant.exceptions import HomeAssistantError, Unauthorized
from homeassistant.helpers.storage import Store

from .const import DOMAIN
from .room_tracking import ESTIMATORS, TemporalTracker
from .walk_test import WalkTestRecorder

_LOGGER = logging.getLogger(__name__)


def _reject(message):
    raise ValueError(message)


def _validate_recordings(samples, ids):
    for sample in samples:
        if (
            not isinstance(sample, dict)
            or sample.get("roomId") not in ids
            or not isinstance(sample.get("signature"), str)
            or len(sample["signature"]) > 1000
        ):
            _reject("Invalid labelled recording")
        signals = sample.get("signals")
        if not isinstance(signals, dict) or len(signals) > 16:
            _reject("Invalid receiver measurements")
        for signal in signals.values():
            if signal is None:
                continue
            if not isinstance(signal, dict):
                _reject("Invalid signal summary")
            for key in ("median", "mad", "count"):
                if not isinstance(signal.get(key), int | float) or not math.isfinite(signal[key]):
                    _reject("Signal summaries must be finite numbers")
            if (
                not -127 < signal["median"] < 0
                or not 0 <= signal["mad"] <= 127
                or not isinstance(signal["count"], int)
                or not 1 <= signal["count"] <= 10000
            ):
                _reject("Invalid RSSI, dispersion or observation count")


def validate_room_model(value):
    """Validate a bounded, portable model before changing persistent state."""
    if not isinstance(value, dict) or len(json.dumps(value, allow_nan=False)) > 2_000_000:
        _reject("Room model must be a JSON object smaller than 2 MB")
    model = copy.deepcopy(value)
    if model.get("version") != 1 or not isinstance(model.get("revision"), int):
        _reject("Unsupported room model version or revision")
    rooms = model.get("rooms")
    samples = model.get("samples")
    if not isinstance(rooms, list) or not 1 <= len(rooms) <= 50 or not isinstance(samples, list) or len(samples) > 120:
        _reject("Expected 1-50 rooms and at most 120 active recordings")
    ids = set()
    for room in rooms:
        if not isinstance(room, dict) or not re.fullmatch(r"[a-z0-9_-]{1,70}", room.get("id", "")) or room["id"] in ids:
            _reject("Room identifiers must be unique")
        if not isinstance(room.get("label"), str) or not 1 <= len(room["label"].strip()) <= 50:
            _reject("Each room needs a short label")
        ids.add(room["id"])
    config = model.get("config", {})
    if not isinstance(config, dict) or not isinstance(config.get("device"), dict):
        _reject("Invalid device configuration")
    device = config.get("device", {})
    receivers = config.get("receivers", [])
    if not isinstance(device.get("id"), str) or not re.fullmatch(r"[a-zA-Z0-9:_-]{1,160}", device["id"]):
        _reject("Choose a valid tracked device identifier")
    if not isinstance(receivers, list) or not 1 <= len(receivers) <= 16:
        _reject("Configure 1-16 receivers")
    sources = set()
    for receiver in receivers:
        if not isinstance(receiver, dict):
            _reject("Receiver configuration must be an object")
        source = receiver.get("key", "")
        if (
            not isinstance(source, str)
            or not re.fullmatch(r"[a-zA-Z0-9:_.-]{1,160}", source)
            or source.lower() in sources
        ):
            _reject("Receiver sources must be unique")
        if receiver.get("roomId") and receiver["roomId"] not in ids:
            _reject("Receiver room is missing from the model")
        weight = receiver.get("weight", 1)
        if not isinstance(weight, int | float) or not math.isfinite(weight) or not 0.1 <= weight <= 5:
            _reject("Receiver weights must be between 0.1 and 5")
        sources.add(source.lower())
    for field in ("areaEntity", "transmitterEntity"):
        if device.get(field) and (
            not isinstance(device[field], str) or not re.fullmatch(r"[a-z_]+\.[a-z0-9_]+", device[field])
        ):
            _reject("Invalid Home Assistant entity identifier")
    edges = model.get("connections", [])
    if (
        not isinstance(edges, list)
        or len(edges) > 200
        or any(
            not isinstance(edge, list) or len(edge) != 2 or edge[0] == edge[1] or any(room not in ids for room in edge)
            for edge in edges
        )
    ):
        _reject("Invalid room connection")
    groups = model.get("roomGroups", [])
    if not isinstance(groups, list) or len(groups) > 20:
        _reject("Too many room groups")
    for group in groups:
        if (
            not isinstance(group, dict)
            or not re.fullmatch(r"[a-z0-9_-]{1,70}", group.get("id", ""))
            or not isinstance(group.get("rooms"), list)
            or not 1 <= len(group["rooms"]) <= 50
            or any(room not in ids for room in group["rooms"])
        ):
            _reject("Invalid room group")
    _validate_recordings(samples, ids)
    if model.get("estimator", "fingerprint") not in ESTIMATORS:
        _reject("Unsupported room estimator")
    if not isinstance(model.get("enabled", False), bool):
        _reject("Enabled must be a boolean")
    return model


class RoomTrackingManager:
    """Keep learning optional, persistent and independent of existing room selection."""

    def __init__(self, hass, entry, coordinator) -> None:
        self.hass = hass
        self.entry = entry
        self.coordinator = coordinator
        self.store = Store(hass, 1, f"bermuda_room_model_{entry.entry_id}")
        self.model = None
        self.result = {"kind": "disabled", "location": None}
        self.tracker = TemporalTracker()
        self.estimator = ESTIMATORS["fingerprint"]()
        self.last_signature = None
        self.unavailable_sources = []
        self._save_lock = asyncio.Lock()
        self.walks = WalkTestRecorder(hass, entry, self)

    async def async_load(self):
        """Load the model without making old or corrupt calibration mandatory."""
        try:
            await self.walks.async_load()
        except (ValueError, TypeError, KeyError, OSError, HomeAssistantError):
            _LOGGER.warning("Walk-test history could not load; calibration storage is independent")
        try:
            stored = await self.store.async_load()
            if stored is not None:
                self.model = validate_room_model(stored)
                self.estimator = ESTIMATORS[self.model.get("estimator", "fingerprint")]()
        except (ValueError, TypeError, KeyError, OSError, HomeAssistantError):
            _LOGGER.warning("Experimental room model could not load; existing Bermuda tracking is unchanged")

    def async_register_services(self):
        """Register model read/save services; no radio changes are made."""
        self.walks.async_register_services()

        async def get_model(_call):
            return {
                "model": copy.deepcopy(self.model),
                "result": self.result,
                "unavailableSources": self.unavailable_sources,
            }

        async def save_model(call):
            if user_id := call.context.user_id:
                user = await self.hass.auth.async_get_user(user_id)
                if user is None or not user.is_admin:
                    raise Unauthorized
            try:
                model = validate_room_model(call.data["model"])
            except (ValueError, TypeError, KeyError) as err:
                message = str(err)
                raise HomeAssistantError(message) from err
            async with self._save_lock:
                expected = call.data["expected_revision"]
                current = self.model.get("revision", 0) if self.model else 0
                if expected != current:
                    message = "Room model changed; reload before saving"
                    raise HomeAssistantError(message)
                model["revision"] = current + 1
                await self.store.async_save(model)
                self.model = model
                self.estimator = ESTIMATORS[model.get("estimator", "fingerprint")]()
                self.update()
                self.coordinator.async_update_listeners()
                return {"model": copy.deepcopy(model), "result": self.result}

        self.hass.services.async_register(DOMAIN, "get_room_model", get_model, supports_response=SupportsResponse.ONLY)
        self.hass.services.async_register(
            DOMAIN,
            "save_room_model",
            save_model,
            vol.Schema({vol.Required("model"): dict, vol.Required("expected_revision"): int}),
            supports_response=SupportsResponse.ONLY,
        )
        for service in ("get_room_model", "save_room_model"):
            self.entry.async_on_unload(lambda service=service: self.hass.services.async_remove(DOMAIN, service))

    def update(self):
        """Run from normal coordinator updates; never modify the legacy Area sensor."""
        if not self.model or not self.model.get("enabled", False):
            self.walks.interrupt()
            self.result = {"kind": "disabled", "location": None}
            return
        try:
            self._update()
        except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
            # An experimental estimator must not fail the existing coordinator.
            self.result = {"kind": "unknown", "location": None, "reason": "Room estimator has no usable observations"}

    def _update(self):
        model = self.model
        config = model["config"]
        device_config = config["device"]
        device = self.coordinator._get_device(device_config["id"])  # noqa: SLF001
        now = monotonic_time_coarse()
        latest = {}
        for advert in device.adverts.values() if device is not None else ():
            key = advert.scanner_address.lower()
            if key not in latest or advert.stamp > latest[key].stamp:
                latest[key] = advert
        vector = {}
        evidence = []
        offline = set()
        for receiver in config["receivers"]:
            key = receiver["key"].lower()
            advert = latest.get(key)
            scanner = self.coordinator._get_device(key)  # noqa: SLF001
            backend = getattr(scanner, "_hascanner", None)
            if scanner is None or (backend is not None and not backend.scanning) or scanner.last_seen < now - 60:
                offline.add(key)
            fresh = advert is not None and advert.rssi_distance is not None and now - advert.stamp <= 30
            vector[key] = advert.rssi if fresh else None
            evidence.append((key, advert.stamp if fresh else None))
        self.unavailable_sources = sorted(offline)
        transmitter = self.hass.states.get(device_config.get("transmitterEntity", ""))
        power = transmitter.attributes.get("Transmitting power", "unknown") if transmitter else "unknown"
        signature = (
            device_config["id"]
            + "|"
            + power
            + "|"
            + ",".join(sorted(vector))
            + "|layout:"
            + str(model.get("receiverLayoutVersion", 0))
        )
        if signature != self.last_signature:
            self.tracker = TemporalTracker()
            self.last_signature = signature
        estimate = self.estimator.estimate(vector, model, signature, offline)
        self.result = {
            **self.tracker.update(estimate, now * 1000, tuple(sorted(evidence)), model),
            "estimator": model.get("estimator", "fingerprint"),
            "weights": estimate.get("weights", {}),
            "observedReceivers": sum(value is not None for value in vector.values()),
            "unavailableSources": self.unavailable_sources,
        }
        self.walks.capture(now, vector, evidence, signature, offline, self.result)
