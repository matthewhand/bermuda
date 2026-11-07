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
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.service import async_register_admin_service
from homeassistant.helpers.storage import Store

from .const import DOMAIN
from .room_tracking import ESTIMATORS, TemporalTracker

_LOGGER = logging.getLogger(__name__)


def _reject(message):
    raise ValueError(message)


def _receiver_signature(sources, layout):
    return f"|{','.join(sorted(sources))}|layout:{layout}"


def _validate_recordings(samples, ids, device_id, sources, layout):
    signature_end = _receiver_signature(sources, layout)
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
        if sample["signature"].startswith(f"{device_id}|") and sample["signature"].endswith(signature_end):
            normalized = {}
            for key, signal in signals.items():
                if not isinstance(key, str) or key.lower() not in sources or key.lower() in normalized:
                    _reject("Recording receiver does not match its configured layout")
                normalized[key.lower()] = signal
            sample["signals"] = normalized
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


def _validate_model_size(value):
    """Bound JSON depth before serializing or copying the model."""
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, dict):
            children = item.values()
        elif isinstance(item, list | tuple):
            children = item
        else:
            continue
        if depth >= 32:
            _reject("Room model is nested too deeply")
        pending.extend((child, depth + 1) for child in children)
    if len(json.dumps(value, allow_nan=False)) > 2_000_000:
        _reject("Room model must be smaller than 2 MB")


def validate_room_model(value):
    """Validate a bounded, portable model before changing persistent state."""
    if not isinstance(value, dict):
        _reject("Room model must be a JSON object")
    _validate_model_size(value)
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
    layout = model.get("receiverLayoutVersion", 0)
    if type(layout) is not int or layout < 0:
        _reject("Receiver layout version must be a nonnegative integer")
    _validate_recordings(samples, ids, device["id"], sources, layout)
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
        self._closed = False

    async def async_close(self):
        """Finish any active write before the next manager can load the store."""
        self._closed = True
        async with self._save_lock:
            pass

    async def async_load(self):
        """Load the model without making old or corrupt calibration mandatory."""
        try:
            stored = await self.store.async_load()
            if stored is not None:
                self.model = validate_room_model(stored)
                self.estimator = ESTIMATORS[self.model.get("estimator", "fingerprint")]()
        except (ValueError, TypeError, KeyError, OSError, HomeAssistantError, RecursionError):
            self.model = None
            self.result = {"kind": "disabled", "location": None}
            _LOGGER.warning("Experimental room model could not load; existing Bermuda tracking is unchanged")

    def async_register_services(self):
        """Register model read/save services; no radio changes are made."""

        async def get_model(_call):
            return {
                "model": copy.deepcopy(self.model),
                "result": self.result,
                "unavailableSources": self.unavailable_sources,
            }

        async def save_model(call):
            try:
                model = validate_room_model(call.data["model"])
            except (ValueError, TypeError, KeyError) as err:
                message = str(err)
                raise HomeAssistantError(message) from err
            write = self.hass.async_create_task(
                self._async_save_model(model, call.data["expected_revision"]), "Bermuda room model save"
            )
            return await asyncio.shield(write)

        async_register_admin_service(
            self.hass, DOMAIN, "get_room_model", get_model, supports_response=SupportsResponse.ONLY
        )
        async_register_admin_service(
            self.hass,
            DOMAIN,
            "save_room_model",
            save_model,
            vol.Schema({vol.Required("model"): dict, vol.Required("expected_revision"): int}),
            supports_response=SupportsResponse.ONLY,
        )
        for service in ("get_room_model", "save_room_model"):
            self.entry.async_on_unload(lambda service=service: self.hass.services.async_remove(DOMAIN, service))

    async def _async_save_model(self, model, expected):
        """Serialize a validated save with unload and other editors."""
        async with self._save_lock:
            if self._closed:
                message = "Room model is reloading; retry after setup"
                raise HomeAssistantError(message)
            current = self.model.get("revision", 0) if self.model else 0
            if expected != current:
                message = "Room model changed; reload before saving"
                raise HomeAssistantError(message)
            if self.model:
                old_layout = self.model.get("receiverLayoutVersion", 0)
                layout = model.get("receiverLayoutVersion", 0)
                old_rooms = {
                    receiver["key"].lower(): receiver.get("roomId") or None
                    for receiver in self.model["config"]["receivers"]
                }
                moved = any(
                    receiver["key"].lower() in old_rooms
                    and old_rooms[receiver["key"].lower()] != (receiver.get("roomId") or None)
                    for receiver in model["config"]["receivers"]
                )
                if layout < old_layout or (moved and layout == old_layout):
                    message = "Increase the receiver layout version after moving a receiver"
                    raise HomeAssistantError(message)
            model["revision"] = current + 1
            await self.store.async_save(model)
            self.model = model
            self.tracker = TemporalTracker()
            self.last_signature = None
            self.estimator = ESTIMATORS[model.get("estimator", "fingerprint")]()
            if not self._closed:
                self.update()
                self.coordinator.async_update_listeners()
            return {"model": copy.deepcopy(model), "result": self.result}

    def update(self):
        """Run from normal coordinator updates; never modify the legacy Area sensor."""
        if not self.model or not self.model.get("enabled", False):
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
        evidence = {}
        eligible = set()
        offline = set()
        for receiver in config["receivers"]:
            key = receiver["key"].lower()
            advert = latest.get(key)
            scanner = self.coordinator._get_device(key)  # noqa: SLF001
            backend = getattr(scanner, "_hascanner", None)
            if scanner is None or (backend is not None and not backend.scanning) or scanner.last_seen < now - 60:
                offline.add(key)
            usable = (
                advert is not None
                and isinstance(advert.rssi, int | float)
                and -127 < advert.rssi < 0
                and math.isfinite(advert.stamp)
                and advert.stamp <= now + 5
            )
            fresh = key not in offline and usable and advert.rssi_distance is not None and now - advert.stamp <= 30
            vector[key] = advert.rssi if fresh else None
            if usable:
                evidence[key] = advert.stamp * 1000
            if fresh:
                eligible.add(key)
        self.unavailable_sources = sorted(offline)
        transmitter = self.hass.states.get(device_config.get("transmitterEntity", ""))
        power = transmitter.attributes.get("Transmitting power", "unknown") if transmitter else "unknown"
        signature = f'{device_config["id"]}|{power}{_receiver_signature(vector, model.get("receiverLayoutVersion", 0))}'
        if signature != self.last_signature:
            self.tracker = TemporalTracker()
            self.last_signature = signature
        estimate = self.estimator.estimate(vector, model, signature, offline)
        self.result = {
            **self.tracker.update(estimate, now * 1000, evidence, model, eligible),
            "estimator": model.get("estimator", "fingerprint"),
            "weights": estimate.get("weights", {}),
            "observedReceivers": sum(value is not None for value in vector.values()),
            "unavailableSources": self.unavailable_sources,
        }
