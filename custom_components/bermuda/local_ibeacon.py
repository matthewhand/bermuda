"""
Fresh packet timestamps for configured iBeacons received by local adapters.

The ordinary Bluetooth discovery callback is change-driven. The address-specific
advertisement callback receives unchanged packets too. Only a matching iBeacon
in the current raw packet may refresh a local receiver's reading.
"""

from __future__ import annotations

import math


class LocalIBeaconCapture:
    """Bounded, unloadable subscriptions; no scanner or radio configuration changes."""

    def __init__(self, register, configured, is_local, clock, max_addresses=32) -> None:
        self._register = register
        self._configured = configured
        self._is_local = is_local
        self._clock = clock
        self._max_addresses = max_addresses
        self._cancel = {}
        self._pending = {}
        self._stamps = {}

    def _identity(self, info):
        data = info.manufacturer_data.get(76, b"")
        if len(data) < 22 or data[:2] != b"\x02\x15":
            return None
        identity = f"{data[2:18].hex()}_{int.from_bytes(data[18:20], 'big')}_{int.from_bytes(data[20:22], 'big')}"
        return identity if identity.upper() in self._configured() else None

    def observe(self, info):
        """Enroll only configured iBeacon addresses as discovery sees them."""
        if self._register is None or self._identity(info) is None:
            return
        address = info.address.upper()
        if address not in self._cancel:
            if len(self._cancel) >= self._max_addresses:
                self._forget(next(iter(self._cancel)))
            self._cancel[address] = self._register(self.capture, address)
        self.capture(info)

    def capture(self, info):
        """Record actual, fresh packets, including equal-RSSI repeats."""
        if self._identity(info) is None or not self._is_local(info.source):
            return
        if not isinstance(info.rssi, int | float) or not -127 < info.rssi < 0:
            return
        address = info.address.upper()
        if address not in self._cancel:
            return
        raw = getattr(info, "raw", None)
        data = info.manufacturer_data.get(76, b"")
        # Manufacturer data is merged across packets by HA. Do not treat a
        # different packet from the same address as proof of this iBeacon.
        if not raw or b"\x4c\x00" + data[:22] not in raw:
            return
        stamp = info.time
        now = self._clock()
        if not isinstance(stamp, int | float) or not math.isfinite(stamp) or stamp > now + 5 or stamp < now - 30:
            return
        key = (info.source.lower(), address)
        if stamp <= self._stamps.get(key, -1):
            return
        self._stamps[key] = stamp
        self._pending[key] = info

    def get_stamp(self, source, address):
        """Return the last proven packet timestamp for this receiver/address."""
        # Retain the last proven stamp, even after expiry: returning None would
        # re-enable the RSSI-change fallback for an unproven scanner-cache entry.
        # Distance expiry runs independently during every calculation cycle.
        # These entries are bounded by subscribed addresses and removed on eviction.
        return self._stamps.get((source.lower(), address.upper()))

    def drain(self):
        """Coalesce high-rate adverts until Bermuda's next normal update."""
        pending = self._pending
        self._pending = {}
        now = self._clock()
        return [info for info in pending.values() if now - info.time <= 30]

    def _forget(self, address):
        cancel = self._cancel.pop(address)
        cancel()
        for key in list(self._stamps):
            if key[1] == address:
                self._stamps.pop(key, None)
                self._pending.pop(key, None)

    def close(self):
        """Release callbacks and observations when the integration unloads."""
        for address in list(self._cancel):
            self._forget(address)
