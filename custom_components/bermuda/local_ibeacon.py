"""
Fresh packet timestamps for configured iBeacons received by local adapters.

The ordinary Bluetooth discovery callback is change-driven. The address-specific
advertisement callback receives unchanged packets too. Only a matching iBeacon
in the current raw packet may refresh a local receiver's reading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from home_assistant_bluetooth import BluetoothServiceInfoBleak


@dataclass(frozen=True, slots=True)
class LocalIBeaconPacket:
    """A received packet with an iBeacon payload proven by its raw bytes."""

    info: BluetoothServiceInfoBleak
    identity: str
    payload: bytes


class LocalIBeaconCapture:
    """Bounded, unloadable subscriptions; no scanner or radio configuration changes."""

    def __init__(
        self,
        register: Callable[[Callable[[BluetoothServiceInfoBleak], None], str], Callable[[], None]] | None,
        configured: Callable[[], Collection[object]],
        is_local: Callable[[str], bool],
        clock: Callable[[], float],
        max_addresses: int = 32,
    ) -> None:
        self._register = register
        self._configured = configured
        self._is_local = is_local
        self._clock = clock
        self._max_addresses = max_addresses
        self._cancel: dict[str, Callable[[], None]] = {}
        self._pending: dict[tuple[str, str], LocalIBeaconPacket] = {}
        self._stamps: dict[str, dict[str, float]] = {}

    def _configured_identity(self, data: bytes) -> str | None:
        if len(data) < 23 or data[:2] != b"\x02\x15":
            return None
        identity = f"{data[2:18].hex()}_{int.from_bytes(data[18:20], 'big')}_{int.from_bytes(data[20:22], 'big')}"
        configured = self._configured()
        # This fork stores canonical lowercase identifiers. Older entries may
        # contain uppercase or dashed UUIDs.
        if identity in configured or any(
            isinstance(address, str) and address.strip().replace("-", "").lower() == identity for address in configured
        ):
            return identity
        return None

    def _configured_packet(self, raw: bytes) -> tuple[str, bytes] | None:
        """Read BLE advertising structures, which include a length and type."""
        offset = 0
        while offset < len(raw):
            length = raw[offset]
            end = offset + 1 + length
            if not length or end > len(raw):
                return None
            if length >= 26 and raw[offset + 1 : offset + 4] == b"\xff\x4c\x00":
                payload = raw[offset + 4 : end]
                if (identity := self._configured_identity(payload)) is not None:
                    return identity, payload
            offset = end
        return None

    def _local_packet(self, info: BluetoothServiceInfoBleak) -> tuple[str, bytes] | None:
        """Return the matching payload only for a usable local packet."""
        if not self._is_local(info.source) or not isinstance(info.rssi, int | float) or not -127 < info.rssi < 0:
            return None
        stamp = info.time
        now = self._clock()
        if not isinstance(stamp, int | float) or not math.isfinite(stamp) or stamp > now + 5 or stamp < now - 30:
            return None
        # HA can retain Apple manufacturer data from an earlier packet.
        return self._configured_packet(getattr(info, "raw", None) or b"")

    def observe(self, info: BluetoothServiceInfoBleak) -> None:
        """Discover configured addresses from HA's merged metadata."""
        if self._register is None or not self._configured_identity(info.manufacturer_data.get(76, b"")):
            return
        address = info.address.upper()
        if address not in self._cancel:
            if len(self._cancel) >= self._max_addresses:
                # Keep speculative addresses for mixed proxy/local discovery, but
                # only genuine local packets can displace a proven subscription.
                candidate = next((key for key in self._cancel if key not in self._stamps), None)
                if candidate is None:
                    if self._local_packet(info) is None:
                        return
                    candidate = min(self._cancel, key=lambda key: max(self._stamps[key].values()))
                self._cancel.pop(candidate)()
            self._cancel[address] = self._register(self.capture, address)
        self.capture(info)

    def capture(self, info: BluetoothServiceInfoBleak) -> None:
        """Record actual, fresh packets, including equal-RSSI repeats."""
        address = info.address.upper()
        if address not in self._cancel:
            return
        if (packet := self._local_packet(info)) is None:
            return
        source = info.source.lower()
        if (previous := self.get_stamp(source, address)) is not None and info.time <= previous:
            return
        self._stamps.setdefault(address, {})[source] = info.time
        self._pending[source, address] = LocalIBeaconPacket(info, *packet)

    def get_stamp(self, source: str, address: str) -> float | None:
        """Return the last proven receipt time, even after subscription eviction."""
        stamps = self._stamps.get(address.upper())
        return stamps.get(source.lower()) if stamps else None

    def drain(self) -> list[LocalIBeaconPacket]:
        """Coalesce high-rate adverts until Bermuda's next normal update."""
        pending = self._pending
        self._pending = {}
        now = self._clock()
        return [packet for packet in pending.values() if now - packet.info.time <= 30]

    def prune(self, devices: Collection[str]) -> None:
        """Forget packet history once Bermuda no longer tracks the address."""
        if not self._stamps:
            return
        for address in self._stamps.keys() - {address.upper() for address in devices}:
            cancel = self._cancel.pop(address, None)
            if cancel is not None:
                cancel()
            for source in self._stamps.pop(address):
                self._pending.pop((source, address), None)

    def close(self) -> None:
        """Release callbacks and observations when the integration unloads."""
        for cancel in self._cancel.values():
            cancel()
        self._cancel.clear()
        self._stamps.clear()
        self._pending.clear()
