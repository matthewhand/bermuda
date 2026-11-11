"""Exercise HA's real Bluetooth callback registration with injected packets."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from bleak.backends.device import BLEDevice
from bluetooth_data_tools import monotonic_time_coarse, parse_advertisement_data_bytes
from habluetooth import get_manager
from home_assistant_bluetooth import BluetoothServiceInfoBleak
from homeassistant.components import bluetooth
from homeassistant.helpers import area_registry

from custom_components.bermuda.const import (
    AREA_MAX_AD_AGE_DEFAULT,
    CONF_DEVICES,
    CONF_DEVTRACK_TIMEOUT,
    DISTANCE_RETENTION_SECONDS,
    SERVICE_UUID_FMDN,
)
from custom_components.bermuda.device_tracker import BermudaDeviceTracker
from custom_components.bermuda.sensor import BermudaSensorRange, BermudaSensorScannerRange
from tests.test_local_ibeacon import ADDRESS, DATA, IDENTITY


@pytest.mark.skipif(
    not hasattr(bluetooth, "async_register_advertisement_callback"), reason="Requires Core 2026.10 or later"
)
async def test_real_callback_to_metadevice_and_reload(hass, setup_bermuda_entry, monkeypatch):
    monkeypatch.setattr(
        "homeassistant.components.bluetooth.manager.discovery_flow.async_create_flow", lambda *args, **kwargs: None
    )
    entry = setup_bermuda_entry
    coordinator = entry.runtime_data.coordinator
    manager = get_manager()
    scanner = next(scanner for scanner in manager.async_current_scanners() if scanner.connectable)
    source = scanner.source
    coordinator.options[CONF_DEVICES] = [IDENTITY]
    coordinator._waitingfor_load_manufacturer_ids = False  # The suite stubs the loader.
    coordinator._refresh_scanners(force=True)
    old_capture = coordinator.local_ibeacon
    start = monotonic_time_coarse()
    now = [start]
    old_capture._clock = lambda: now[0]
    device = BLEDevice(ADDRESS, "Synthetic beacon", {})
    raw = bytes.fromhex("1aff4c00") + DATA

    def send(time, payload=raw, address=ADDRESS):
        now[0] = time
        manager.scanner_adv_received(
            BluetoothServiceInfoBleak(
                "Synthetic beacon",
                address,
                -64,
                {76: DATA},
                {},
                [],
                source,
                device if address == ADDRESS else BLEDevice(address, "Synthetic beacon", {}),
                None,
                True,
                time,
                -59,
                payload,
            )
        )

    with (
        patch("custom_components.bermuda.coordinator.monotonic_time_coarse", lambda: now[0]),
        patch("custom_components.bermuda.bermuda_device.monotonic_time_coarse", lambda: now[0]),
        patch("custom_components.bermuda.bermuda_advert.monotonic_time_coarse", lambda: now[0]),
    ):
        send(start)
        coordinator._async_update_data_internal()
        assert old_capture.get_stamp(source, ADDRESS) == start
        meta = coordinator._get_device(IDENTITY)
        assert meta is not None and meta.adverts
        apple = bytes.fromhex("0f184c00") + DATA
        send(start + 0.5, payload=bytes((1 + len(apple), 0x16)) + apple)
        coordinator._async_update_data_internal()
        assert old_capture.get_stamp(source, ADDRESS) == start
        for seconds in range(1, 61):
            send(start + seconds)
            coordinator._async_update_data_internal()
        advert = next(iter(meta.adverts.values()))
        assert advert.stamp == start + 60 and advert.rssi_distance is not None
        now[0] = start + 60 + AREA_MAX_AD_AGE_DEFAULT + 1
        coordinator._async_update_data_internal()
        assert advert.rssi_distance is None

    await hass.async_block_till_done()
    assert await hass.config_entries.async_reload(entry.entry_id)
    new = entry.runtime_data.coordinator
    assert new is not coordinator
    new.options[CONF_DEVICES] = [IDENTITY]
    new._waitingfor_load_manufacturer_ids = False
    new._refresh_scanners(force=True)
    assert not old_capture._cancel and not old_capture._stamps
    rotated = "02:00:00:00:01:02"
    send(monotonic_time_coarse(), address=rotated)
    new._async_update_data_internal()
    assert new.local_ibeacon.get_stamp(source, rotated) is not None
    assert not old_capture._cancel and not old_capture._pending
    await hass.async_block_till_done()
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert not new.local_ibeacon._cancel and not new.local_ibeacon._stamps


@pytest.mark.skipif(
    not hasattr(bluetooth, "async_register_advertisement_callback"), reason="Requires Core 2026.10 or later"
)
@pytest.mark.parametrize("supply_raw", [True, False])
async def test_shared_address_keeps_find_my_native_without_renewing_ibeacon(
    hass, setup_bermuda_entry, monkeypatch, supply_raw
):
    """The native EID resolver and physical reading survive while the logical beacon expires."""
    monkeypatch.setattr(
        "homeassistant.components.bluetooth.manager.discovery_flow.async_create_flow", lambda *args, **kwargs: None
    )
    entry = setup_bermuda_entry
    coordinator = entry.runtime_data.coordinator
    manager = get_manager()
    scanner = next(scanner for scanner in manager.async_current_scanners() if scanner.connectable)
    coordinator.options[CONF_DEVICES] = [IDENTITY]
    coordinator.options[CONF_DEVTRACK_TIMEOUT] = 30
    coordinator._waitingfor_load_manufacturer_ids = False
    coordinator._refresh_scanners(force=True)
    scanner_device = coordinator._get_device(scanner.source)
    assert scanner_device is not None and not scanner_device.is_remote_scanner
    area = area_registry.async_get(hass).async_get_or_create("Office")
    scanner_device.area_id = area.id
    scanner_device.area_name = area.name
    cache = {}
    backend = SimpleNamespace(
        source=scanner.source,
        time_since_last_detection=lambda: 0,
        discovered_devices_and_advertisement_data=cache,
    )
    monkeypatch.setattr(coordinator, "_scanner_init_pending", False)
    monkeypatch.setattr(coordinator, "_hascanners", [backend])
    monkeypatch.setattr(scanner_device, "_hascanner", backend)
    resolve = Mock(return_value=([], None))
    monkeypatch.setattr(coordinator.fmdn, "process_resolution_all_with_status", resolve)
    start = monotonic_time_coarse()
    now = [start]
    monkeypatch.setattr(coordinator.local_ibeacon, "_clock", lambda: now[0])
    for module in ("coordinator", "bermuda_device", "bermuda_advert", "metadevice_manager", "area_selection"):
        monkeypatch.setattr(f"custom_components.bermuda.{module}.monotonic_time_coarse", lambda: now[0])

    device = BLEDevice(ADDRESS, "Shared address", {})
    raw_ibeacon = bytes.fromhex("1aff4c00") + DATA
    eid = b"\x40" + bytes(range(20))
    raw_find_my = bytes((len(eid) + 3, 0x16, 0xAA, 0xFE)) + eid
    service_data = next(
        item
        for item in parse_advertisement_data_bytes(raw_find_my)
        if isinstance(item, dict) and SERVICE_UUID_FMDN in item
    )

    def send(elapsed, rssi, payload, services):
        now[0] = start + elapsed
        # HA may preserve the old Apple field in the merged advertisement.
        info = BluetoothServiceInfoBleak(
            "Shared address",
            ADDRESS,
            rssi,
            {76: DATA},
            services,
            [],
            scanner.source,
            device,
            None,
            True,
            now[0],
            -59,
            payload,
        )
        cache[ADDRESS] = (device, info.advertisement)
        manager.scanner_adv_received(info)
        coordinator._async_update_data_internal()

    for elapsed in range(7):
        send(elapsed, -64, raw_ibeacon, {})
    last_beacon = start + 6
    meta = coordinator._get_device(IDENTITY)
    physical = coordinator._get_device(ADDRESS)
    assert meta is not None and physical is not None
    assert physical.last_seen == last_beacon
    tracked = BermudaDeviceTracker(coordinator, entry, IDENTITY)
    distance = BermudaSensorRange(coordinator, entry, IDENTITY)
    scanner_distance = BermudaSensorScannerRange(coordinator, entry, IDENTITY, scanner_device.address)
    assert tracked.is_connected
    assert distance.native_value is not None, (
        meta.diag_area_switch,
        meta.area_id,
        meta.area_distance,
        meta.get_scanner(scanner_device.address).rssi_distance,
    )
    assert scanner_distance.native_value is not None

    native_packet = raw_find_my if supply_raw else None
    send(10, -72, native_packet, service_data)
    assert resolve.call_args_list[0].args == (eid, physical.address)
    assert physical.get_scanner(scanner_device.address).rssi == -72
    assert physical.last_seen > meta.last_seen == last_beacon
    assert meta.get_scanner(scanner_device.address).stamp == last_beacon

    # The scanner is online and the same MAC continues sending, but iBeacon packets stop.
    send(6 + AREA_MAX_AD_AGE_DEFAULT + 5, -74, native_packet, service_data)
    assert physical.last_seen > meta.last_seen == last_beacon
    assert physical.get_scanner(scanner_device.address).rssi == -74
    assert meta.get_scanner(scanner_device.address).stamp == last_beacon
    assert not tracked.is_connected
    assert scanner_distance.native_value is None

    send(6 + DISTANCE_RETENTION_SECONDS + 5, -76, native_packet, service_data)
    assert physical.get_scanner(scanner_device.address).rssi == -76
    assert not tracked.is_connected
    assert distance.native_value is None
