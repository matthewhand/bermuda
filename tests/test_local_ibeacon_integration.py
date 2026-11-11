"""Exercise HA's real Bluetooth callback registration with injected packets."""

from unittest.mock import patch

import pytest

from bleak.backends.device import BLEDevice
from bluetooth_data_tools import monotonic_time_coarse
from habluetooth import get_manager
from home_assistant_bluetooth import BluetoothServiceInfoBleak
from homeassistant.components import bluetooth

from custom_components.bermuda.const import AREA_MAX_AD_AGE_DEFAULT, CONF_DEVICES
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
