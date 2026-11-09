"""Regression tests for genuine local-adapter packet freshness."""

from types import SimpleNamespace

import pytest

from custom_components.bermuda.bermuda_advert import BermudaAdvert
from custom_components.bermuda.const import AREA_MAX_AD_AGE_DEFAULT
from custom_components.bermuda.local_ibeacon import LocalIBeaconCapture

IDENTITY = "00112233445566778899aabbccddeeff_100_40004"
SOURCE = "11:22:33:44:55:66"
ADDRESS = "AA:BB:CC:DD:EE:01"
DATA = bytes.fromhex("021500112233445566778899aabbccddeeff00649c44c5")


@pytest.fixture
def capture_fixture(monkeypatch):
    clock = [100.0]
    callbacks = {}

    def register(callback, address):
        callbacks[address] = callback
        return lambda: callbacks.pop(address, None)

    capture = LocalIBeaconCapture(
        register, lambda: [IDENTITY.upper()], lambda source: source == SOURCE, lambda: clock[0], max_addresses=2
    )
    monkeypatch.setattr("custom_components.bermuda.bermuda_advert.monotonic_time_coarse", lambda: clock[0])
    return capture, clock, callbacks


def packet(clock, **changes):
    advert = SimpleNamespace(
        rssi=-64, tx_power=-59, local_name=None, manufacturer_data={76: DATA}, service_data={}, service_uuids=[]
    )
    fields = dict(
        address=ADDRESS,
        source=SOURCE,
        time=clock[0],
        rssi=-64,
        manufacturer_data={76: DATA},
        raw=bytes.fromhex("1aff4c00") + DATA,
        advertisement=advert,
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_equal_rssi_actual_packets_stay_current_but_cached_packets_expire(capture_fixture):
    capture, clock, _ = capture_fixture
    first = packet(clock)
    capture.observe(first)
    scanner = SimpleNamespace(
        address=SOURCE, name="Local", area_id="office", area_name="Office", is_remote_scanner=False, last_seen=100.0
    )
    scanner.async_as_scanner_get_stamp = lambda address: capture.get_stamp(SOURCE, address)
    device = SimpleNamespace(
        address=ADDRESS.lower(),
        ref_power=-59,
        name="Phone",
        name_bt_local_name=None,
        make_name=lambda: None,
        process_manufacturer_data=lambda _: None,
    )
    options = dict(rssi_offsets={}, ref_power=-59, attenuation=3, max_velocity=3, smoothing_samples=20)
    link = BermudaAdvert(device, first.advertisement, options, scanner)
    link.calculate_data()
    capture.drain()
    for tick in range(1, 61):
        clock[0] = 100 + tick
        capture.capture(packet(clock))
        newest = capture.drain()[0]
        link.update_advertisement(newest.advertisement, scanner)
        link.calculate_data()
        assert link.rssi_distance is not None
    clock[0] = 160 + AREA_MAX_AD_AGE_DEFAULT + 1
    link.update_advertisement(first.advertisement, scanner)
    link.calculate_data()
    assert link.rssi_distance is None


def test_merged_manufacturer_data_cannot_refresh_an_unrelated_raw_packet(capture_fixture):
    capture, clock, _ = capture_fixture
    capture.observe(packet(clock))
    capture.drain()
    clock[0] = 105
    service_data = b"\x0f\x18\x4c\x00" + DATA
    wrong_type = bytes((1 + len(service_data), 0x16)) + service_data
    for raw in (wrong_type, bytes.fromhex("1bff4c00") + DATA, None):
        capture.capture(packet(clock, raw=raw, rssi=-71))
    assert capture.drain() == []
    assert capture.get_stamp(SOURCE, ADDRESS) == 100


def test_remote_observations_do_not_enter_the_local_path(capture_fixture):
    capture, clock, _ = capture_fixture
    capture.observe(packet(clock, source="remote"))
    assert capture.drain() == []
    assert capture.get_stamp("remote", ADDRESS) is None


def test_proxy_discovery_enrolls_local_packets_without_evicting_proven_links(capture_fixture):
    capture, clock, callbacks = capture_fixture
    capture.observe(packet(clock, source="remote"))
    # The local packet may reach only the per-address callback if the proxy wins discovery.
    callbacks[ADDRESS](packet(clock, raw=bytes.fromhex("0201061aff4c00") + DATA))
    for index in range(2, 34):
        capture.observe(packet(clock, address=f"AA:BB:CC:DD:EE:{index:02X}", source="remote"))
    assert len(callbacks) == 2
    assert ADDRESS in callbacks
    assert capture.get_stamp(SOURCE, ADDRESS) == 100


def test_rotation_is_bounded_and_callbacks_are_unloaded(capture_fixture):
    capture, clock, callbacks = capture_fixture
    for suffix in ("01", "02", "03"):
        capture.observe(packet(clock, address="AA:BB:CC:DD:EE:" + suffix))
    assert len(callbacks) == 2
    assert ADDRESS not in callbacks
    assert capture.get_stamp(SOURCE, ADDRESS) == 100
    capture.prune(["aa:bb:cc:dd:ee:02", "aa:bb:cc:dd:ee:03"])
    assert capture.get_stamp(SOURCE, ADDRESS) is None
    capture.close()
    assert callbacks == {}
    assert capture.drain() == []


def test_old_future_and_invalid_rssi_packets_are_not_fresh(capture_fixture):
    capture, clock, _ = capture_fixture
    capture.observe(packet(clock))
    capture.drain()
    for stamp in (100, 99, 50, 200, float("nan"), float("inf")):
        capture.capture(packet(clock, time=stamp))
    clock[0] = 105
    capture.capture(packet(clock, rssi=-127))
    assert capture.drain() == []
    assert capture.get_stamp(SOURCE, ADDRESS) == 100


def test_original_local_rssi_change_fallback_expires_unchanged_readings(capture_fixture):
    """Control case: the original no-timestamp path cannot refresh equal RSSI."""
    _, clock, _ = capture_fixture
    scanner = SimpleNamespace(
        address=SOURCE,
        name="Local",
        area_id="office",
        area_name="Office",
        is_remote_scanner=False,
        last_seen=100.0,
        async_as_scanner_get_stamp=lambda address: None,
    )
    device = SimpleNamespace(
        address=ADDRESS.lower(),
        ref_power=-59,
        name="Phone",
        name_bt_local_name=None,
        make_name=lambda: None,
        process_manufacturer_data=lambda _: None,
    )
    options = dict(rssi_offsets={}, ref_power=-59, attenuation=3, max_velocity=3, smoothing_samples=20)
    advert = packet(clock).advertisement
    link = BermudaAdvert(device, advert, options, scanner)
    link.calculate_data()
    assert link.rssi_distance is not None
    for tick in range(1, 61):
        clock[0] = 100 + tick
        link.update_advertisement(advert, scanner)
        link.calculate_data()
    assert link.rssi_distance is None


def test_missing_api_and_unconfigured_beacons_do_not_subscribe():
    """Older HA and unrelated beacon identities retain their existing path."""
    calls = []
    register = lambda callback, address: calls.append(address)
    unsupported = LocalIBeaconCapture(None, lambda: [IDENTITY.upper()], lambda _: True, lambda: 100)
    unsupported.observe(packet([100]))
    unconfigured = LocalIBeaconCapture(register, lambda: [], lambda _: True, lambda: 100)
    unconfigured.observe(packet([100]))
    assert calls == []
    assert unsupported.drain() == []
    assert unconfigured.drain() == []


def test_repeated_observe_at_capacity_never_replaces_a_live_cancel_handle():
    """Existing addresses cannot be chosen for eviction while observing themselves."""
    registered = []
    cancelled = []

    def register(callback, address):
        handle = len(registered)
        registered.append((address, handle))
        return lambda: cancelled.append(handle)

    capture = LocalIBeaconCapture(register, lambda: [IDENTITY.upper()], lambda _: True, lambda: 100, max_addresses=2)
    capture.observe(packet([100]))
    capture.observe(packet([100], address="AA:BB:CC:DD:EE:02"))
    for _ in range(10):
        capture.observe(packet([100]))
    assert len(registered) == 2
    assert cancelled == []
    capture.observe(packet([100], address="AA:BB:CC:DD:EE:03"))
    assert cancelled == [0]
    capture.close()
    assert sorted(cancelled) == [0, 1, 2]


def test_actual_cache_skip_guard_does_not_prevent_distance_expiry(capture_fixture, monkeypatch):
    """Exercise real coordinator gathering, scanner stamps and advert expiry together."""
    from custom_components.bermuda.coordinator import BermudaDataUpdateCoordinator
    from custom_components.bermuda.bermuda_device import BermudaDevice

    capture, clock, callbacks = capture_fixture
    monkeypatch.setattr("custom_components.bermuda.coordinator.monotonic_time_coarse", lambda: clock[0])
    coor = SimpleNamespace(local_ibeacon=capture, _scanner_init_pending=False, stamp_last_update_started=0)
    scanner = SimpleNamespace(
        address=SOURCE,
        name="Local",
        area_id="office",
        area_name="Office",
        is_scanner=True,
        is_remote_scanner=False,
        last_seen=100.0,
        _coordinator=coor,
        async_as_scanner_update=lambda _: None,
    )
    scanner.async_as_scanner_get_stamp = lambda address: BermudaDevice.async_as_scanner_get_stamp(scanner, address)
    device = SimpleNamespace(
        address=ADDRESS.lower(),
        ref_power=-59,
        name="Phone",
        name_bt_local_name=None,
        make_name=lambda: None,
        process_manufacturer_data=lambda _: None,
    )
    options = dict(rssi_offsets={}, ref_power=-59, attenuation=3, max_velocity=3, smoothing_samples=20)
    links = []
    processed = []

    def process(scanner, advert):
        processed.append(advert.rssi)
        if links:
            links[0].update_advertisement(advert, scanner)
        else:
            links.append(BermudaAdvert(device, advert, options, scanner))

    device.process_advertisement = process
    coor._get_device = lambda _: scanner
    coor._get_or_create_device = lambda _: device
    resolved = []
    coor.irk_manager = SimpleNamespace(scan_device=lambda address: resolved.append(("irk", address)))
    coor.fmdn = SimpleNamespace(handle_advertisement=lambda device, _: resolved.append(("fmdn", device.address)))

    def process_scanner_advertisement(scanner, address, advert):
        BermudaDataUpdateCoordinator._process_scanner_advertisement(coor, scanner, address, advert)

    coor._process_scanner_advertisement = process_scanner_advertisement
    old = packet(clock).advertisement
    old.rssi = -95
    backend = SimpleNamespace(
        source=SOURCE, discovered_devices_and_advertisement_data={ADDRESS: (SimpleNamespace(address=ADDRESS), old)}
    )
    coor._hascanners = [backend]
    capture.observe(packet(clock))
    BermudaDataUpdateCoordinator._async_gather_advert_data(coor)
    links[0].calculate_data()
    assert links[0].rssi == -64
    assert resolved == [("irk", ADDRESS), ("fmdn", ADDRESS.lower())]
    clock[0] = 104
    capture.capture(packet(clock))
    BermudaDataUpdateCoordinator._async_gather_advert_data(coor)
    links[0].calculate_data()
    assert links[0].stamp == 104
    assert links[0].rssi_distance is not None
    # Valid local rotations can evict a subscription, but its cache still
    # cannot replace the last proven packet or refresh distance.
    for suffix in ("02", "03"):
        capture.observe(packet(clock, address="AA:BB:CC:DD:EE:" + suffix))
    assert ADDRESS not in callbacks
    processed.clear()
    clock[0] = 104 + AREA_MAX_AD_AGE_DEFAULT + 1
    coor.stamp_last_update_started = clock[0] - 1
    BermudaDataUpdateCoordinator._async_gather_advert_data(coor)
    assert processed == []  # Actual coordinator cache skip guard was taken.
    assert capture.get_stamp(SOURCE, ADDRESS) == 104
    links[0].calculate_data()  # Runs independently of gathering in normal coordinator updates.
    assert links[0].rssi_distance is None
