"""
Regression tests for per-cycle state writes (state_changed flood).

Home Assistant only fires ``state_changed`` (and the recorder only writes a
``states`` row) when the state or the *complete* attribute dict differs from
the previous write. ``_unrecorded_attributes`` does not take part in that
decision; it only filters what the recorder stores. Attributes whose value is
derived from ``monotonic_time_coarse()`` therefore force one ``state_changed``
per coordinator cycle (~1.05 s) for every affected entity.

Expected results (commit "add regression tests" / branch head):

=====================================================  ==========  ====
Test                                                    tests only  head
=====================================================  ==========  ====
test_area_floor_distance_attributes_stable_across_cycles  fail      pass
test_scanner_online_attributes_stable_across_cycles       fail      pass
test_no_second_state_changed_without_change               fail      pass
test_count_active_scanners_computed_once_per_cycle        fail      pass
test_area_floor_distance_attribute_keys_exact             fail      pass
test_scanner_online_attribute_keys_exact                  fail      pass
test_area_flags_change_only_on_transition                 pass      pass
test_count_active_scanners_fresh_outside_cycle            pass      pass
=====================================================  ==========  ====
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from homeassistant.const import EVENT_STATE_CHANGED
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback
from pytest_homeassistant_custom_component.common import MockEntityPlatform

from custom_components.bermuda import coordinator as coordinator_module
from custom_components.bermuda.binary_sensor import BermudaScannerOnlineSensor
from custom_components.bermuda.bermuda_device import BermudaDevice
from custom_components.bermuda.const import AREA_MAX_AD_AGE, CONF_DEVTRACK_TIMEOUT, SCANNER_ACTIVITY_TIMEOUT
from custom_components.bermuda.coordinator import BermudaDataUpdateCoordinator
from custom_components.bermuda.sensor import BermudaSensor, BermudaSensorFloor, BermudaSensorRange

from .test_coordinator_methods import _make_coordinator

CYCLE = 1.05


def _patch_time(monkeypatch: pytest.MonkeyPatch, now: list[float]) -> None:
    """Route every monotonic_time_coarse used here through ``now[0]``."""
    for target in (
        "bluetooth_data_tools.monotonic_time_coarse",
        "custom_components.bermuda.bermuda_device.monotonic_time_coarse",
        "custom_components.bermuda.coordinator.monotonic_time_coarse",
        "custom_components.bermuda.binary_sensor.monotonic_time_coarse",
    ):
        monkeypatch.setattr(target, lambda: now[0])


def _area_device(coordinator: BermudaDataUpdateCoordinator, now: float) -> BermudaDevice:
    """Real BermudaDevice with a fresh, applied area selection."""
    device = coordinator._get_or_create_device("AA:BB:CC:DD:EE:01")
    device.area_id = "living_room"
    device.area_name = "Living Room"
    device.area_state_stamp = now - 5
    device.area_distance_stamp = now - 5
    device.area_state_source = "11:22:33:44:55:66"
    device.current_mac = "aa:bb:cc:dd:ee:01"
    return device


def _sensor(cls: type[BermudaSensor], device: BermudaDevice) -> BermudaSensor:
    sensor = object.__new__(cls)
    sensor._device = device
    return sensor


def _attrs(entity: Any) -> dict[str, Any]:
    return dict(entity.extra_state_attributes or {})


@pytest.mark.parametrize("cls", [BermudaSensor, BermudaSensorFloor, BermudaSensorRange])
def test_area_floor_distance_attributes_stable_across_cycles(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, cls: type[BermudaSensor]
) -> None:
    """Without a change in the device, attributes must be identical between cycles."""
    now = [1000.0]
    _patch_time(monkeypatch, now)
    coordinator = _make_coordinator(hass)
    sensor = _sensor(cls, _area_device(coordinator, now[0]))

    first = _attrs(sensor)
    for offset in (CYCLE, 30.0):
        now[0] = 1000.0 + offset
        assert _attrs(sensor) == first, f"attributes changed after {offset}s without a device change"


def test_area_flags_change_only_on_transition(hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch) -> None:
    """Crossing AREA_MAX_AD_AGE flips area_is_stale; the stable keys stay unchanged."""
    now = [1000.0]
    _patch_time(monkeypatch, now)
    coordinator = _make_coordinator(hass)
    device = _area_device(coordinator, now[0])
    sensor = _sensor(BermudaSensor, device)

    before = _attrs(sensor)
    now[0] = device.area_state_stamp + AREA_MAX_AD_AGE + 1  # type: ignore[operator]
    after = _attrs(sensor)

    assert before["area_is_stale"] is False
    assert after["area_is_stale"] is True
    stable = ("area_id", "area_name", "floor_id", "floor_name", "floor_level", "current_mac", "area_source")
    assert {k: before[k] for k in stable} == {k: after[k] for k in stable}


def _scanner_online_sensor(last_seen: float) -> BermudaScannerOnlineSensor:
    device = MagicMock()
    device.last_seen = last_seen
    sensor = object.__new__(BermudaScannerOnlineSensor)
    sensor._device = device
    return sensor


def test_scanner_online_attributes_stable_across_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    """While is_on does not change, the scanner_online attributes must not change either."""
    now = [1000.0]
    _patch_time(monkeypatch, now)
    sensor = _scanner_online_sensor(last_seen=998.0)

    first_on, first = sensor.is_on, _attrs(sensor)
    now[0] += CYCLE
    assert sensor.is_on == first_on
    assert _attrs(sensor) == first


async def test_no_second_state_changed_without_change(hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Two coordinator updates without a device change must not fire state_changed.

    Drives the real entity write path (_handle_coordinator_update ->
    async_write_ha_state) against the real HA state machine.
    """
    now = [1000.0]
    _patch_time(monkeypatch, now)
    coordinator = _make_coordinator(hass)
    coordinator.last_update_success = True
    # The lightweight test coordinator skips DataUpdateCoordinator.__init__.
    monkeypatch.setattr(coordinator, "async_add_listener", MagicMock(return_value=lambda: None))
    device = _area_device(coordinator, now[0])
    sensor = BermudaSensor(coordinator, SimpleNamespace(options={}), device.address)  # type: ignore[arg-type]
    sensor.entity_id = "sensor.dedup_probe_area"
    platform = MockEntityPlatform(hass, domain="sensor", platform_name="bermuda")
    await platform.async_add_entities([sensor])
    await hass.async_block_till_done()

    events: list[Event[EventStateChangedData]] = []

    @callback
    def _listener(event: Event[EventStateChangedData]) -> None:
        events.append(event)

    unsub = hass.bus.async_listen(EVENT_STATE_CHANGED, _listener)
    for _ in range(2):
        now[0] += CYCLE
        sensor._handle_coordinator_update()
        await hass.async_block_till_done()
    unsub()

    state = hass.states.get(sensor.entity_id)
    assert state is not None
    assert state.state == "Living Room"
    assert len(events) == 0


AREA_KEYS = {
    "area_id",
    "area_name",
    "floor_id",
    "floor_name",
    "floor_level",
    "area_is_stale",
    "area_retained",
    "area_source",
    "current_mac",
}
DISTANCE_KEYS = {"area_is_stale", "area_retained", "area_source", "current_mac"}


@pytest.mark.parametrize(
    ("cls", "expected"),
    [(BermudaSensor, AREA_KEYS), (BermudaSensorFloor, AREA_KEYS), (BermudaSensorRange, DISTANCE_KEYS)],
)
def test_area_floor_distance_attribute_keys_exact(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, cls: type[BermudaSensor], expected: set[str]
) -> None:
    """The published attribute keys are exactly the stable, functional ones."""
    now = [1000.0]
    _patch_time(monkeypatch, now)
    coordinator = _make_coordinator(hass)
    sensor = _sensor(cls, _area_device(coordinator, now[0]))

    assert set(_attrs(sensor)) == expected


@pytest.mark.parametrize("last_seen", [0.0, 998.0])
def test_scanner_online_attribute_keys_exact(monkeypatch: pytest.MonkeyPatch, last_seen: float) -> None:
    """The scanner online sensor only publishes its timeout."""
    now = [1000.0]
    _patch_time(monkeypatch, now)

    assert _attrs(_scanner_online_sensor(last_seen=last_seen)) == {"timeout_seconds": SCANNER_ACTIVITY_TIMEOUT}


def _cycle_coordinator(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, now: list[float], stale_devices: int
) -> tuple[BermudaDataUpdateCoordinator, list[int]]:
    """Coordinator with 3 active scanners and ``stale_devices`` devices past their tracker timeout."""
    coordinator = _make_coordinator(hass)
    coordinator.options[CONF_DEVTRACK_TIMEOUT] = 30
    coordinator.room_tracking = MagicMock()
    for idx in range(3):
        scanner = coordinator._get_or_create_device(f"11:22:33:44:55:{idx:02X}")
        scanner.last_seen = now[0] - 1
        coordinator._scanners.add(scanner)
    for idx in range(stale_devices):
        device = coordinator._get_or_create_device(f"AA:BB:CC:00:00:{idx:02X}")
        device.last_seen = now[0] - 3600
        device.options[CONF_DEVTRACK_TIMEOUT] = 30

    # Isolate the per-device loop: the steps around it are not under test.
    for name in (
        "_async_gather_advert_data",
        "update_metadevices",
        "_refresh_areas_by_min_distance",
        "aggregate_source_data_to_metadevices",
        "prune_devices",
    ):
        monkeypatch.setattr(coordinator, name, MagicMock(return_value=True))
    monkeypatch.setattr(coordinator_module, "update_scanner_calibration", MagicMock())
    coordinator.scanner_calibration = MagicMock()

    scans = [0]
    original = BermudaDataUpdateCoordinator.get_scanners

    def _counting(self: BermudaDataUpdateCoordinator) -> set[BermudaDevice]:
        scans[0] += 1
        return original.fget(self)  # type: ignore[no-any-return,attr-defined]

    monkeypatch.setattr(BermudaDataUpdateCoordinator, "get_scanners", property(_counting))
    return coordinator, scans


def test_count_active_scanners_computed_once_per_cycle(hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch) -> None:
    """50 stale devices in one cycle must not iterate the scanner set 50 times."""
    now = [5000.0]
    _patch_time(monkeypatch, now)
    coordinator, scans = _cycle_coordinator(hass, monkeypatch, now, stale_devices=50)

    assert coordinator._async_update_data_internal() is True

    assert scans[0] == 1, f"active scanners computed {scans[0]} times in one cycle"
    assert coordinator.update_in_progress is False


def test_count_active_scanners_fresh_outside_cycle(hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside an update cycle every call sees the current scanner state."""
    now = [5000.0]
    _patch_time(monkeypatch, now)
    coordinator, _scans = _cycle_coordinator(hass, monkeypatch, now, stale_devices=0)

    assert coordinator.count_active_scanners() == 3
    next(iter(coordinator._scanners)).last_seen = now[0] - 3600
    assert coordinator.count_active_scanners() == 2
