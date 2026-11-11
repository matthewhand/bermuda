"""
Regression tests: the orphan cleanup must only remove entities whose device is really gone.

On a live install every start removed (and then recreated) almost all Bermuda entities:

1. async_setup_entry() ran the cleanup right after the first refresh, which returns early
   while the Bluetooth ID data is loading, so coordinator.devices was empty or partial.
2. Even with a complete device list, base-address extraction failed for real unique_ids:
   iBeacon addresses contain underscores, scanner entities use the WiFi MAC, and
   "_training_floor" was cut as "_floor".
3. The duplicate check keyed entities by suffix only, so one device's per-scanner
   sensors (and the global sensors) counted as duplicates of each other.

Each removal fires an entity registry "remove" event; Proximity, for example, turns that
into a tracked_entity_removed repair.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.bermuda as bermuda_init
from custom_components.bermuda.const import DOMAIN
from custom_components.bermuda.coordinator import BermudaDataUpdateCoordinator

from .test_coordinator_methods import _make_coordinator

MAC = "aa:bb:cc:dd:ee:ff"
SCANNER_BLE = "88:57:21:b6:44:3a"
SCANNER_WIFI = "88:57:21:b6:44:38"
IBEACON = "50765cb7d9ea4e2199a4fa879613a492_12591_47910"


def _setup(hass: HomeAssistant, unique_ids: list[tuple[str, str]]):
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    coordinator = _make_coordinator(hass)
    coordinator.config_entry = entry
    registry = er.async_get(hass)
    entities = [
        registry.async_get_or_create(domain, DOMAIN, unique_id, config_entry=entry).entity_id
        for domain, unique_id in unique_ids
    ]
    removed: list[str] = []
    hass.bus.async_listen(
        er.EVENT_ENTITY_REGISTRY_UPDATED,
        lambda event: removed.append(event.data["entity_id"]) if event.data["action"] == "remove" else None,
    )
    return coordinator, entities, removed


async def _cleanup(hass: HomeAssistant, coordinator) -> None:
    await coordinator.async_cleanup_orphaned_entities()
    await hass.async_block_till_done()


async def test_cleanup_skipped_while_bt_ids_loading(hass: HomeAssistant) -> None:
    """A pending ID load means the device list was never built: nothing may be removed."""
    coordinator, _, removed = _setup(hass, [("device_tracker", MAC)])
    coordinator._get_or_create_device("11:22:33:44:55:66")
    coordinator._waitingfor_load_manufacturer_ids = True

    await _cleanup(hass, coordinator)

    assert removed == []


async def test_cleanup_skipped_with_empty_device_list(hass: HomeAssistant) -> None:
    """An empty device list is never evidence that every entity is orphaned."""
    coordinator, _, removed = _setup(hass, [("device_tracker", MAC)])

    await _cleanup(hass, coordinator)

    assert removed == []


async def test_cleanup_keeps_ibeacon_per_scanner_entities(hass: HomeAssistant) -> None:
    """iBeacon addresses contain underscores; their per-scanner and training entities stay."""
    coordinator, _, removed = _setup(
        hass,
        [
            ("sensor", f"{IBEACON}_{SCANNER_BLE}_range"),
            ("sensor", f"{IBEACON}_{SCANNER_BLE}_range_raw"),
            ("select", f"{IBEACON}_training_floor"),
            ("number", f"{IBEACON}_ref_power"),
            ("device_tracker", IBEACON),
        ],
    )
    coordinator._get_or_create_device(IBEACON)

    await _cleanup(hass, coordinator)

    assert removed == []


async def test_cleanup_keeps_scanner_entities_keyed_by_wifi_mac(hass: HomeAssistant) -> None:
    """Scanner entities use the WiFi MAC while the device is keyed by its BLE MAC."""
    coordinator, _, removed = _setup(hass, [("binary_sensor", f"{SCANNER_WIFI}_scanner_online")])
    coordinator._get_or_create_device(SCANNER_BLE).address_wifi_mac = SCANNER_WIFI

    await _cleanup(hass, coordinator)

    assert removed == []


async def test_cleanup_still_removes_real_orphans(hass: HomeAssistant) -> None:
    """Entities of addresses that no device has are still removed."""
    coordinator, entities, removed = _setup(hass, [("device_tracker", MAC), ("sensor", f"{MAC}_{SCANNER_BLE}_range")])
    coordinator._get_or_create_device("11:22:33:44:55:66")

    await _cleanup(hass, coordinator)

    assert sorted(removed) == sorted(entities)


async def test_setup_defers_cleanup(hass: HomeAssistant) -> None:
    """Setup schedules the cleanup for later instead of running it after the first refresh."""
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    with (
        patch("custom_components.bermuda.BermudaDataUpdateCoordinator") as coordinator_cls,
        patch("custom_components.bermuda.async_call_later", create=True) as call_later,
        patch.object(hass.config_entries, "async_forward_entry_setups"),
    ):
        coordinator = coordinator_cls.return_value
        coordinator.async_cleanup_device_registry_connections = _noop
        coordinator.async_refresh = _noop
        coordinator.room_tracking.async_load = _noop
        coordinator.last_update_success = True
        coordinator.async_cleanup_orphaned_entities = _fail
        await bermuda_init.async_setup_entry(hass, entry)

    call_later.assert_called_once()
    assert call_later.call_args.args[1] == getattr(bermuda_init, "ORPHAN_CLEANUP_DELAY", None)


async def test_deferred_cleanup_runs_orphan_cleanup(hass: HomeAssistant) -> None:
    """The callback scheduled by setup runs the orphan cleanup when the delay has passed."""
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    calls: list[None] = []

    async def _record(*_args, **_kwargs) -> None:
        calls.append(None)

    with (
        patch("custom_components.bermuda.BermudaDataUpdateCoordinator") as coordinator_cls,
        patch("custom_components.bermuda.async_call_later", create=True) as call_later,
        patch.object(hass.config_entries, "async_forward_entry_setups"),
    ):
        coordinator = coordinator_cls.return_value
        coordinator.async_cleanup_device_registry_connections = _noop
        coordinator.async_refresh = _noop
        coordinator.room_tracking.async_load = _noop
        coordinator.last_update_success = True
        coordinator.async_cleanup_orphaned_entities = _record
        await bermuda_init.async_setup_entry(hass, entry)
        assert calls == []
        await call_later.call_args.args[2](None)

    assert calls == [None]


async def _noop(*_args, **_kwargs) -> None:
    return None


async def _fail(*_args, **_kwargs) -> None:
    raise AssertionError("orphan cleanup must not run during setup")


async def test_cleanup_keeps_per_scanner_entities_of_one_device(hass: HomeAssistant) -> None:
    """One device has a distance sensor per scanner; they are not duplicates of each other."""
    other_scanner = "6c:c8:40:86:7b:a6"
    coordinator, _, removed = _setup(
        hass,
        [
            ("sensor", f"{MAC}_{SCANNER_BLE}_range"),
            ("sensor", f"{MAC}_{other_scanner}_range"),
            ("sensor", f"{MAC}_{SCANNER_BLE}_range_raw"),
            ("sensor", f"{MAC}_{other_scanner}_range_raw"),
        ],
    )
    coordinator._get_or_create_device(MAC)
    _attach_to_one_device(hass, coordinator)

    await _cleanup(hass, coordinator)

    assert removed == []


async def test_cleanup_keeps_entities_of_scanner_tracked_as_device(hass: HomeAssistant) -> None:
    """
    A scanner that is also a tracked device: devices is keyed by its BLE MAC, its unique_ids use the WiFi MAC.

    The duplicate key must come from the prefix that matched (the WiFi MAC), not from the
    devices key, or the per-scanner sensors of one kind share a key and are removed as duplicates.
    """
    other_scanner = "6c:c8:40:86:7b:a6"
    coordinator, _, removed = _setup(
        hass,
        [
            ("sensor", SCANNER_WIFI),
            ("sensor", f"{SCANNER_WIFI}_range"),
            ("sensor", f"{SCANNER_WIFI}_{other_scanner}_range"),
            ("sensor", f"{SCANNER_WIFI}_{MAC}_range"),
            ("sensor", f"{SCANNER_WIFI}_{other_scanner}_range_raw"),
            ("sensor", f"{SCANNER_WIFI}_{MAC}_range_raw"),
            ("sensor", f"{SCANNER_WIFI}_estimated_broadcast_interval"),
        ],
    )
    coordinator._get_or_create_device(SCANNER_BLE).address_wifi_mac = SCANNER_WIFI
    _attach_to_one_device(hass, coordinator)

    await _cleanup(hass, coordinator)

    assert removed == []


async def test_cleanup_keeps_live_prefix_of_scanner_duplicates(hass: HomeAssistant) -> None:
    """
    Real duplicates under the BLE and the WiFi prefix: the entity with the live prefix stays.

    Entities are created with BermudaDevice.unique_id, which is the WiFi MAC for such a
    scanner. Leftovers under the BLE MAC are the stale copies, even though the BLE MAC is
    the devices key.
    """
    other_scanner = "6c:c8:40:86:7b:a6"
    coordinator, entities, removed = _setup(
        hass,
        [
            # Stale copies first, so registry order alone would keep them.
            ("sensor", f"{SCANNER_BLE}_estimated_broadcast_interval"),
            ("sensor", f"{SCANNER_WIFI}_estimated_broadcast_interval"),
            ("sensor", f"{SCANNER_BLE}_{other_scanner}_range"),
            ("sensor", f"{SCANNER_WIFI}_{other_scanner}_range"),
        ],
    )
    scanner = coordinator._get_or_create_device(SCANNER_BLE)
    scanner.address_wifi_mac = SCANNER_WIFI
    scanner.unique_id = SCANNER_WIFI
    _attach_to_one_device(hass, coordinator)

    await _cleanup(hass, coordinator)

    assert sorted(removed) == sorted([entities[0], entities[2]])


async def test_cleanup_keeps_disabled_live_entity_over_enabled_stale_copy(hass: HomeAssistant) -> None:
    """
    The live prefix outranks the enabled state among duplicates.

    A scanner moved from its BLE to its WiFi prefix: the user had enabled the old diagnostic
    entity, the new one is disabled by default. The stale copy only shows unavailable, and
    the live unique_id is the one the platforms create, so the stale copy is removed.
    """
    coordinator, entities, removed = _setup(
        hass,
        [
            ("sensor", f"{SCANNER_BLE}_estimated_broadcast_interval"),
            ("sensor", f"{SCANNER_WIFI}_estimated_broadcast_interval"),
        ],
    )
    er.async_get(hass).async_update_entity(entities[1], disabled_by=er.RegistryEntryDisabler.INTEGRATION)
    scanner = coordinator._get_or_create_device(SCANNER_BLE)
    scanner.address_wifi_mac = SCANNER_WIFI
    scanner.unique_id = SCANNER_WIFI
    _attach_to_one_device(hass, coordinator)

    await _cleanup(hass, coordinator)

    assert removed == [entities[0]]


async def test_cleanup_keeps_ibeacon_on_tracked_scanner_device(hass: HomeAssistant) -> None:
    """
    An iBeacon sent by a scanner shares that scanner's HA device; with the scanner tracked, both are live.

    Their entities have equal remainders after the prefix ("", "_range", the per-scanner
    sensors), so they share duplicate keys. Entities under a live prefix are never duplicates
    of each other: none of the eight may be removed.
    """
    other_scanner = "6c:c8:40:86:7b:a6"
    coordinator, _, removed = _setup(
        hass,
        [
            ("device_tracker", SCANNER_WIFI),
            ("sensor", SCANNER_WIFI),
            ("sensor", f"{SCANNER_WIFI}_range"),
            ("sensor", f"{SCANNER_WIFI}_{other_scanner}_range"),
            ("device_tracker", IBEACON),
            ("sensor", IBEACON),
            ("sensor", f"{IBEACON}_range"),
            ("sensor", f"{IBEACON}_{other_scanner}_range"),
        ],
    )
    scanner = coordinator._get_or_create_device(SCANNER_BLE)
    scanner.address_wifi_mac = SCANNER_WIFI
    scanner.unique_id = SCANNER_WIFI
    coordinator._get_or_create_device(IBEACON)
    _attach_to_one_device(hass, coordinator)

    await _cleanup(hass, coordinator)

    assert removed == []


async def test_cleanup_keeps_entities_under_device_unique_id(hass: HomeAssistant) -> None:
    """A device whose unique_id differs from its devices key and has no WiFi MAC keeps its entities."""
    registry_mac = "12:34:56:78:9a:bc"
    coordinator, _, removed = _setup(hass, [("sensor", registry_mac), ("sensor", f"{registry_mac}_range")])
    coordinator._get_or_create_device(SCANNER_BLE).unique_id = registry_mac

    await _cleanup(hass, coordinator)

    assert removed == []


async def test_cleanup_keeps_exact_case_live_entity_among_duplicates(hass: HomeAssistant) -> None:
    """A stale copy that differs from the live unique_id only in case is the one removed."""
    coordinator, entities, removed = _setup(
        hass,
        [
            # Stale upper-case copy first, so registry order alone would keep it.
            ("sensor", f"{MAC.upper()}_floor"),
            ("sensor", f"{MAC}_floor"),
        ],
    )
    coordinator._get_or_create_device(MAC)
    _attach_to_one_device(hass, coordinator)

    await _cleanup(hass, coordinator)

    assert removed == [entities[0]]


async def test_cleanup_keeps_global_entities(hass: HomeAssistant) -> None:
    """The global sensors share one device but are distinct entities."""
    coordinator, _, removed = _setup(
        hass,
        [
            ("sensor", "BERMUDA_GLOBAL_PROXY_COUNT"),
            ("sensor", "BERMUDA_GLOBAL_ACTIVE_PROXY_COUNT"),
            ("sensor", "BERMUDA_GLOBAL_DEVICE_COUNT"),
            ("sensor", "BERMUDA_GLOBAL_VISIBLE_DEVICE_COUNT"),
        ],
    )
    coordinator._get_or_create_device(MAC)
    _attach_to_one_device(hass, coordinator)

    await _cleanup(hass, coordinator)

    assert removed == []


async def test_cleanup_removes_old_address_format_duplicate(hass: HomeAssistant) -> None:
    """Same device under an old and a new address: the entity of the unknown address goes."""
    coordinator, entities, removed = _setup(hass, [("sensor", f"{MAC}_floor"), ("sensor", "fmdn:old-format-id_floor")])
    coordinator._get_or_create_device(MAC)
    _attach_to_one_device(hass, coordinator)

    with patch.object(coordinator, "_check_address_exists", return_value=True):
        await _cleanup(hass, coordinator)

    assert removed == [entities[1]]


def _attach_to_one_device(hass: HomeAssistant, coordinator) -> None:
    """Put all of the entry's entities on one HA device, as the platforms do."""
    device = dr.async_get(hass).async_get_or_create(
        config_entry_id=coordinator.config_entry.entry_id, identifiers={(DOMAIN, "shared")}
    )
    registry = er.async_get(hass)
    for entity in er.async_entries_for_config_entry(registry, coordinator.config_entry.entry_id):
        registry.async_update_entity(entity.entity_id, device_id=device.id)


def _linear_reference(self: BermudaDataUpdateCoordinator, unique_id: str) -> tuple[str, str] | None:
    """Reference: the linear prefix search the cleanup used before the per-run index."""
    uid = unique_id.lower()
    best: tuple[str, str] | None = None
    for address, device in self.devices.items():
        for candidate in (address, getattr(device, "address_wifi_mac", None), getattr(device, "unique_id", None)):
            if not candidate:
                continue
            cand = candidate.lower()
            if (uid == cand or uid.startswith(cand + "_")) and (best is None or len(cand) > len(best[1])):
                best = (address, cand)
    return best


def _oracle_devices(coordinator: BermudaDataUpdateCoordinator) -> None:
    """Devices for the prefix oracle; insertion order matters for the shared candidate."""
    coordinator._get_or_create_device(IBEACON)
    wifi = coordinator._get_or_create_device(SCANNER_BLE)
    wifi.address_wifi_mac = SCANNER_WIFI
    wifi.unique_id = SCANNER_WIFI
    own_id = coordinator._get_or_create_device("12:34:56:78:9a:bc")
    own_id.unique_id = "registry-id"
    first = coordinator._get_or_create_device("02:00:00:00:00:01")
    first.address_wifi_mac = "02:00:00:00:00:aa"
    second = coordinator._get_or_create_device("02:00:00:00:00:02")
    second.address_wifi_mac = "02:00:00:00:00:aa"
    outer = coordinator._get_or_create_device("02:00:00:00:00:03")
    outer.unique_id = "nest_a"
    inner = coordinator._get_or_create_device("02:00:00:00:00:04")
    inner.unique_id = "nest_a_b"
    upper = coordinator._get_or_create_device("02:00:00:00:00:05")
    upper.unique_id = "0A:BC:DE:F0:12:34"  # scanner source address, upper case
    empty = coordinator._get_or_create_device("02:00:00:00:00:06")
    empty.address_wifi_mac = ""
    trailing = coordinator._get_or_create_device("02:00:00:00:00:07")
    trailing.unique_id = "trail_"


@pytest.mark.parametrize(
    "unique_ids",
    [
        [f"{IBEACON}_range", f"{IBEACON}_{SCANNER_BLE}_range", f"{IBEACON}_training_floor"],
        [f"{SCANNER_WIFI}_{MAC}_range", f"{SCANNER_BLE}_floor", f"{SCANNER_WIFI.upper()}_range"],
        ["registry-id_range", "registry-id"],
        ["02:00:00:00:00:aa_range", "02:00:00:00:00:aa"],
        ["nest_a_b_floor", "nest_a_floor", "nest_a"],
        ["0A:BC:DE:F0:12:34_range", "0a:bc:de:f0:12:34_range"],
        [SCANNER_WIFI, "nest_a_b", "registry-id"],
        ["_floor", "__range"],
        ["trail__range", "trail_"],
        ["ff:ff:ff:ff:ff:ff_range", "unknown", ""],
    ],
    ids=[
        "ibeacon",
        "wifi_ungleich_schluessel",
        "unique_id_kandidat",
        "geteilt",
        "verschachtelt",
        "gross",
        "uid_gleich_kandidat",
        "leer_fuehrend",
        "unterstrich_am_ende",
        "kein_treffer",
    ],
)
async def test_prefix_match_equals_linear_reference(hass: HomeAssistant, unique_ids: list[str]) -> None:
    """The prefix match of the cleanup resolves every unique_id exactly like the linear reference."""
    coordinator = _make_coordinator(hass)
    _oracle_devices(coordinator)

    for unique_id in unique_ids:
        assert coordinator._match_known_unique_id_prefix(unique_id) == _linear_reference(coordinator, unique_id), (
            unique_id
        )


async def test_cleanup_keeps_live_entity_when_device_unique_id_is_none(hass: HomeAssistant) -> None:
    """Without a unique_id the devices key is the live prefix: the copy differing in case goes."""
    coordinator, entities, removed = _setup(
        hass,
        [
            # Stale upper-case copy first, so registry order alone would keep it.
            ("sensor", f"{MAC.upper()}_floor"),
            ("sensor", f"{MAC}_floor"),
        ],
    )
    coordinator._get_or_create_device(MAC).unique_id = None
    _attach_to_one_device(hass, coordinator)

    await _cleanup(hass, coordinator)

    assert removed == [entities[0]]


async def test_cleanup_builds_prefix_index_once(hass: HomeAssistant) -> None:
    """One cleanup run resolves all entities through a single index instead of rebuilding it per lookup."""
    coordinator, _, removed = _setup(
        hass,
        [
            ("sensor", f"{MAC.upper()}_floor"),
            ("sensor", f"{MAC}_floor"),
            ("sensor", f"{MAC}_range"),
            ("sensor", "ff:ff:ff:ff:ff:ff_range"),
        ],
    )
    coordinator._get_or_create_device(MAC)
    _attach_to_one_device(hass, coordinator)

    with patch.object(
        coordinator, "_build_unique_id_prefix_index", wraps=coordinator._build_unique_id_prefix_index
    ) as build:
        await _cleanup(hass, coordinator)

    assert build.call_count == 1
    assert len(removed) == 2
