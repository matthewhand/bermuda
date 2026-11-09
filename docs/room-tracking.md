# Experimental learned-room tracking

This optional estimator compares current RSSI patterns with labelled room recordings. It adds a separate **Learned room** sensor and existing Area sensors and distance calibration keep their existing behaviour. No additional scanning is started.

## Estimator and movement model

The initial `fingerprint` estimator uses multiple prototypes per room, robust signal residuals, dispersion normalization and receiver weights. Separate positions and body orientations remain separate examples. Missing reception is a soft mismatch; a known unavailable scanner is excluded rather than interpreted as evidence against a room. A single shared receiver cannot produce a confident room match.

The estimator interface and registry allow future modes without coupling signal comparison to movement rules. There is currently one implemented mode. Physical coordinates, metre dimensions and RSSI-to-distance conversion are not required.

The tracker holds the displayed room for up to eight seconds after its last confirmation. The Learned room sensor sets its `held` attribute during this grace period. The model response distinguishes `heldLocation` from a new `location`. Cached observations cannot renew the grace period.

Fresh packets can sustain evidence for a connected room through brief interruptions in the signal ranking. A move relying on two matching receivers requires both readings to be recent. A distant fix requires additional evidence after the previous room has dropped from view. Confirmed results include a `placement` field so clients can distinguish an initial fix, a move, a reacquisition after a gap, or a continued fix. These timing rules are heuristics, not a trained motion probability model.

### Experimental adaptive hybrid

Under **Bermuda → Configure → Experimental room tracking**, enable **Use adaptive hybrid** for an enabled room model. This switch is off by default. The native Area sensor keeps its own selection.

- **Decay half-life in expected packet intervals** defaults to 4. The tracker estimates the interval from recent distinct BLE receipts and allows one interval of grace. Freshness then halves after the configured number of intervals. A held placement expires after three half-lives or 45 seconds, whichever comes first.
- **Minimum adjacent-room travel time** defaults to 4 seconds. It controls movement between connected rooms and still requires repeated packet evidence. It does not extend the old room's hold.
- A trained native UKF area can supply a provisional startup location. A model room uses its optional Home Assistant `areaId` link, or its own identifier when that matches the Area ID. Only a unique match is used. Without a native match, the learned tracker acquires a room from the recorded fingerprints.

The model service can return `provisionalLocation` or `heldLocation` and a `placementFreshness` value for fading a marker. Freshness describes elapsed time, rather than the probability of a correct room. The Learned room sensor flags `provisional` and `held` placements. Consumers that need a confirmed room can use the `location` field.

## Model services

`bermuda.get_room_model` returns the model, current tracking result and unavailable sources. `bermuda.save_room_model` accepts `model` and `expected_revision`; read the current revision before saving. The first model uses expected revision zero. A successful save increments the revision. Concurrent stale saves are rejected. Both services require an administrator user context or an internal automation context.

Models are stored through Home Assistant's Store separately from integration source files. There is no model and tracking remains disabled until one is saved with `enabled: true`. The integration supports one config entry, and this estimator calibrates one configured device.

The version-1 model contains:

- `version: 1`, integer `revision`, boolean `enabled`, `estimator: fingerprint`.
- `config.device.id`: Bermuda's tracked device identifier. Optional `transmitterEntity` supplies the Companion app's `Transmitting power` attribute; missing power is recorded as `unknown`.
- `config.receivers`: one to sixteen objects with unique `key`, optional `roomId`, and optional numeric `weight` from 0.1 to 5.
- `rooms`: objects with unique `id` and `label`. Drawing coordinates are optional and ignored by the estimator.
- `connections`: pairs of connected room identifiers.
- `receiverLayoutVersion`: increment when moving receivers so previous recordings cannot silently train the new arrangement.
- `samples`: up to 120 labelled prototypes. Each has `roomId`, `signature`, and `signals` mapping receiver keys to either null or an object containing `median`, `mad` and `count`.

The signature is `device-id|power|sorted,receiver,keys|layout:N`. Receiver keys should use lowercase scanner addresses, matching Bermuda's observations. Re-record after changing transmitter power, the tracked device, receiver membership or receiver placement.

Null means no measured reception; it is not an invented weak RSSI. A prototype's count is the number of distinct observed packet timestamps, not the number of screen refreshes.

## Limits and verification

The estimator returns discrete room labels using calibrated signal patterns. Similar signal patterns may remain uncertain regardless of the number of recordings. Timings and comparison thresholds need independent household evaluation. A successful replay of training recordings does not establish accuracy on new walks.

Tests cover signal patterns, missing/offline receivers, multiple orientations, ambiguity, distinct-observation confirmation, elapsed-time transitions, expired observations, persistent model round trips, administrator permissions and concurrent revision checks. They do not measure real-world room accuracy or replace the full integration test suite.
