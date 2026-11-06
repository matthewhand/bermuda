# Experimental learned-room tracking

This optional estimator compares current RSSI patterns with labelled room recordings. It adds a separate **Learned room** sensor; existing Area sensors and distance calibration keep their existing behaviour. No additional scanning is started.

## Estimator and movement model

The initial `fingerprint` estimator uses multiple prototypes per room, robust signal residuals, dispersion normalization and receiver weights. Separate positions and body orientations remain separate examples. Missing reception is a soft mismatch; a known unavailable scanner is excluded rather than interpreted as evidence against a room. A single shared receiver cannot produce a confident room match.

The estimator interface and registry allow future modes without coupling signal comparison to movement rules. There is currently one implemented mode. Physical coordinates, metre dimensions and RSSI-to-distance conversion are not required.

An independent temporal tracker requires distinct observations and elapsed dwell time. Room connections constrain transition timing. Unconnected rooms require longer confirmation; an old location is labelled last known rather than presented as a fresh fix. Optional room groups allow a coarse open-plan estimate when individual zones are indistinguishable. These timing rules are heuristics, not a trained motion probability model.

## Model services

`bermuda.get_room_model` returns the model, current tracking result and unavailable sources. `bermuda.save_room_model` accepts `model` and `expected_revision`; read the current revision before saving. The first model uses expected revision zero. A successful save increments the revision. Concurrent stale saves are rejected. Saves require an administrator user context, or an internal automation context.

Models are stored through Home Assistant's Store separately from integration source files. There is no model and tracking remains disabled until one is saved with `enabled: true`. The initial implementation tracks one configured device per integration entry.

The version-1 model contains:

- `version: 1`, integer `revision`, boolean `enabled`, `estimator: fingerprint`.
- `config.device.id`: Bermuda's tracked device identifier. Optional `transmitterEntity` supplies the Companion app's `Transmitting power` attribute; missing power is recorded as `unknown`.
- `config.receivers`: one to sixteen objects with unique `key`, optional `roomId`, and optional numeric `weight` from 0.1 to 5.
- `rooms`: objects with unique `id` and `label`. Drawing coordinates are optional and ignored by the estimator.
- `connections`: pairs of connected room identifiers.
- Optional `roomGroups`: objects with `id`, `label` and a list of `rooms`.
- `receiverLayoutVersion`: increment when moving receivers so previous recordings cannot silently train the new arrangement.
- `samples`: up to 120 labelled prototypes. Each has `roomId`, `signature`, and `signals` mapping receiver keys to either null or an object containing `median`, `mad` and `count`.

The signature is `device-id|power|sorted,receiver,keys|layout:N`. Receiver keys should use lowercase scanner addresses, matching Bermuda's observations. Re-record after changing transmitter power, the tracked device, receiver membership or receiver placement.

Null means no measured reception; it is not an invented weak RSSI. A prototype's count is the number of distinct observed packet timestamps, not the number of screen refreshes.

## Limits and verification

This is room classification, not geometric triangulation or a continuous physical position. Similar signal patterns may remain uncertain regardless of the number of recordings. Timings and comparison thresholds need independent household evaluation. A successful replay of training recordings does not establish accuracy on new walks.

Tests cover signal patterns, missing/offline receivers, multiple orientations, ambiguity, distinct-observation confirmation, elapsed-time transitions, expired observations, persistent model round trips, administrator permissions and concurrent revision checks. They do not measure real-world room accuracy or replace the full integration test suite.


## Walk-test recorder and replay

The optional `bermuda.walk_test` response service supports `status`, `start`, `stop`, `get`, `annotate` and `delete`. Mutations require an administrator context (or internal automation). Start accepts a label and `setup_confirmed` boolean; leave it false for diagnostic walks with receivers moved from calibration positions. Get/stop/delete use `walk_id`; annotate additionally uses `expected_revision`, `from_ms`, `to_ms` and `room_id`.

Walks are stored separately from the calibration model. Each captures an immutable model snapshot, initial temporal state, RSSI vectors, actual packet evidence identities, unavailable sources and the backend's results on normal coordinator updates. There are no new radio subscriptions. Recording continues without a dashboard and stops after ten minutes or if the tracked signature/model revision changes. Up to six walks are retained; export and explicitly delete older walks to make space. Periodic saves occur every fifteen seconds; graceful manual stops save immediately. After a restart an unfinished walk is marked interrupted rather than silently resumed.

Room interval annotations evaluate a walk; they never become training samples. Overlapping intervals and stale annotation revisions are rejected. Replay reports exact room accuracy only over independently labelled, observed time. Coarse group matches, uncertainty and gaps are separate quantities. Receiver placement not confirmed means diagnostic-only; no accuracy claim is appropriate. A replay need not agree exactly with recorded predictions after a deliberate estimator change. Candidate calibration comparisons start with a cold temporal state.

The mobile frontend includes walk controls and a portable offline replay page. The browser replay can export its report and run without a Home Assistant connection. A Node CLI is also available in the frontend source: `node presence-replay.js walk.json [candidate-model.json]`.
