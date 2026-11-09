# Live body simulator: visual mouth

`duck-body` renders the commanded mouth for each duck in the passive MuJoCo
viewer and in enabled simulated camera frames. It uses the same render-only
world for both. The runtime sends targets in radians at wire index 9:
-5 degrees is closed and +30 degrees fully open; the display clamps to this
range. Coin-Coin's audio-synchronized mouth commands therefore animate the
beak during speech. A closed target closes it; torque-off and daemon disconnect
also close the display. Without a closing command, the last target remains
visible while the connection stays open.

The render-only copy rotates the visual jaw meshes around an approximate
hinge; the motion is illustrative, not a calibrated servo or linkage.
Physics, collision meshes, mouth-tip site, inertia, joints, actuators,
observations and the 14-action / 61D policy contract are unchanged. Mouth
position, velocity and current in simulated sensor replies remain zero: this
cannot validate a physical mouth servo or its feedback.

The default model and attached ducks supply the required jaw meshes. A custom
scene lacking either visual mesh logs a warning and has no animated beak.
The head camera is mounted on the duck, so it may not see its own beak; other
ducks' beaks can appear in its view. `--headless` skips the window, while
`--cameras a` (or `--cameras all`) separately opts into camera rendering.


# Hardware realism, no-assist, obstacles and the oracle

Everything here is opt-in. Without the flags below, `duck-body` behaves exactly as before
(`tests/test_sim_realism.py` checks that the `off` trajectory, sensors and ToF frames are
bit-identical).

## Flags and environment

| flag | env (used when the flag is absent) | default | meaning |
|---|---|---|---|
| `--realism off\|real\|hard` | `DUCK_SIM_REALISM` | `off` | sensor, bus and body imperfections |
| `--realism-seed N` (N >= 0) | `DUCK_SIM_REALISM_SEED` | `0` | seed for every realism draw |
| `--no-assist` | `DUCK_SIM_NO_ASSIST=1` (`""`, `0`, `false`, `no` = off) | off | no simulation-only hand; `mode` stops swapping equipment |
| `--obstacles FILE` | `DUCK_SIM_OBSTACLES` | none | JSON obstacles placed at launch |
| `--obstacle-slots N` | `DUCK_SIM_OBSTACLE_SLOTS` | `0`, or `max(8, file count)` with `--obstacles` | how many obstacles can exist at once (runtime `obstacle.add` needs N > 0) |

When any of these is used, the server prints one line at start-up:
`== realism <profile> (seed <n>) · assist on|off · <k> obstacle(s) in <slots> slot(s)`.
Launchers (`duck-sim`, `live_stack.py`) need no change if they export these variables before
starting the body server, because it inherits their environment.

## Realism profiles (`sim/realism.py`)

`real` is the envelope the walking policies were trained in; `hard` goes past it. The source of
each number is cited next to it in `realism.py`. In summary:

| effect | `real` | `hard` | source |
|---|---|---|---|
| gyro white noise (std) | 0.017 rad/s | 0.035 rad/s | training obs noise ±0.03 rad/s |
| gyro bias / random walk | 0.010 rad/s / 0.0005 | 0.0175 / 0.002 | LSM6DSV16X zero-rate ±1 dps |
| projected-gravity noise | 0.0058 | 0.015 | training obs noise ±0.01 |
| IMU mount tilt | ≤2° | ≤6° | training IMU misalignment ≤6° |
| quaternion yaw drift | ~1.7°/min | ~7°/min | SFLP has no magnetometer (assumed rate) |
| quaternion precision | float16 | float16 | SFLP game rotation vector |
| joint position bias / noise | ±0.015 rad / 0.00058 rad | ±0.03 / 0.002 | `ENCODER_BIAS_RANGE`, training obs noise |
| position quantisation | 2π/4096 | same | XL330 encoder |
| velocity noise / quantisation | 0.144 rad/s / 0.229 rpm | 0.3 / same | training obs noise, XL330 velocity unit |
| sensor latency | 0–20 ms | 0–40 ms | training IMU delay (1 tick at 50 Hz) |
| command latency | 15–30 ms | 15–45 ms | BAM actuator delay 3–6 physics steps |
| dropped read / write | 1% / 0.5% | 5% / 2% | assumed; robotd counts stale reads |
| floor+feet friction factor | 0.7–1.3 | 0.5–1.3 | training `foot_friction` |
| per-body mass & inertia | ×0.95–1.05 | ×0.9–1.1 | training mass randomisation |
| per-joint motor kp | ×0.85–1.15 | ×0.8–1.2 | training kp randomisation |
| ToF zone dropout (status 4) | 2% (×2 in corners) | 8% | assumed |
| ToF max range / dark targets | 3.5 m / 1.5 m | 3.0 m / 1.0 m | VL53L5CX datasheet |
| ToF through glass (alpha < 0.6) | ghost 30% else sees through | 60% | multipath, assumed |

A dropped read answers the previous frame again (the daemon closes the link on an error answer,
so the simulator never returns one); a dropped write is silently ignored. Torque off clears the
queue of delayed commands. Draws are seeded by `(seed, duck, stream)` and links are numbered in
connection order, so a run with the same seed and the same connection order repeats exactly.
Friction, mass and kp are drawn per part *name* and survive an equipment change.

`read` keeps its shape. Under realism, `imu.*` and `joints.*` are the *measured* values above;
`trunk`, `trunk_z` and `sim_time` remain the truth (see the oracle below).

## No-assist (`--no-assist`)

By default the simulator gives two hands a robot does not have: it holds the trunk while
equipment is swapped (`mode` with `support: true`) and during assisted fall recovery / the home
ramp, and it holds a duck in place until it is enabled. `--no-assist` removes all of them: a placed
duck simply obeys physics, and a fall or a swap goes as it would on the robot.

On the robot, `RobotIo::set_drive_mode` does nothing to the body — a human fits the skates. So in
no-assist the daemon's `mode` request is accepted (`{}`), validated (`walk`/`roller`) and recorded
as `commanded_mode`, and **the fitted equipment does not change**: the daemon then runs the policy
it selected on whatever is fitted (e.g. the roller policy on feet), exactly like hardware. The
operator or a test changes the equipment with `equipment.fit`. Without `--no-assist`, `mode` swaps
equipment as before, and `equipment.fit` also works.

## Obstacles (`sim/obstacles.py`)

Obstacles collide with the duck, are hit by the ToF rays and are rendered by the head camera and
the viewer. They are kinematic: they push the duck and are not pushed back. The world is compiled
with a fixed number of slots (`--obstacle-slots`), so adding one at runtime does not recompile.

File: `{"obstacles": [ ... ]}` or a bare list; each entry:

    name      unique, required
    shape     box | bag | cylinder | chair | door | parts
    size      box/bag: [x, y, z] full extents (m); cylinder: [diameter, height];
              chair: [width, depth, seat_height, back_height]; door: [width, thickness, height]
    parts     shape "parts" only: [{"size": [x, y, z], "pos": [x, y, z]}, ...] (max 6 boxes)
    pose      {"x", "y", "z" = 0, "yaw_deg" = 0}: world frame, z of the bottom
    rgba      optional; alpha < 0.6 makes glass for the ToF realism model
    motion    optional {"waypoints": [[x, y], ...], "speed": m/s}: ping-pong, a pacing pet

Every obstacle must fit within 4 m.

## Protocol additions

Same line-delimited JSON socket as the daemon's ops. These are for the operator, tests and
scoring — `robotd` never sends them. Errors answer `{"error": "..."}` as usual.

| request | answer |
|---|---|
| `{"op": "equipment.fit", "equipment": "walk"\|"roller"}` | `{"mode": "walk"\|"roller"}` |
| `{"op": "oracle.pose"}` | `{"oracle": true, "x", "y", "z", "yaw", "quat": [w,x,y,z], "sim_time", "mode"}` |
| `{"op": "sim.info"}` | `{"protocol", "duck", "ducks", "realism", "realism_seed", "no_assist", "obstacle_slots", "mode", "commanded_mode"}` |
| `{"op": "obstacle.add", "obstacle": {...entry...}}` | `{"obstacle": {...entry, "pose": {"x","y","z","yaw_deg"}}}` |
| `{"op": "obstacle.move", "name": n, "pose": {"x"?, "y"?, "z"?, "yaw_deg"?}}` | `{"obstacle": {...}}` (stops any `motion`) |
| `{"op": "obstacle.remove", "name": n}` | `{}` |
| `{"op": "obstacle.list"}` | `{"obstacles": [...], "slots": N}` |

`obstacle.*` without slots fails with an error naming `--obstacle-slots`. In no-assist,
`read` also carries `"commanded_mode"`.

## The oracle

`oracle.pose` is the true pose. **It is for scoring a run, never for steering one** — no robot can
know it. Controllers, planners and localisation must use only what the robot has (`read.imu`,
joints, ToF, camera). `read.trunk`, `read.trunk_z` and `read.sim_time` are the same oracle; they stay
for backwards compatibility (Coin-Coin's `SimPose` reads them today) but are **deprecated** for
controllers: new code must call `oracle.pose`, and only to score.