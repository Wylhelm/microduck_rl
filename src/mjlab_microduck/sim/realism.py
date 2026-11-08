"""Opt-in sensor, link and body imperfections, so the twin stops being better than the robot.

    duck-body --realism real --realism-seed 7      # what the training envelope says hardware is
    duck-body --realism hard                       # past that envelope, to find the margins

`off` (the default) changes nothing: no random draw, no buffer, no model edit — the body server
behaves exactly as it did before this module existed, and a test locks that in.

**Where the numbers come from.** Every magnitude below is either a datasheet figure or a value the
policies were already trained against (`tasks/microduck_velocity_env_cfg.py`, `robot/
microduck_constants.py`), cited beside it. `real` stays inside the training envelope: a policy that
falls under `real` has a sim2real gap worth knowing about. `hard` goes past it on purpose.

**Reproducible.** Every stream is its own `numpy` generator, seeded from `(seed, duck, stream)`
through `SeedSequence`, so adding a consumer to one stream never shifts another, and per-part draws
(a body's mass, a motor's gain) are keyed by the part's *name* — the same duck gets the same
motors after an equipment swap recompiles its model.
"""

from __future__ import annotations

import math
import zlib
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

# XL330: 4096 counts per turn (e-manual, "Present Position"; `scripts/validate_bam_testbench.py`
# ERROR_GAIN uses the same 4096).
ENCODER_STEP = 2.0 * math.pi / 4096.0
# XL330 "Present Velocity" unit: 0.229 rev/min.
VELOCITY_STEP = 0.229 * 2.0 * math.pi / 60.0

# One stream per concern, so their draws never interleave.
_IMU, _ENCODER, _MODEL, _TOF, _LINK, _COMMAND = range(6)


@dataclass(frozen=True)
class Profile:
    name: str
    # Trunk IMU: LSM6DSV16X on the imu_to_dxl board (`duck-control/src/imu.rs`).
    gyro_noise: float = 0.0  # rad/s, std of white noise per sample
    gyro_bias: float = 0.0  # rad/s, std of the per-axis bias at power-up
    gyro_bias_walk: float = 0.0  # rad/s/sqrt(s), random walk of that bias
    gravity_noise: float = 0.0  # std per component of the projected gravity
    imu_tilt_deg: float = (
        0.0  # mounting error: random axis, magnitude uniform in [0, this]
    )
    yaw_drift: float = 0.0  # rad/s, std of the game-rotation-vector heading drift rate
    quat_half: bool = False  # quaternion components shipped as IEEE binary16
    # Joint encoders: XL330 present position / velocity.
    joint_noise: float = 0.0  # rad, std
    joint_bias: float = 0.0  # rad, per-joint offset uniform in [-this, this]
    joint_quant: float = 0.0  # rad, quantisation step
    velocity_noise: float = 0.0  # rad/s, std
    velocity_quant: float = 0.0  # rad/s, quantisation step
    # The bus between robotd and the servos.
    sensor_latency_s: tuple[float, float] = (0.0, 0.0)
    command_latency_s: tuple[float, float] = (0.0, 0.0)
    read_drop: float = 0.0  # probability a read returns the previous (stale) frame
    write_drop: float = 0.0  # probability a target write is lost
    # The body and the floor, drawn once per model.
    friction: tuple[float, float] = (
        1.0,
        1.0,
    )  # sliding-friction scale, floor and duck geoms
    mass: tuple[float, float] = (1.0, 1.0)  # per-body mass and inertia scale
    kp: tuple[float, float] = (1.0, 1.0)  # per-motor position-gain (strength) scale
    # VL53L5CX, on top of `tof.py`'s range noise.
    tof_dropout: float = (
        0.0  # per-zone probability of a failed status (corners twice that)
    )
    tof_max_range: float = 4.0  # m, beyond which a zone reports no target
    tof_dark_range: float = (
        4.0  # m, the same for dark surfaces (luminance < DARK_LUMINANCE)
    )
    tof_glass_ghost: float = (
        0.0  # probability a translucent surface returns a mixed range
    )

    @property
    def active(self) -> bool:
        return self.name != "off"


PROFILES = {
    "off": Profile("off"),
    # Inside the envelope the walking policies were trained in.
    "real": Profile(
        "real",
        # Uniform ±0.03 rad/s actor-obs noise on base_ang_vel in training → std 0.03/√3.
        gyro_noise=0.0173,
        # LSM6DSV16X zero-rate level is ±1 dps typical (DS13510); 0.6 dps std keeps the sum of
        # bias and noise inside the ±0.03 rad/s the policy was trained with.
        gyro_bias=0.010,
        gyro_bias_walk=0.0005,
        # Uniform ±0.01 on projected gravity in training → std 0.01/√3.
        gravity_noise=0.0058,
        # Training: up to 6° random-axis IMU misalignment; the runtime removes the board's
        # systematic ~5° pitch (imu-pitch-offset), so what is left on a robot is small.
        imu_tilt_deg=2.0,
        # SFLP's game rotation vector has no magnetometer: heading drifts. ~1.7°/min assumed.
        yaw_drift=0.0005,
        quat_half=True,
        # Uniform ±0.001 rad joint_pos noise in training → std 0.001/√3; ENCODER_BIAS_RANGE ±0.015.
        joint_noise=0.00058,
        joint_bias=0.015,
        joint_quant=ENCODER_STEP,
        # Uniform ±0.25 rad/s joint_vel noise in training → std 0.25/√3.
        velocity_noise=0.144,
        velocity_quant=VELOCITY_STEP,
        # base_ang_vel / projected_gravity delay_max_lag=1 at 50 Hz in training: 0–20 ms.
        sensor_latency_s=(0.0, 0.020),
        # BAM actuator delay_min_lag=3, delay_max_lag=6 at the 5 ms physics step: 15–30 ms.
        command_latency_s=(0.015, 0.030),
        # Assumed. robotd counts stale reads (`duck-control/src/io.rs`) because the bus has them.
        read_drop=0.01,
        write_drop=0.005,
        # foot_friction DR range (0.7, 1.3) in training.
        friction=(0.7, 1.3),
        # MASS_INERTIA_RANDOMIZATION_RANGE and KP_RANDOMIZATION_RANGE in training.
        mass=(0.95, 1.05),
        kp=(0.85, 1.15),
        # VL53L5CX (DS13754): 4 m is the dark-room figure for a white target; ambient light and
        # grey targets shorten it, low-reflectance ones to around a metre and a half. Assumed rates.
        tof_dropout=0.02,
        tof_max_range=3.5,
        tof_dark_range=1.5,
        tof_glass_ghost=0.3,
    ),
    # Past the envelope, to find where the policies break.
    "hard": Profile(
        "hard",
        gyro_noise=0.035,
        gyro_bias=0.0175,  # the full ±1 dps zero-rate level
        gyro_bias_walk=0.002,
        gravity_noise=0.015,
        imu_tilt_deg=6.0,  # the training maximum, without the runtime's correction
        yaw_drift=0.002,
        quat_half=True,
        joint_noise=0.002,
        joint_bias=0.03,
        joint_quant=ENCODER_STEP,
        velocity_noise=0.3,
        velocity_quant=VELOCITY_STEP,
        # The pre-audit training envelope was up to 3 ticks (60 ms) of IMU lag.
        sensor_latency_s=(0.0, 0.040),
        command_latency_s=(0.015, 0.045),
        read_drop=0.05,
        write_drop=0.02,
        friction=(0.5, 1.3),
        mass=(0.9, 1.1),
        kp=(0.8, 1.2),
        tof_dropout=0.08,
        tof_max_range=3.0,
        tof_dark_range=1.0,
        tof_glass_ghost=0.6,
    ),
}

DARK_LUMINANCE = 0.15
GLASS_ALPHA = 0.6
# VL53L5CX statuses (`tof/src/lib.rs`: 5 and 9 valid, 255 no target, anything else failed).
STATUS_CONSISTENCY_FAILED = 4


def profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"realism must be one of {', '.join(PROFILES)}, not {name!r}"
        ) from None


def stream(seed: int, duck: int, kind: int, *extra: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, duck, kind, *extra]))


def keyed_uniform(
    seed: int, duck: int, name: str, bounds: tuple[float, float]
) -> float:
    """One draw per named part, independent of how many parts the model has or their order.

    `duck` -1 is the world itself (the floor).
    """
    low, high = bounds
    if low == high:
        return low
    owner = (0, 1) if duck < 0 else (duck, 0)
    rng = stream(seed, owner[0], _MODEL, owner[1], zlib.crc32(name.encode()))
    return float(rng.uniform(low, high))


def kp_scale(prof: Profile, seed: int, duck: int, joint: str) -> float:
    return keyed_uniform(seed, duck, "kp:" + joint, prof.kp)


def randomize_model(
    model: mujoco.MjModel, prof: Profile, seed: int, prefixes: list[str]
) -> None:
    """Friction and mass, applied once to a freshly compiled model.

    The friction scale is one per world and is applied to the floor *and* to every duck geom:
    MuJoCo combines a contact's friction as the max of its two geoms', so scaling one side alone
    would do nothing whenever the other is grippier. Mass is per body, keyed by name.
    """
    if not prof.active:
        return
    friction = keyed_uniform(seed, -1, "friction", prof.friction)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor >= 0:
        model.geom_friction[floor, 0] *= friction
    for duck, prefix in enumerate(prefixes):
        trunk = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + "trunk_base"
        )
        if trunk < 0:
            continue
        for geom in range(model.ngeom):
            if model.body_rootid[model.geom_bodyid[geom]] == trunk:
                model.geom_friction[geom, 0] *= friction
        for body in range(model.nbody):
            if model.body_rootid[body] != trunk:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or str(body)
            scale = keyed_uniform(
                seed, duck, "mass:" + name.removeprefix(prefix), prof.mass
            )
            model.body_mass[body] *= scale
            model.body_inertia[body] *= scale


def _yaw_quat(angle: float) -> np.ndarray:
    return np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


class SensorModel:
    """What one duck's IMU and encoders report, given what MuJoCo knows.

    Driven by simulated time, not by reads: noise is drawn when physics records a frame, so a
    second client reading the same port (a scorer, a test) cannot change what the daemon sees.
    """

    def __init__(self, prof: Profile, seed: int, duck: int, joints: tuple[str, ...]):
        self.profile = prof
        self.imu = stream(seed, duck, _IMU)
        self.encoder = stream(seed, duck, _ENCODER)
        self.joint_bias = np.array(
            [
                keyed_uniform(
                    seed, duck, "encoder:" + name, (-prof.joint_bias, prof.joint_bias)
                )
                for name in joints
            ]
        )
        self.gyro_bias = (
            self.imu.normal(0.0, prof.gyro_bias, 3) if prof.gyro_bias else np.zeros(3)
        )
        axis = self.imu.normal(size=3)
        axis /= max(np.linalg.norm(axis), 1e-12)
        angle = (
            math.radians(self.imu.uniform(0.0, prof.imu_tilt_deg))
            if prof.imu_tilt_deg
            else 0.0
        )
        tilt = np.zeros(4)
        mujoco.mju_axisAngle2Quat(tilt, axis, angle)
        self.mount = np.zeros(9)
        mujoco.mju_quat2Mat(self.mount, tilt)
        self.mount = self.mount.reshape(3, 3)
        self.mount_quat = tilt
        self.yaw_rate = self.imu.normal(0.0, prof.yaw_drift) if prof.yaw_drift else 0.0
        self.yaw_error = 0.0
        self.last_time: float | None = None
        self.frames: deque[tuple[float, dict]] = deque(maxlen=64)

    def _quantize(self, values: np.ndarray, step: float) -> np.ndarray:
        return np.round(values / step) * step if step else values

    def record(
        self,
        time: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        quat: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
    ) -> None:
        prof = self.profile
        dt = 0.0 if self.last_time is None else max(0.0, time - self.last_time)
        self.last_time = time
        if prof.gyro_bias_walk and dt:
            self.gyro_bias = self.gyro_bias + self.imu.normal(
                0.0, prof.gyro_bias_walk * math.sqrt(dt), 3
            )
        self.yaw_error += self.yaw_rate * dt

        gyro = self.mount @ gyro + self.gyro_bias
        if prof.gyro_noise:
            gyro = gyro + self.imu.normal(0.0, prof.gyro_noise, 3)
        gravity = self.mount @ gravity
        if prof.gravity_noise:
            gravity = gravity + self.imu.normal(0.0, prof.gravity_noise, 3)
        # The chip's orientation: true, rotated by the heading drift and the mounting error.
        measured = np.zeros(4)
        mujoco.mju_mulQuat(measured, _yaw_quat(self.yaw_error), quat)
        mounted = np.zeros(4)
        mujoco.mju_mulQuat(mounted, measured, self.mount_quat)
        if prof.quat_half:
            # SFLP ships x/y/z as binary16 and the reader reconstructs w (`imu.rs`).
            xyz = mounted[1:].astype(np.float16).astype(np.float64) * (
                1 if mounted[0] >= 0 else -1
            )
            w = math.sqrt(max(0.0, 1.0 - float(xyz @ xyz)))
            mounted = np.array([w, *xyz])

        positions = positions + self.joint_bias
        if prof.joint_noise:
            positions = positions + self.encoder.normal(
                0.0, prof.joint_noise, positions.shape
            )
        positions = self._quantize(positions, prof.joint_quant)
        if prof.velocity_noise:
            velocities = velocities + self.encoder.normal(
                0.0, prof.velocity_noise, velocities.shape
            )
        velocities = self._quantize(velocities, prof.velocity_quant)

        self.frames.append(
            (
                time,
                {
                    "positions": positions,
                    "velocities": velocities,
                    "gyro": gyro,
                    "gravity": gravity,
                    "quat": mounted,
                },
            )
        )

    def frame_at(self, time: float) -> dict | None:
        """The newest frame recorded no later than `time` — what a late read actually returns."""
        chosen = None
        for stamp, frame in self.frames:
            if stamp <= time + 1e-9:
                chosen = frame
            else:
                break
        if chosen is None and self.frames:
            chosen = self.frames[0][1]
        return chosen


class Link:
    """One connection's bus faults: latency and dropped packets, seeded per connection."""

    def __init__(self, prof: Profile, seed: int, duck: int, connection: int):
        self.profile = prof
        self.random = stream(seed, duck, _LINK, connection)
        self.last_frame: dict | None = None

    def sensor_latency(self) -> float:
        low, high = self.profile.sensor_latency_s
        return float(self.random.uniform(low, high)) if high > 0 else 0.0

    def command_latency(self) -> float:
        low, high = self.profile.command_latency_s
        return float(self.random.uniform(low, high)) if high > 0 else 0.0

    def drop_read(self) -> bool:
        return (
            bool(self.profile.read_drop)
            and self.random.random() < self.profile.read_drop
        )

    def drop_write(self) -> bool:
        return (
            bool(self.profile.write_drop)
            and self.random.random() < self.profile.write_drop
        )


class CommandQueue:
    """Targets in flight to the servos. Applied in order; a later write never overtakes."""

    def __init__(self):
        self.pending: deque[tuple[float, list[float], float]] = deque()
        self.last_due = -math.inf

    def push(self, due: float, targets: list[float], mouth: float) -> None:
        due = max(due, self.last_due)
        self.last_due = due
        self.pending.append((due, targets, mouth))

    def due(self, time: float) -> tuple[list[float], float] | None:
        latest = None
        while self.pending and self.pending[0][0] <= time + 1e-9:
            _, targets, mouth = self.pending.popleft()
            latest = (targets, mouth)
        return latest

    def clear(self) -> None:
        self.pending.clear()


class TofModel:
    """VL53L5CX failure modes on top of `tof.py`'s range noise."""

    def __init__(self, prof: Profile, seed: int, duck: int):
        self.profile = prof
        self.random = stream(seed, duck, _TOF)

    def surface(self, model: mujoco.MjModel, geom: int) -> tuple[float, float]:
        """(luminance, alpha) of what a ray hit; a material, if present, tints the geom colour."""
        rgba = model.geom_rgba[geom].copy()
        material = model.geom_matid[geom]
        if material >= 0:
            rgba = rgba * model.mat_rgba[material]
        luminance = float(0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2])
        return luminance, float(rgba[3])

    def dropout(self, zone: int, rows: int, cols: int) -> bool:
        if not self.profile.tof_dropout:
            return False
        row, col = divmod(zone, cols)
        corner = row in (0, rows - 1) and col in (0, cols - 1)
        rate = self.profile.tof_dropout * (2.0 if corner else 1.0)
        return self.random.random() < rate
