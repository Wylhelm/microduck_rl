"""A microduck body in MuJoCo, served to the real `robotd` over TCP.

    uv run duck-body                      # one duck, a viewer
    uv run duck-body --ducks 4            # four, sharing one world and one window
    uv run duck-body --port 7801
    uv run duck-body --headless           # no window, for tests and for many ducks

Then, on the daemon side:

    robotd --sim 127.0.0.1:7801

One duck per TCP port, from `--port` upwards, and one `mj_step` for all of them — so ducks share
a floor and can bump into each other, which is the difference between a room with four robots in
it and four robots on the same screen.

Everything above `duck_control::io::RobotIo` is the code that runs on a real robot — the 50 Hz loop,
the ONNX policies, safety, fall detection, odometry, kinematics, every IPC call. This process is the
only part that knows there is no robot.

**Why this repo.** It already owns the scenes, the BAM actuator models fitted to the real XL330s and
mjlab; serving a body to a daemon is the mirror of the sim2real it does today. The daemon-side half
lives in `microduck` because it implements an in-repo trait against an in-repo protocol.

## The protocol

Newline-delimited JSON over TCP, one request and one answer per line, `protocol` checked in the
handshake — the two halves live in two repositories, so "your simulator is old" and "your daemon is
old" must not be the same symptom. `duck_control::sim` is the other end and carries the reasoning
for TCP-not-a-unix-socket and JSON-not-a-packed-struct.

## Two mappings this side owns, on purpose

**Fifteen joints out here, fourteen in the model.** The daemon indexes joints as `JOINT_NAMES`,
which includes `mouth` at index 9; no alpha policy drives it and the walking model does not have it.
This inserts a zero sensor value and uses the target only for a separate, visual-only beak.

**Gravity, not just orientation.** The policy observes projected gravity in the trunk frame. MuJoCo
gives an orientation quaternion, so this does the rotation — the same arithmetic the IMU's SFLP
filter does on the robot, on the other side of the same wire.

## Making the twin no better than the robot (all opt-in)

- `--realism off|real|hard` (`sim/realism.py`): sensor noise, IMU bias and drift, encoder
  quantisation, bus latency and dropped packets, floor friction, mass and motor strength, ToF
  failures. `off` is the default and is exactly the old behaviour.
- `--no-assist`: no simulation-only hand. Nothing holds a duck before it is enabled, nothing steadies
  the trunk through an equipment change or the home ramp, and a `mode` request does not swap feet
  for skates — on the robot a human does that, here `equipment.fit` does.
- `--obstacles FILE` and `obstacle.*` (`sim/obstacles.py`): chairs, bags, closed doors, a pet.
- `oracle.pose`: the true pose, for scoring only. The `trunk`/`trunk_z` fields of `read` are the
  same oracle and are kept for old tools; a controller must use neither.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import socketserver
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.sim import realism
from mjlab_microduck.sim.camera import FPS as CAMERA_FPS
from mjlab_microduck.sim.camera import Camera, FrameHandler, FrameServer
from mjlab_microduck.sim.obstacles import DEFAULT_SLOTS, Obstacles, add_slots
from mjlab_microduck.sim.obstacles import load as load_obstacles
from mjlab_microduck.sim.tof import COLS, ROWS, Tof
from mjlab_microduck.sim.visual import MOUTH_CLOSED, EquipmentVisualWorld, VisualWorld

PROTOCOL = 2

# What the policies were trained at, and what `scripts/infer_policy.py` sets. The scenes ship 0.002;
# with that script's decimation of 4 this is exactly the 50 Hz the daemon's control loop runs at.
# Not a performance knob: the BAM actuator fit, the contact solref and the joint armature are all
# tuned at this step, so 0.002 gives a duck whose legs reach the right angles and still cannot hold
# itself up.
TIMESTEP = 0.005

# Where `scripts/infer_policy.py` puts a duck before it starts: trunk this high, upright, every joint
# at the home pose. Not a keyframe — the keyframes are poses and this is a *placement*.
HOME_TRUNK_Z = 0.125

# `duck_ipc_proto::JOINT_NAMES`, which is protocol: every positional array on the wire is indexed by
# it. Duplicated here rather than shared, because the two repositories cannot share a constant — and
# checked against the model at startup, which is the next best thing.
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
MOUTH_INDEX = JOINT_NAMES.index("mouth")

# `duck_control::DEFAULT_POSITION`, and `DEFAULT_POSE` in `infer_policy.py` with the mouth put back.
# The right leg is mirrored, not symmetric — worth reading rather than assuming.
HOME_POSE = (
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
)

SCENES = Path(__file__).resolve().parents[1] / "robot" / "microduck"
# `scene.xml`, not `scene_walk.xml`: the walking scene includes the model the RL work trains
# against, whose actuator default classes carry `contype="0" conaffinity="0"`, so the robot collides
# with nothing and sinks through a floor the scene really does contain.
DEFAULT_SCENE = SCENES / "scene.xml"
# Far enough apart not to touch at rest, close enough to be in one screenful.
SPACING = 0.5

# What the robot reports and nothing here simulates. Constants rather than omissions, so
# `robotctl health` shows a plausible robot instead of an alarming one.
NOMINAL_VOLTS = 7.4
NOMINAL_TEMP_C = 32.0
MODE_SUPPORT_SETTLE_S = 0.3


def duck_prefix(index: int) -> str:
    """`""` for the first duck — it is the scene's own — and `d1_`, `d2_` … for attached ones."""
    return "" if index == 0 else f"d{index}_"


def robot_file(scene: Path, mode: str | None = None) -> Path:
    root = ET.parse(scene).getroot()
    include = next(
        (node for node in root.findall("include")
         if Path(node.get("file", "")).name.startswith("robot_")),
        None,
    )
    if include is None:
        raise ValueError(f"{scene.name} has no replaceable robot include")
    original = scene.parent / include.attrib["file"]
    if mode is None:
        return original
    backlash = "_backlash" if "backlash" in original.stem else ""
    if mode == "roller":
        return SCENES / f"robot_groundcontact_rollers{backlash}.xml"
    if mode == "walk":
        return (
            SCENES / f"robot_groundcontact{backlash}.xml"
            if "rollers" in original.stem else original
        )
    raise ValueError("mode must be walk or roller")


def build_world(
    scene: Path, count: int, modes: list[str] | None = None, obstacle_slots: int = 0
) -> mujoco.MjModel:
    """One model holding `count` ducks, so they share a floor and can bump into each other.

    The scene already contains one duck; the rest are attached to it under a name prefix, which is
    what `MjSpec` is for. Sharing a world rather than running N simulators is the whole point: two
    ducks in separate physics can be beside each other on a screen and never touch, and "beside each
    other" is what every social behaviour is about.

    `obstacle_slots` parked mocap bodies are compiled in for `sim/obstacles.py`; none by default,
    so a world without obstacles is the model it always was.
    """
    if count == 1 and modes is None and not obstacle_slots:
        return mujoco.MjModel.from_xml_path(str(scene))

    if modes is None:
        spec = mujoco.MjSpec.from_file(str(scene))
    else:
        root = ET.parse(scene).getroot()
        original = robot_file(scene)
        for node in root.findall("include"):
            path = scene.parent / node.attrib["file"]
            node.set("file", str(robot_file(scene, modes[0]) if path == original else path))
        # Equipment changes nq; a keyframe indexed for the old body cannot be carried across.
        for keys in root.findall("keyframe"):
            root.remove(keys)
        ET.SubElement(root, "compiler", meshdir=str(SCENES / "assets"))
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    for index in range(1, count):
        # A fresh child every time: `attach` renames the spec it is given, so reusing one gives the
        # third duck names like `d2_d1_left_hip_yaw` and a compile error about incompatible ids.
        robot = mujoco.MjSpec.from_file(str(
            robot_file(scene, modes[index]) if modes is not None else robot_file(scene)
        ))
        frame = spec.worldbody.add_frame(pos=[0.0, index * SPACING, 0.0])
        spec.attach(robot, prefix=duck_prefix(index), frame=frame)
    add_slots(spec, obstacle_slots)
    return spec.compile()


def pose_table(
    scene: Path, keyframe: str, *, model: mujoco.MjModel | None = None
) -> tuple[dict[str, float] | None, float]:
    """A named pose from the scene's keyframes, as joint name → angle.

    Read from a single-duck model and applied by name, because attaching a robot does not bring the
    scene's keyframes with it — and a pose is a fact about the robot, not about how many of them are
    in the room.
    """
    if keyframe.upper() == "HOME":
        return None, HOME_TRUNK_Z
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(scene))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
    if keyframe not in names:
        raise SystemExit(
            f"no keyframe {keyframe!r} in {scene.name}. It has: {', '.join(n for n in names if n)}"
        )
    qpos = model.key_qpos[names.index(keyframe)]
    table = {}
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if name in JOINT_NAMES:
            table[name] = float(qpos[model.jnt_qposadr[joint]])
    # The keyframe's own trunk height, not the home one: a seated duck placed at standing height is
    # a duck hovering above the floor, which drops the moment anybody enables it.
    return table, float(qpos[2])


def gravity_in_trunk(quat: np.ndarray) -> np.ndarray:
    """World gravity in the trunk frame. Upright is `[0, 0, -1]`.

    What the policy actually observes, and it is here rather than in the daemon because the daemon's
    IMU delivers exactly this, already rotated, from the sensor's own filter.
    """
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quat)
    return -rotation.reshape(3, 3).T[:, 2]


def floor_clearance(model: mujoco.MjModel, data: mujoco.MjData, prefix: str) -> float:
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor < 0 or model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError("equipment switching requires a floor plane named 'floor'")
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "trunk_base")
    distances = [
        mujoco.mj_geomDistance(model, data, geom, floor, 10.0, None)
        for geom in range(model.ngeom)
        if model.body_rootid[model.geom_bodyid[geom]] == trunk
        and (model.geom_contype[geom] or model.geom_conaffinity[geom])
    ]
    if not distances:
        raise ValueError("equipment switching requires robot collision geoms")
    return min(distances)


class World:
    """The physics, shared by every duck in it.

    One `mj_step` advances all of them, which is what makes a shove real rather than decorative. The
    lock is held only to copy numbers in or out, never across a step: a daemon asking for sensors
    must not wait on the solver, for the same reason a real bus read does not wait on a servo.
    """

    def __init__(
        self, scene: Path, count: int = 1, *, preload_equipment: bool = False,
        realism_profile: str = "off", realism_seed: int = 0, no_assist: bool = False,
        obstacle_slots: int = 0,
    ):
        if preload_equipment and count != 1:
            raise ValueError("preloading equipment requires exactly one duck")
        if obstacle_slots < 0:
            raise ValueError("obstacle slots cannot be negative")
        if int(realism_seed) < 0:
            raise ValueError("the realism seed cannot be negative")
        self.scene = scene.resolve()
        self.count = count
        self.profile = realism.profile(realism_profile)
        self.seed = int(realism_seed)
        self.no_assist = bool(no_assist)
        self.obstacles = Obstacles(obstacle_slots)
        self._connections = 0
        initial_mode = "roller" if "rollers" in robot_file(self.scene).stem else "walk"
        self.modes = [initial_mode] * count
        self.model = self._prepare(build_world(scene, count, obstacle_slots=obstacle_slots))
        self.preloaded_models: dict[str, tuple[mujoco.MjModel, np.ndarray]] = {}
        if preload_equipment:
            other_mode = "walk" if initial_mode == "roller" else "roller"
            other_model = self._prepare(
                build_world(self.scene, count, [other_mode], obstacle_slots=obstacle_slots)
            )
            for mode, model in ((initial_mode, self.model), (other_mode, other_model)):
                self.preloaded_models[mode] = (model, model.actuator_gainprm[:, 0].copy())
        self.data = mujoco.MjData(self.model)
        self.obstacles.sync(self.model, self.data)
        self.lock = threading.Lock()
        self.equipment_lock = threading.Lock()
        self.bodies: list[Body] = []

    def _prepare(self, model: mujoco.MjModel) -> mujoco.MjModel:
        """Every freshly compiled model, once: the timestep, and this world's random body."""
        model.opt.timestep = TIMESTEP
        realism.randomize_model(
            model, self.profile, self.seed, [duck_prefix(i) for i in range(self.count)]
        )
        return model

    def next_connection(self) -> int:
        with self.lock:
            self._connections += 1
            return self._connections

    # ── obstacles ─────────────────────────────────────────────────────────

    def _require_slots(self) -> None:
        if not self.obstacles.capacity:
            raise ValueError(
                "this world has no obstacle slots: start it with --obstacles FILE or "
                "--obstacle-slots N"
            )

    def _place_obstacles(self, geometry: bool = True) -> None:
        with self.lock:
            self.obstacles.sync(self.model, self.data, geometry=geometry)
            # Positions now, not at the next step: a ToF frame or a render in between must see it.
            mujoco.mj_kinematics(self.model, self.data)

    def add_obstacle(self, entry: dict) -> dict:
        self._require_slots()
        obstacle = self.obstacles.add(entry)
        self._place_obstacles()
        return obstacle.describe()

    def remove_obstacle(self, name: str) -> None:
        self._require_slots()
        self.obstacles.remove(name)
        self._place_obstacles()

    def move_obstacle(self, name: str, pose: dict) -> dict:
        self._require_slots()
        obstacle = self.obstacles.move(name, pose)
        self._place_obstacles(geometry=False)
        return obstacle.describe()

    def set_mode(self, index: int, mode: str) -> None:
        if mode not in ("walk", "roller"):
            raise ValueError("mode must be walk or roller")
        # Compile away from the physics lock: sensor reads can finish on the current model.
        with self.equipment_lock:
            if self.modes[index] == mode:
                return
            modes = self.modes.copy()
            modes[index] = mode
            if self.preloaded_models:
                model, nominal_gains = self.preloaded_models[mode]
                # Models retain the last gain applied to them; rebind against their original gains.
                model.actuator_gainprm[:, 0] = nominal_gains
            else:
                model = self._prepare(
                    build_world(self.scene, self.count, modes, self.obstacles.capacity)
                )
            data = mujoco.MjData(model)
            with self.lock:
                previous, state = self.model, self.data
                self.obstacles.sync(model, data)
                clearance = self.bodies[index].floor_clearance()
                for joint in range(model.njnt):
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                    old = mujoco.mj_name2id(previous, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if old < 0:
                        continue
                    kind = model.jnt_type[joint]
                    nq = 7 if kind == mujoco.mjtJoint.mjJNT_FREE else (
                        4 if kind == mujoco.mjtJoint.mjJNT_BALL else 1
                    )
                    nv = 6 if kind == mujoco.mjtJoint.mjJNT_FREE else (
                        3 if kind == mujoco.mjtJoint.mjJNT_BALL else 1
                    )
                    new_q, old_q = model.jnt_qposadr[joint], previous.jnt_qposadr[old]
                    new_v, old_v = model.jnt_dofadr[joint], previous.jnt_dofadr[old]
                    data.qpos[new_q:new_q + nq] = state.qpos[old_q:old_q + nq]
                    data.qvel[new_v:new_v + nv] = state.qvel[old_v:old_v + nv]
                for actuator in range(model.nu):
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
                    old = mujoco.mj_name2id(previous, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                    if old >= 0:
                        data.ctrl[actuator] = state.ctrl[old]
                data.time = state.time
                mujoco.mj_forward(model, data)
                body = self.bodies[index]
                trunk_joint = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, body.prefix + "trunk_base_freejoint"
                )
                trunk_qpos = model.jnt_qposadr[trunk_joint]
                data.qpos[trunk_qpos + 2] += (
                    clearance - floor_clearance(model, data, body.prefix)
                )
                mujoco.mj_forward(model, data)
                self.model, self.data = model, data
                for body in self.bodies:
                    body.bind_model()
                    body._apply_torque()
                # Putting wheels on must not bury them in the floor; taking them off must not
                # drop the duck. Preserve clearance, planar position, orientation and momentum.
                for body in self.bodies:
                    if not body.released:
                        body.remember()
                    if body.support_pose is not None:
                        body.support_pose = data.qpos[body.trunk:body.trunk + 7].copy()
                self.modes = modes
            print(f"== duck {index}: equipment switched to {mode}", flush=True)

    def step(self, times: int = 1) -> None:
        """Advance the world, taking the lock once for the whole batch.

        **Batched because the lock is the bottleneck, not the solver.** Four daemons at 50 Hz make
        400 requests a second, each of which wants this lock, and Python hands the GIL around
        between every one of them — so a physics loop taking and releasing it 200 times a second
        loses. Four steps at 5 ms is 20 ms of world, which is exactly one control tick, so nothing
        sees a sensor older than the tick it belongs to.
        """
        with self.lock:
            moving = self.obstacles.moving()
            for _ in range(times):
                for body in self.bodies:
                    if body.commands.pending:
                        body.apply_due_commands(self.data.time)
                if moving:
                    self.obstacles.advance(self.data.time)
                    self.obstacles.sync(self.model, self.data, geometry=False)
                mujoco.mj_step(self.model, self.data)
                # A duck nobody has enabled yet is put back where it was. Physics is shared, so
                # it cannot simply not be stepped — and a hand steadying one robot while another
                # walks about is an ordinary thing for a room to contain. Not with --no-assist:
                # nobody holds a real robot that has not been enabled.
                for body in self.bodies:
                    if not body.released:
                        if not self.no_assist:
                            body.restore()
                    elif body.support_pose is not None:
                        if body.support_until is None or self.data.time < body.support_until:
                            self.data.qpos[body.trunk:body.trunk + 7] = body.support_pose
                            self.data.qvel[body.trunk_dof:body.trunk_dof + 6] = 0.0
                        else:
                            body.support_pose = None
                            body.support_until = None
                    if body.sensor_model is not None:
                        body.record()


class Body:
    """One duck's view of the world: its joints, its actuators, its trunk."""

    def __init__(self, world: World, index: int, limp: bool = False, kp: float = 200.0):
        self.world = world
        self.index = index
        self.prefix = duck_prefix(index)
        self.camera: Camera | None = None
        self.bind_model()
        self.released = limp
        self.torque_on = not limp
        self.kp = kp
        self.held = None
        self.support_pose: np.ndarray | None = None
        self.support_until: float | None = None
        self.visual_mouth = MOUTH_CLOSED
        # What the daemon last asked for, which with --no-assist need not be what is fitted.
        self.commanded_mode = world.modes[index]
        prof, seed = world.profile, world.seed
        self.commands = realism.CommandQueue()
        self.sensor_model = (
            realism.SensorModel(prof, seed, index, JOINT_NAMES) if prof.active else None
        )
        self.tof_realism = realism.TofModel(prof, seed, index) if prof.active else None
        # For callers that are not a connection (tests, tools): connection 0.
        self.link = realism.Link(prof, seed, index, 0) if prof.active else None

    def new_link(self) -> realism.Link | None:
        if not self.world.profile.active:
            return None
        return realism.Link(self.world.profile, self.world.seed, self.index,
                            self.world.next_connection())

    def bind_model(self) -> None:
        model = self.world.model

        def ident(kind, name):
            found = mujoco.mj_name2id(model, kind, self.prefix + name)
            if found < 0:
                raise ValueError(f"the model has no {name!r} for duck {self.index}")
            return found

        # By name, never by index: an MJCF edit reorders silently.
        self.actuators = []
        self.to_wire = []
        for wire_index, name in enumerate(JOINT_NAMES):
            found = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.prefix + name)
            if found >= 0:
                self.actuators.append(found)
                self.to_wire.append(wire_index)
        if not self.actuators:
            raise ValueError(f"no actuated joints for duck {self.index} (prefix {self.prefix!r})")

        self.qpos_adr = np.array(
            [model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in self.actuators]
        )
        self.qvel_adr = np.array(
            [model.jnt_dofadr[model.actuator_trnid[a, 0]] for a in self.actuators]
        )
        # The depth sensor, on the model's own `tof` site — so a head that turns takes it along,
        # which is what makes `robot.look` a way to scan a room.
        self.tof = Tof(model, ident(mujoco.mjtObj.mjOBJ_SITE, "tof"), seed=self.index)
        # Built only when this duck is one of `--cameras`: a renderer costs 12 ms a frame, which is
        # forty times what stepping four ducks' physics costs.
        self.trunk = int(model.jnt_qposadr[ident(mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")])
        self.trunk_dof = int(model.jnt_dofadr[ident(mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")])

        self.actuator_slice = np.array(self.actuators)
        self._gain = model.actuator_gainprm[self.actuator_slice, 0].copy()
        if self.world.profile.active:
            # A batch of XL330s is not one motor: each gets its own strength, kept by name across
            # equipment changes.
            self._gain *= [
                realism.kp_scale(self.world.profile, self.world.seed, self.index,
                                 JOINT_NAMES[wire])
                for wire in self.to_wire
            ]

        # **Held until the daemon takes it.** A biped at a static pose is not stable: holding the
        # home pose with position control alone puts this duck on the ground in under a second, at
        # any timestep, from any placement — `infer_policy.py` never does it, because it has the
        # policy balancing from step zero. `robotd` deliberately does not enable torque when it
        # starts, so the seconds before it would be spent falling over.
    def floor_clearance(self) -> float:
        return floor_clearance(self.world.model, self.world.data, self.prefix)

    def set_mode_support(self, on: bool) -> None:
        """A simulation-only hand steadies the trunk, leaving the leg motors free.

        Never with --no-assist: a robot changing policy, or ramping home, is on its own.
        """
        if self.world.no_assist:
            return
        with self.world.lock:
            if on and self.released and self.torque_on:
                if self.support_pose is None:
                    model, data = self.world.model, self.world.data
                    placement = mujoco.MjData(model)
                    placement.qpos[:] = data.qpos
                    w, x, y, z = data.qpos[self.trunk + 3:self.trunk + 7]
                    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                    placement.qpos[self.trunk + 3:self.trunk + 7] = [
                        math.cos(yaw / 2), 0, 0, math.sin(yaw / 2),
                    ]
                    placement.qpos[self.qpos_adr] = [
                        HOME_POSE[index] for index in self.to_wire
                    ]
                    mujoco.mj_forward(model, placement)
                    placement.qpos[self.trunk + 2] += (
                        0.001 - floor_clearance(model, placement, self.prefix)
                    )
                    self.support_pose = placement.qpos[self.trunk:self.trunk + 7].copy()
                self.support_until = None
            elif self.support_pose is not None and self.support_until is None:
                # The new policy gets a few ticks at home before the hand lets go.
                self.support_until = self.world.data.time + MODE_SUPPORT_SETTLE_S

    # ── placement ─────────────────────────────────────────────────────────

    def place(self, pose: dict[str, float] | None, trunk_z: float, offset_y: float) -> None:
        data = self.world.data
        data.qpos[self.trunk + 0] = 0.0
        data.qpos[self.trunk + 1] = offset_y
        data.qpos[self.trunk + 2] = trunk_z
        data.qpos[self.trunk + 3 : self.trunk + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[self.trunk_dof : self.trunk_dof + 6] = 0.0
        for slot, wire_index in enumerate(self.to_wire):
            name = JOINT_NAMES[wire_index]
            value = HOME_POSE[wire_index] if pose is None else pose.get(name, HOME_POSE[wire_index])
            data.qpos[self.qpos_adr[slot]] = value
            data.qvel[self.qvel_adr[slot]] = 0.0
        data.ctrl[self.actuator_slice] = data.qpos[self.qpos_adr]
        self._apply_torque()
        self.remember()

    def remember(self) -> None:
        data = self.world.data
        self.held = (
            data.qpos[self.trunk : self.trunk + 7].copy(),
            data.qpos[self.qpos_adr].copy(),
        )

    def restore(self) -> None:
        """Put this duck back where it was, for the one that has not been enabled yet."""
        if self.held is None:
            return
        trunk, joints = self.held
        data = self.world.data
        data.qpos[self.trunk : self.trunk + 7] = trunk
        data.qpos[self.qpos_adr] = joints
        data.qvel[self.trunk_dof : self.trunk_dof + 6] = 0.0
        data.qvel[self.qvel_adr] = 0.0

    # ── what the daemon sees ──────────────────────────────────────────────

    def record(self) -> None:
        """Hand this instant to the sensor model. Called by `World.step`, under the world lock."""
        data = self.world.data
        positions = np.zeros(len(JOINT_NAMES))
        velocities = np.zeros(len(JOINT_NAMES))
        positions[self.to_wire] = data.qpos[self.qpos_adr]
        velocities[self.to_wire] = data.qvel[self.qvel_adr]
        quat = data.qpos[self.trunk + 3 : self.trunk + 7].copy()
        self.sensor_model.record(
            float(data.time), positions, velocities, quat,
            data.qvel[self.trunk_dof + 3 : self.trunk_dof + 6].copy(), gravity_in_trunk(quat),
        )

    def sensors(self, link: realism.Link | None = None) -> dict:
        link = link if link is not None else self.link
        if link is not None and link.last_frame is not None and link.drop_read():
            # A read the bus lost: the daemon gets what it had, as a stale IMU block on the robot.
            return link.last_frame
        with self.world.lock:
            data = self.world.data
            sim_time = float(data.time)
            positions = data.qpos[self.qpos_adr].copy()
            velocities = data.qvel[self.qvel_adr].copy()
            force = data.actuator_force[self.actuator_slice].copy()
            quat = data.qpos[self.trunk + 3 : self.trunk + 7].copy()
            gyro = data.qvel[self.trunk_dof + 3 : self.trunk_dof + 6].copy()
            trunk = [float(v) for v in data.qpos[self.trunk : self.trunk + 3]]
            trunk_z = trunk[2]
            mode = self.world.modes[self.index]
            supported = self.support_pose is not None
            measured = None
            if self.sensor_model is not None:
                latency = link.sensor_latency() if link is not None else 0.0
                measured = self.sensor_model.frame_at(sim_time - latency)
                if measured is None:
                    self.record()
                    measured = self.sensor_model.frame_at(sim_time)

        if measured is not None:
            present = np.zeros(len(JOINT_NAMES), dtype=bool)
            present[self.to_wire] = True
            wire_pos = np.where(present, measured["positions"], 0.0).tolist()
            wire_vel = np.where(present, measured["velocities"], 0.0).tolist()
            imu = {
                "gyro": [float(v) for v in measured["gyro"]],
                "gravity": [float(v) for v in measured["gravity"]],
                "quat": [float(v) for v in measured["quat"]],
            }
        else:
            wire_pos = [0.0] * len(JOINT_NAMES)
            wire_vel = [0.0] * len(JOINT_NAMES)
            for slot, wire_index in enumerate(self.to_wire):
                wire_pos[wire_index] = float(positions[slot])
                wire_vel[wire_index] = float(velocities[slot])
            imu = {
                "gyro": [float(v) for v in gyro],
                "gravity": [float(v) for v in gravity_in_trunk(quat)],
                "quat": [float(v) for v in quat],
            }
        wire_cur = [0.0] * len(JOINT_NAMES)
        for slot, wire_index in enumerate(self.to_wire):
            # Not calibrated against a real servo: a stand-in with the right shape, so a consumer
            # watching load sees load. Amps from a simulated torque would be a fiction with a unit.
            wire_cur[wire_index] = abs(float(force[slot])) * 100.0

        answer = {
            "positions": wire_pos,
            "velocities": wire_vel,
            "currents_ma": wire_cur,
            # **Not part of the protocol, and deliberately extra.** No robot can measure how high its
            # own trunk is, and serde ignores these on the daemon side. They are here because a tool
            # asking "did it stand up?" has no other way to know — a duck sitting on its bottom with
            # a vertical trunk has gravity [0, 0, -1] too — and because a simulator whose seconds are
            # not seconds ruins a policy silently.
            # DEPRECATED for anything but scoring and old tools: `trunk` and `trunk_z` are the true
            # pose — an oracle. Use `oracle.pose`, which says so; a controller must use neither.
            "trunk_z": trunk_z,
            # Where this duck is in the room. No robot knows that either — it is here so a simulated
            # radio can decide who is close enough to hear whom, which is the one thing a real BLE
            # advertisement gets for free and a faked one has to be told.
            "trunk": trunk,
            "sim_time": sim_time,
            "mode": mode,
            "supported": supported,
            "imu": imu,
        }
        if self.world.no_assist:
            answer["commanded_mode"] = self.commanded_mode
        if link is not None:
            link.last_frame = answer
        return answer

    def oracle_pose(self) -> dict:
        """The true pose. For scoring a run — never for steering one: no robot can know it."""
        with self.world.lock:
            data = self.world.data
            x, y, z = (float(v) for v in data.qpos[self.trunk : self.trunk + 3])
            w, qx, qy, qz = (float(v) for v in data.qpos[self.trunk + 3 : self.trunk + 7])
            sim_time = float(data.time)
            mode = self.world.modes[self.index]
        return {
            "oracle": True,
            "x": x, "y": y, "z": z,
            "yaw": math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)),
            "quat": [w, qx, qy, qz],
            "sim_time": sim_time,
            "mode": mode,
        }

    def slow_sensors(self) -> dict:
        return {"volts": NOMINAL_VOLTS, "temps_c": [NOMINAL_TEMP_C] * len(JOINT_NAMES)}

    def depth(self) -> dict:
        """One 8x8 depth frame, in the units `tofd` publishes.

        Sixty-four ray casts, so this is the most expensive thing here — asked for at the sensor's
        own 15 Hz rather than the control loop's 50, exactly as the hardware is.
        """
        with self.world.lock:
            distance_mm, status = self.tof.frame(self.world.data, self.tof_realism)
        return {"rows": ROWS, "cols": COLS, "distance_mm": distance_mm, "status": status}

    # ── what the daemon commands ──────────────────────────────────────────

    def set_targets(self, wire_targets: list[float], link: realism.Link | None = None) -> None:
        if len(wire_targets) != len(JOINT_NAMES):
            raise ValueError(f"expected {len(JOINT_NAMES)} targets, got {len(wire_targets)}")
        mouth = float(wire_targets[MOUTH_INDEX])
        if not math.isfinite(mouth):
            raise ValueError("mouth target must be finite")
        link = link if link is not None else self.link
        if link is not None:
            if link.drop_write():
                return
            latency = link.command_latency()
            with self.world.lock:
                # In flight on the bus and in the servo's own loop; lands a few steps later.
                self.commands.push(
                    self.world.data.time + latency, [float(v) for v in wire_targets], mouth
                )
            return
        with self.world.lock:
            for slot, wire_index in enumerate(self.to_wire):
                self.world.data.ctrl[self.actuators[slot]] = wire_targets[wire_index]
            self.visual_mouth = mouth

    def apply_due_commands(self, time: float) -> None:
        """Targets whose latency has elapsed reach the motors. Under the world lock."""
        due = self.commands.due(time)
        if due is None:
            return
        targets, mouth = due
        for slot, wire_index in enumerate(self.to_wire):
            self.world.data.ctrl[self.actuators[slot]] = targets[wire_index]
        self.visual_mouth = mouth

    def close_visual_mouth(self) -> None:
        with self.world.lock:
            self.visual_mouth = MOUTH_CLOSED

    def set_gain(self, kp: int) -> None:
        with self.world.lock:
            self.kp = float(kp)
            self._apply_torque()

    def set_torque(self, on: bool) -> None:
        with self.world.lock:
            self.torque_on = bool(on)
            # Targets still in flight were for the robot as it was; they must not land after this.
            self.commands.clear()
            if not on:
                self.visual_mouth = MOUTH_CLOSED
                self.support_pose = None
                self.support_until = None
            if on:
                self.released = True
                # Torque arriving must not fling the robot at a stale target.
                self.world.data.ctrl[self.actuator_slice] = self.world.data.qpos[self.qpos_adr]
            self._apply_torque()

    def _apply_torque(self) -> None:
        """Torque off means limp, not frozen.

        Refusing to command a fallen robot only freezes it in the pose it fell in — which is why
        `RobotIo::set_gain` exists at all. Zero gain is the simulated equivalent of cutting power.
        The daemon's kp is a Dynamixel register value whose 200 is what BAM fitted the model's own
        gain to, so it is a ratio against 200 rather than a number in the same units.
        """
        scale = (self.kp / 200.0) if self.torque_on else 0.0
        model = self.world.model
        model.actuator_gainprm[self.actuator_slice, 0] = self._gain * scale
        model.actuator_biasprm[self.actuator_slice, 1] = -self._gain * scale


class Handler(socketserver.StreamRequestHandler):
    """One duck's daemon. One connection at a time, which is the real relationship too."""

    def handle(self) -> None:
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        body: Body = self.server.body
        self.link = body.new_link()
        print(f"== duck {body.index}: daemon connected from {self.client_address}", flush=True)
        try:
            for raw in self.rfile:
                try:
                    answer = self.dispatch(body, json.loads(raw))
                except Exception as error:  # a bad frame must not take the simulator down with it
                    answer = {"error": str(error)}
                self.wfile.write((json.dumps(answer) + "\n").encode())
                self.wfile.flush()
        finally:
            if getattr(self, "mode_support", False):
                body.set_mode_support(False)
            body.close_visual_mouth()
            print(f"== duck {body.index}: daemon disconnected", flush=True)

    def dispatch(self, body: Body, request: dict) -> dict:
        op = request.get("op")
        if op == "hello":
            asked = request.get("protocol")
            if asked != PROTOCOL:
                raise ValueError(
                    f"the daemon speaks protocol {asked} and this simulator speaks {PROTOCOL}"
                )
            return {"protocol": PROTOCOL}
        if op == "read":
            return body.sensors(getattr(self, "link", None))
        if op == "write":
            body.set_targets(request["targets"], getattr(self, "link", None))
            return {}
        if op == "gain":
            body.set_gain(int(request["kp"]))
            return {}
        if op == "torque":
            body.set_torque(bool(request["on"]))
            return {}
        if op == "mode":
            support = request.get("support", False)
            if not isinstance(support, bool):
                raise ValueError("support must be a boolean")
            mode = request["mode"]
            if body.world.no_assist:
                # What `RobotIo::set_drive_mode` does on the robot: nothing to the body. The daemon
                # runs the policy it asked for on whatever is fitted, until `equipment.fit`.
                if mode not in ("walk", "roller"):
                    raise ValueError("mode must be walk or roller")
                body.commanded_mode = mode
                return {}
            body.world.set_mode(body.index, mode)
            body.commanded_mode = mode
            body.set_mode_support(support)
            self.mode_support = support
            return {}
        if op == "slow":
            return body.slow_sensors()
        if op == "tof":
            return body.depth()
        # ── not the daemon's: the operator's, the test's and the scorer's ──
        if op == "equipment.fit":
            # A human taking skates on or off. No hand steadies the trunk while they do.
            body.world.set_mode(body.index, request["equipment"])
            return {"mode": body.world.modes[body.index]}
        if op == "oracle.pose":
            return body.oracle_pose()
        if op == "sim.info":
            world = body.world
            return {
                "protocol": PROTOCOL,
                "duck": body.index,
                "ducks": world.count,
                "realism": world.profile.name,
                "realism_seed": world.seed,
                "no_assist": world.no_assist,
                "obstacle_slots": world.obstacles.capacity,
                "mode": world.modes[body.index],
                "commanded_mode": body.commanded_mode,
            }
        if op == "obstacle.add":
            return {"obstacle": body.world.add_obstacle(request["obstacle"])}
        if op == "obstacle.remove":
            body.world.remove_obstacle(request["name"])
            return {}
        if op == "obstacle.move":
            return {"obstacle": body.world.move_obstacle(request["name"], request["pose"])}
        if op == "obstacle.list":
            return {
                "obstacles": body.world.obstacles.list(),
                "slots": body.world.obstacles.capacity,
            }
        raise ValueError(f"unknown op {op!r}")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def step_loop(world: World, stop: threading.Event) -> None:
    """Keep physics at the daemon's wall-clock rate, independently of rendering."""
    dt = world.model.opt.timestep
    batch = max(1, round(0.020 / dt))
    period = batch * dt
    next_step = time.perf_counter()
    behind = 0
    while not stop.is_set():
        world.step(batch)
        next_step += period
        slack = next_step - time.perf_counter()
        if slack > 0:
            stop.wait(slack)
        elif slack < -0.25:
            behind += 1
            print(
                f"== physics behind real time by {-slack:.2f}s (x{behind})",
                flush=True,
            )
            next_step = time.perf_counter()


def run(
    world: World, headless: bool, visual: VisualWorld | None = None,
    *, stop: threading.Event | None = None, camera_fps: int = CAMERA_FPS,
    viewer_visual: VisualWorld | EquipmentVisualWorld | None = None,
) -> None:
    """Step in real time.

    **Real time, not as fast as possible.** The daemon's loop is wall-clock and its health gate
    fails below 45 of 50 Hz, so a simulator running at its own pace does not merely look wrong — it
    makes every duck report unhealthy and the updater start rolling releases back.
    """
    if visual is None and any(b.camera is not None for b in world.bodies):
        raise ValueError("cameras require the visual world used to create their renderers")
    if not headless and viewer_visual is None:
        # The video stream keeps the real sensor's roll; the desktop view is upright.
        viewer_visual = VisualWorld(world, upright_cameras=True)
        viewer_visual.update()
    viewer = None
    if not headless:
        try:
            import mujoco.viewer

            # No side panels: this window is for watching ducks, and everything the panels would
            # drive belongs to the daemons.
            viewer = mujoco.viewer.launch_passive(
                viewer_visual.model, viewer_visual.data, show_left_ui=False, show_right_ui=False
            )
        except Exception as error:
            print(f"== no viewer ({error}); running headless", flush=True)

    stop = stop if stop is not None else threading.Event()
    eyes = [b for b in world.bodies if b.camera is not None]
    eye_period = 1.0 / max(1, camera_fps)
    render_period = min(1.0 / 30.0, eye_period) if viewer is not None else eye_period
    next_eye = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="duck-physics")
    physics = executor.submit(step_loop, world, stop)
    try:
        while not stop.is_set():
            if physics.done():
                physics.result()
                break
            if viewer is not None and not viewer.is_running():
                break
            started = time.perf_counter()
            viewer_frame = viewer is not None
            camera_frame = bool(eyes) and started >= next_eye
            if viewer_frame:
                old_model = viewer_visual.model
                with viewer.lock():
                    viewer_visual.update()
                if old_model is not viewer_visual.model:
                    import mujoco.viewer

                    lookat = viewer.cam.lookat.copy()
                    distance, azimuth, elevation = (
                        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation
                    )
                    viewer.close()
                    viewer = mujoco.viewer.launch_passive(
                        viewer_visual.model, viewer_visual.data,
                        show_left_ui=False, show_right_ui=False,
                    )
                    viewer.cam.lookat[:] = lookat
                    viewer.cam.distance = distance
                    viewer.cam.azimuth = azimuth
                    viewer.cam.elevation = elevation
                viewer.sync()
            if camera_frame:
                if visual is not viewer_visual or not viewer_frame:
                    visual.update()
                for body in eyes:
                    if body.camera is not None:
                        body.camera.render(visual, viewer if visual is viewer_visual else None)
                next_eye = started + eye_period
            stop.wait(max(0.0, render_period - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        executor.shutdown(wait=True)
        if viewer is not None:
            viewer.close()
        physics.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--ducks", type=int, default=1, help="how many, sharing one world")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7801, help="the first duck's port; +1 each")
    parser.add_argument("--headless", action="store_true", help="no viewer window")
    parser.add_argument(
        "--preload-equipment",
        action="store_true",
        help="compile and retain both equipment models at startup (one duck only)",
    )
    parser.add_argument(
        "--cameras",
        default="",
        help="which ducks render a head camera, by letter — `a`, `a,c`, or `all`. Opt in, because a "
        "rendered frame costs 12 ms and four cameras is most of a core; four ducks without them is "
        "nothing. Each becomes a frame port at --frame-port + its index",
    )
    parser.add_argument("--frame-port", type=int, default=7901, help="the first camera's port")
    parser.add_argument("--camera-fps", type=int, default=CAMERA_FPS)
    parser.add_argument(
        "--limp",
        action="store_true",
        help="start with no torque, so a duck collapses where it stands — a robot found on the "
        "floor, which is what `robotd`'s seated-boot path is for",
    )
    parser.add_argument(
        "--keyframe",
        default="SIT",
        help="where to start. SIT is a duck folded on the floor, which is stable while it waits and "
        "which the standing policy rises from on its own. HOME is infer_policy.py's placement — "
        "home pose, trunk 0.125 m, upright — and STAND and FOLD are the scene's other poses",
    )
    env = os.environ
    parser.add_argument(
        "--realism",
        choices=sorted(realism.PROFILES),
        default=env.get("DUCK_SIM_REALISM", "off"),
        help="sensor, bus and body imperfections (sim/realism.py). off (default) is the ideal "
        "twin; real is the envelope the policies were trained in; hard is past it. "
        "Env: DUCK_SIM_REALISM",
    )
    parser.add_argument(
        "--realism-seed",
        type=int,
        default=int(env.get("DUCK_SIM_REALISM_SEED", "0")),
        help="seed for every realism draw, so a run can be repeated. Env: DUCK_SIM_REALISM_SEED",
    )
    parser.add_argument(
        "--no-assist",
        action="store_true",
        default=env.get("DUCK_SIM_NO_ASSIST", "") not in ("", "0", "false", "no"),
        help="no simulation-only hand: nothing holds a duck before it is enabled or steadies it "
        "through a mode change or the home ramp, and `mode` no longer swaps equipment — "
        "`equipment.fit` does, as a human does on the robot. Env: DUCK_SIM_NO_ASSIST=1",
    )
    parser.add_argument(
        "--obstacles",
        type=Path,
        default=Path(env["DUCK_SIM_OBSTACLES"]) if env.get("DUCK_SIM_OBSTACLES") else None,
        help="JSON file of obstacles to place at launch (sim/obstacles.py). Env: DUCK_SIM_OBSTACLES",
    )
    parser.add_argument(
        "--obstacle-slots",
        type=int,
        default=int(env["DUCK_SIM_OBSTACLE_SLOTS"]) if env.get("DUCK_SIM_OBSTACLE_SLOTS") else None,
        help=f"how many obstacles can exist at once, for obstacle.add at runtime. Default 0, or "
        f"{DEFAULT_SLOTS} (at least the file's count) with --obstacles. Env: DUCK_SIM_OBSTACLE_SLOTS",
    )
    args = parser.parse_args()

    if not args.scene.exists():
        raise SystemExit(
            f"no scene at {args.scene}. Available:\n  "
            + "\n  ".join(sorted(p.name for p in SCENES.glob("scene*.xml")))
        )
    if args.ducks < 1:
        raise SystemExit("--ducks needs at least one duck")
    entries = load_obstacles(args.obstacles) if args.obstacles is not None else []
    slots = args.obstacle_slots
    if slots is None:
        slots = max(DEFAULT_SLOTS, len(entries)) if args.obstacles is not None else 0
    if slots < len(entries):
        raise SystemExit(f"--obstacle-slots {slots} cannot hold the file's {len(entries)} obstacles")

    world = World(
        args.scene, args.ducks, preload_equipment=args.preload_equipment,
        realism_profile=args.realism, realism_seed=args.realism_seed,
        no_assist=args.no_assist, obstacle_slots=slots,
    )
    for entry in entries:
        world.add_obstacle(entry)
    pose, trunk_z = pose_table(
        args.scene, args.keyframe,
        model=world.model if args.ducks == 1 and not slots else None,
    )
    wanted = set()
    if args.cameras.strip() == "all":
        wanted = set(range(args.ducks))
    elif args.cameras.strip():
        wanted = {ord(c.strip()) - ord("a") for c in args.cameras.split(",") if c.strip()}
    for index in range(args.ducks):
        body = Body(world, index, limp=args.limp)
        body.place(pose, trunk_z, offset_y=index * SPACING)
        world.bodies.append(body)

    # Finish costly render setup before accepting control-loop connections.
    mujoco.mj_forward(world.model, world.data)
    equipment_viewer = not args.headless and bool(world.preloaded_models)
    visual = VisualWorld(world) if wanted or (not args.headless and not equipment_viewer) else None
    if visual is not None:
        visual.update()
    viewer_visual = None
    if equipment_viewer:
        viewer_model = build_world(
            world.scene, 2, ["walk", "roller"], obstacle_slots=world.obstacles.capacity
        )
        viewer_visual = EquipmentVisualWorld(world, viewer_model)
        viewer_visual.update()
    for body in world.bodies:
        if body.index in wanted:
            body.camera = Camera(
                visual.model, f"{body.prefix}head_camera",
                preloaded_models=visual.preloaded_models,
            )

    servers = []
    for body in world.bodies:
        index = body.index
        server = Server((args.host, args.port + index), Handler)
        server.body = body
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        if body.camera is not None:
            frames = FrameServer((args.host, args.frame_port + index), FrameHandler)
            frames.camera = body.camera
            frames.fps = args.camera_fps
            threading.Thread(target=frames.serve_forever, daemon=True).start()
            servers.append(frames)

    print(f"== {args.scene.name}: {args.ducks} duck(s), starting at {args.keyframe}", flush=True)
    if world.profile.active or world.no_assist or slots:
        print(
            f"== realism {world.profile.name} (seed {world.seed}) · "
            f"assist {'off' if world.no_assist else 'on'} · "
            f"{len(entries)} obstacle(s) in {slots} slot(s)",
            flush=True,
        )
    for index in range(args.ducks):
        eye = f" · camera on {args.host}:{args.frame_port + index}" if index in wanted else ""
        print(f"==   duck {index}: robotd --sim {args.host}:{args.port + index}{eye}", flush=True)

    try:
        run(
            world, headless=args.headless, visual=visual, camera_fps=args.camera_fps,
            viewer_visual=viewer_visual,
        )
    finally:
        for body in world.bodies:
            if body.camera is not None:
                body.camera.close()


if __name__ == "__main__":
    main()
