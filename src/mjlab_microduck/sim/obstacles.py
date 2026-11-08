"""Things nobody put on the map: a chair in the corridor, a closed door, a bag, a cat.

    duck-body --scene /path/to/scene.xml --obstacles /path/to/obstacles.json
    {"op": "obstacle.add", "obstacle": {"name": "box", "shape": "box", "size": [0.3, 0.2, 0.25],
                                        "pose": {"x": 1.0, "y": 0.0, "yaw_deg": 30}}}

**Slots, compiled in once.** Adding a body to a compiled MuJoCo model means recompiling it, which
takes seconds for the apartment and would drop every daemon's control tick on the floor. So the
world is compiled with a fixed number of mocap bodies (`obstacle_0` …), each with `PARTS` box geoms
parked under the floor, collision off and invisible. An obstacle is a slot whose geoms are given a
shape, a colour and the floor's collision bits, and whose mocap body is moved to its pose. Mocap
bodies are kinematic: they push the duck and are not pushed back, which is what a door or a heavy
chair does, and what a cat that ignores you does.

**Seen, not just felt.** The geoms are in group 0 with an opaque colour, so the head camera renders
them and the ToF rays (`mj_ray` skips only fully transparent geoms) hit them. A translucent colour
(alpha < 0.6) makes a glass panel, which the `real`/`hard` ToF models treat as glass.

**One logical state, many models.** The physics model, the render copies and the equipment viewer
are separate `MjModel`s; `sync(model, data)` writes the current obstacles into any of them, by name,
so each can have its own ids.

File format (`--obstacles FILE`): `{"obstacles": [ {...}, ... ]}` or a bare list, each entry:

    name      unique, required
    shape     box | cylinder | bag | chair | door | parts
    size      box/bag: [x, y, z] full extents in metres; cylinder: [diameter, height];
              chair: [width, depth, seat_height, back_height]; door: [width, thickness, height]
    parts     shape "parts" only: [{"size": [x, y, z], "pos": [x, y, z]}, ...] boxes, body frame
    pose      {"x", "y", "z" = 0, "yaw_deg" = 0}: world frame, z of the obstacle's bottom
    rgba      optional colour, alpha < 0.6 for glass
    motion    optional {"waypoints": [[x, y], ...], "speed": m/s}: ping-pong along the waypoints,
              heading along the way — a pet that wanders a fixed beat
"""

from __future__ import annotations

import json
import math
import threading
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

PARTS = 6
PARK_Z = -20.0
# The slot geoms are compiled this large so the compiled bounding volumes cover any runtime shape:
# MuJoCo does not recompute a body's BVH when a geom changes size at runtime.
SLOT_HALF = 2.0
DEFAULT_SLOTS = 8
SHAPES = ("box", "bag", "cylinder", "chair", "door", "parts")
DEFAULT_RGBA = {
    "box": [0.55, 0.42, 0.28, 1.0],
    "bag": [0.15, 0.2, 0.35, 1.0],
    "cylinder": [0.6, 0.6, 0.6, 1.0],
    "chair": [0.45, 0.3, 0.18, 1.0],
    "door": [0.85, 0.83, 0.78, 1.0],
    "parts": [0.5, 0.5, 0.5, 1.0],
}


def slot_body(slot: int) -> str:
    return f"obstacle_{slot}"


def slot_geom(slot: int, part: int) -> str:
    return f"obstacle_{slot}_{part}"


def add_slots(spec: mujoco.MjSpec, count: int) -> None:
    for slot in range(count):
        body = spec.worldbody.add_body(
            name=slot_body(slot), mocap=True, pos=[0.0, 0.0, PARK_Z]
        )
        for part in range(PARTS):
            body.add_geom(
                name=slot_geom(slot, part),
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[SLOT_HALF] * 3,
                contype=0,
                conaffinity=0,
                group=0,
                rgba=[0.5, 0.5, 0.5, 0.0],
            )


def slot_count(model: mujoco.MjModel) -> int:
    count = 0
    while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, slot_body(count)) >= 0:
        count += 1
    return count


@dataclass
class Part:
    kind: int  # mjtGeom
    size: tuple[float, float, float]  # MuJoCo half-sizes
    pos: tuple[float, float, float]  # in the obstacle's frame, z up from its bottom


@dataclass
class Obstacle:
    name: str
    shape: str
    parts: list[Part]
    rgba: list[float]
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    speed: float = 0.0
    spec: dict = field(default_factory=dict)

    def describe(self) -> dict:
        out = dict(self.spec)
        out["pose"] = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw_deg": math.degrees(self.yaw),
        }
        return out


def _box(
    sx: float, sy: float, sz: float, x: float = 0.0, y: float = 0.0, z0: float = 0.0
) -> Part:
    return Part(
        mujoco.mjtGeom.mjGEOM_BOX, (sx / 2, sy / 2, sz / 2), (x, y, z0 + sz / 2)
    )


def _floats(value, count: int, what: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{what} must be a list of {count} numbers")
    out = [float(v) for v in value]
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{what} must be finite")
    return out


def parse(entry: dict) -> Obstacle:
    if not isinstance(entry, dict):
        raise ValueError("an obstacle must be an object")
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("an obstacle needs a name")
    shape = entry.get("shape", "box")
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {', '.join(SHAPES)}")
    if shape in ("box", "bag"):
        sx, sy, sz = _floats(entry.get("size"), 3, "size")
        parts = [_box(sx, sy, sz)]
    elif shape == "cylinder":
        diameter, height = _floats(entry.get("size"), 2, "size")
        parts = [
            Part(
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                (diameter / 2, height / 2, 0.0),
                (0.0, 0.0, height / 2),
            )
        ]
    elif shape == "door":
        width, thickness, height = _floats(
            entry.get("size", [0.8, 0.04, 2.0]), 3, "size"
        )
        parts = [_box(thickness, width, height)]
    elif shape == "chair":
        width, depth, seat, back = _floats(
            entry.get("size", [0.42, 0.42, 0.45, 0.4]), 4, "size"
        )
        leg, plate = 0.03, 0.03
        dx, dy = depth / 2 - leg / 2, width / 2 - leg / 2
        parts = [
            _box(leg, leg, seat - plate, sx * dx, sy * dy)
            for sx in (-1, 1)
            for sy in (-1, 1)
        ]
        parts.append(_box(depth, width, plate, z0=seat - plate))
        parts.append(_box(plate, width, back, -depth / 2 + plate / 2, z0=seat))
    else:
        raw = entry.get("parts")
        if not isinstance(raw, list) or not raw:
            raise ValueError("shape 'parts' needs a non-empty parts list")
        parts = []
        for item in raw:
            size = _floats(item.get("size"), 3, "part size")
            pos = _floats(item.get("pos", [0, 0, 0]), 3, "part pos")
            parts.append(
                Part(mujoco.mjtGeom.mjGEOM_BOX, tuple(s / 2 for s in size), tuple(pos))
            )
    if len(parts) > PARTS:
        raise ValueError(f"an obstacle has at most {PARTS} parts")
    for part in parts:
        if min(part.size[:2]) <= 0 or (
            part.kind == mujoco.mjtGeom.mjGEOM_BOX and part.size[2] <= 0
        ):
            raise ValueError("obstacle sizes must be positive")

    # Centre the parts on the mocap frame, so they stay inside the compiled bounding volume.
    low = np.min([np.subtract(p.pos, _half(p)) for p in parts], axis=0)
    high = np.max([np.add(p.pos, _half(p)) for p in parts], axis=0)
    if np.any(high - low > 2 * SLOT_HALF):
        raise ValueError(f"an obstacle must fit in {2 * SLOT_HALF:g} m on every side")

    rgba = entry.get("rgba", DEFAULT_RGBA[shape])
    rgba = _floats(rgba, 4, "rgba")
    if rgba[3] <= 0:
        raise ValueError(
            "an obstacle must not be fully transparent: nothing would see it"
        )
    pose = entry.get("pose", {})
    if not isinstance(pose, dict):
        raise ValueError("pose must be an object")
    obstacle = Obstacle(
        name=name,
        shape=shape,
        parts=parts,
        rgba=rgba,
        x=float(pose.get("x", 0.0)),
        y=float(pose.get("y", 0.0)),
        z=float(pose.get("z", 0.0)),
        yaw=math.radians(float(pose.get("yaw_deg", 0.0))),
        spec={k: v for k, v in entry.items() if k != "pose"},
    )
    motion = entry.get("motion")
    if motion is not None:
        points = [tuple(_floats(p, 2, "waypoint")) for p in motion.get("waypoints", [])]
        speed = float(motion.get("speed", 0.3))
        if len(points) < 2 or not speed > 0:
            raise ValueError("motion needs at least two waypoints and a positive speed")
        obstacle.waypoints, obstacle.speed = points, speed
        obstacle.x, obstacle.y = points[0]
    return obstacle


def _half(part: Part) -> np.ndarray:
    if part.kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return np.array([part.size[0], part.size[0], part.size[1]])
    return np.array(part.size)


def load(path: Path) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    entries = raw.get("obstacles") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError(
            f'{path}: expected a list of obstacles or {{"obstacles": [...]}}'
        )
    return entries


class Obstacles:
    """The obstacles in a world: which slot holds what, and where it is now."""

    def __init__(self, slots: int):
        self.slots: list[Obstacle | None] = [None] * slots
        self.lock = threading.Lock()
        self.version = 0
        self._id_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    @property
    def capacity(self) -> int:
        return len(self.slots)

    def find(self, name: str) -> int:
        for slot, obstacle in enumerate(self.slots):
            if obstacle is not None and obstacle.name == name:
                return slot
        raise KeyError(f"no obstacle named {name!r}")

    def add(self, entry: dict) -> Obstacle:
        obstacle = parse(entry)
        with self.lock:
            if any(o is not None and o.name == obstacle.name for o in self.slots):
                raise ValueError(f"an obstacle named {obstacle.name!r} already exists")
            if None not in self.slots:
                raise ValueError(
                    f"all {self.capacity} obstacle slots are in use (start with --obstacle-slots)"
                )
            self.slots[self.slots.index(None)] = obstacle
            self.version += 1
        return obstacle

    def remove(self, name: str) -> None:
        with self.lock:
            self.slots[self.find(name)] = None
            self.version += 1

    def move(self, name: str, pose: dict) -> Obstacle:
        if not isinstance(pose, dict):
            raise ValueError("pose must be an object")
        with self.lock:
            obstacle = self.slots[self.find(name)]
            x = float(pose.get("x", obstacle.x))
            y = float(pose.get("y", obstacle.y))
            z = float(pose.get("z", obstacle.z))
            yaw = (
                math.radians(float(pose["yaw_deg"]))
                if "yaw_deg" in pose
                else obstacle.yaw
            )
            if not all(math.isfinite(v) for v in (x, y, z, yaw)):
                raise ValueError("pose must be finite")
            obstacle.x, obstacle.y, obstacle.z, obstacle.yaw = x, y, z, yaw
            # A moved pet stops its beat: whoever moves it is driving it now.
            obstacle.waypoints = []
            self.version += 1
            return obstacle

    def list(self) -> list[dict]:
        with self.lock:
            return [o.describe() for o in self.slots if o is not None]

    def moving(self) -> bool:
        return any(o is not None and o.waypoints for o in self.slots)

    def advance(self, time: float) -> None:
        """Move every obstacle with a beat to where it is at simulated `time`."""
        with self.lock:
            self._advance(time)

    def _advance(self, time: float) -> None:
        for obstacle in self.slots:
            if obstacle is None or not obstacle.waypoints:
                continue
            points = np.array(obstacle.waypoints)
            legs = np.linalg.norm(np.diff(points, axis=0), axis=1)
            total = float(legs.sum())
            if total <= 0:
                continue
            travelled = (time * obstacle.speed) % (2 * total)
            forward = travelled <= total
            along = travelled if forward else 2 * total - travelled
            for leg, length in enumerate(legs):
                if along <= length or leg == len(legs) - 1:
                    a, b = points[leg], points[leg + 1]
                    t = min(1.0, along / length) if length > 0 else 0.0
                    obstacle.x, obstacle.y = (a + (b - a) * t).tolist()
                    heading = b - a if forward else a - b
                    obstacle.yaw = math.atan2(heading[1], heading[0])
                    break
                along -= length

    def _ids(self, model: mujoco.MjModel) -> list[tuple[int, int, list[int]]]:
        cached = self._id_cache.get(model)
        if cached is not None:
            return cached
        out = []
        for slot in range(min(self.capacity, slot_count(model))):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, slot_body(slot))
            geoms = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, slot_geom(slot, p))
                for p in range(PARTS)
            ]
            out.append((body, int(model.body_mocapid[body]), geoms))
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # What the duck collides with, it collides with: the floor's bits, whatever the scene's.
        contype = int(model.geom_contype[floor]) if floor >= 0 else 0
        conaffinity = int(model.geom_conaffinity[floor]) if floor >= 0 else 0
        self._id_cache[model] = (out, contype or 1, conaffinity or 1)
        return self._id_cache[model]

    def sync(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        *,
        geometry: bool = True,
    ) -> None:
        """Write the obstacles into `model` (shapes, colour, collision) and `data` (poses)."""
        if not self.capacity:
            return
        with self.lock:
            self._sync(model, data, geometry)

    def _sync(
        self, model: mujoco.MjModel, data: mujoco.MjData | None, geometry: bool
    ) -> None:
        slots, contype, conaffinity = self._ids(model)
        for slot, (body, mocap, geoms) in enumerate(slots):
            obstacle = self.slots[slot]
            if geometry:
                active = obstacle is not None
                model.body_contype[body] = contype if active else 0
                model.body_conaffinity[body] = conaffinity if active else 0
                for part, geom in enumerate(geoms):
                    piece = (
                        obstacle.parts[part]
                        if active and part < len(obstacle.parts)
                        else None
                    )
                    if piece is None:
                        model.geom_contype[geom] = 0
                        model.geom_conaffinity[geom] = 0
                        model.geom_rgba[geom, 3] = 0.0
                        model.geom_pos[geom] = 0.0
                        continue
                    half = _half(piece)
                    centre = self._centre(obstacle)
                    model.geom_type[geom] = piece.kind
                    model.geom_size[geom] = piece.size
                    model.geom_pos[geom] = np.subtract(piece.pos, centre)
                    model.geom_quat[geom] = [1.0, 0.0, 0.0, 0.0]
                    # Compiled at the body origin, the slot is flagged "same frame" and
                    # kinematics would ignore geom_pos: every part would collapse onto the centre.
                    model.geom_sameframe[geom] = 0
                    model.geom_rgba[geom] = obstacle.rgba
                    model.geom_contype[geom] = contype
                    model.geom_conaffinity[geom] = conaffinity
                    model.geom_rbound[geom] = float(np.linalg.norm(half))
                    model.geom_aabb[geom] = [0.0, 0.0, 0.0, *half]
            if data is not None:
                if obstacle is None:
                    data.mocap_pos[mocap] = [0.0, 0.0, PARK_Z]
                    data.mocap_quat[mocap] = [1.0, 0.0, 0.0, 0.0]
                else:
                    centre = self._centre(obstacle)
                    c, s = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
                    data.mocap_pos[mocap] = [
                        obstacle.x + c * centre[0] - s * centre[1],
                        obstacle.y + s * centre[0] + c * centre[1],
                        obstacle.z + centre[2],
                    ]
                    data.mocap_quat[mocap] = [
                        math.cos(obstacle.yaw / 2),
                        0.0,
                        0.0,
                        math.sin(obstacle.yaw / 2),
                    ]

    def _centre(self, obstacle: Obstacle) -> np.ndarray:
        low = np.min([np.subtract(p.pos, _half(p)) for p in obstacle.parts], axis=0)
        high = np.max([np.add(p.pos, _half(p)) for p in obstacle.parts], axis=0)
        return (low + high) / 2
