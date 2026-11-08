"""A render-only copy of the world, including the commanded beak animation."""

from __future__ import annotations

import copy
import math
import threading
from types import SimpleNamespace

import mujoco
import numpy as np

MOUTH_CLOSED = math.radians(-5.0)
MOUTH_OPEN = math.radians(30.0)


class VisualWorld:
    """Render from separate data and a model that is not used for physics."""

    def __init__(self, world, *, own_model: bool = False, upright_cameras: bool = False):
        self.world = world
        self._own_model = own_model
        self._upright_cameras = upright_cameras
        self._preloaded = {}
        with world.lock:
            self._bind_model(world.model)
            if world.preloaded_models:
                self._save_model()
                for source, _ in world.preloaded_models.values():
                    if source is not world.model:
                        self._bind_model(source)
                        self._save_model()
                self._restore_model(world.model)

    def _save_model(self) -> None:
        self._preloaded[id(self.source_model)] = (
            self.source_model, self.model, self.data, self.jaws
        )

    def _restore_model(self, source: mujoco.MjModel) -> None:
        self.source_model, self.model, self.data, self.jaws = self._preloaded[id(source)]

    @property
    def preloaded_models(self) -> tuple[mujoco.MjModel, ...]:
        return tuple(state[1] for state in self._preloaded.values())

    def _bind_model(self, source: mujoco.MjModel) -> None:
        world = self.world
        self.source_model = source
        self.model = source if self._own_model else copy.copy(source)
        if self._upright_cameras:
            # Reverse the CAD camera's backward view, then undo its sensor-mount roll.
            turn = np.array([0.0, 1.0, 0.0, 0.0])
            roll = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
            for camera in range(self.model.ncam):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
                if name is not None and name.endswith("head_camera"):
                    fixed = np.empty(4)
                    mujoco.mju_mulQuat(fixed, self.model.cam_quat[camera], turn)
                    mujoco.mju_mulQuat(self.model.cam_quat[camera], fixed, roll)
        self.data = mujoco.MjData(self.model)
        self.jaws = []
        for body in world.bodies:
            head = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body.prefix + "jaw_soft"
            )
            meshes = {
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, body.prefix + name)
                for name in ("jaw", "jaw_soft")
            }
            geoms = [
                i for i in range(self.model.ngeom)
                if self.model.geom_bodyid[i] == head
                and self.model.geom_group[i] == 2
                and self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                and self.model.geom_dataid[i] in meshes
            ]
            if head < 0 or -1 in meshes or len(geoms) != 2:
                print(f"== duck {body.index}: beak meshes unavailable; no visual mouth", flush=True)
                continue
            # CAD jaw linkage pivot in the jaw_soft body's frame. Only the two visual meshes
            # rotate; collision meshes and mouth_tip stay at their physical model positions.
            pivot = np.array([-0.0045, 0.0, -0.018])
            self.jaws.append(
                (body, geoms, self.model.geom_pos[geoms].copy(),
                 self.model.geom_quat[geoms].copy(), pivot)
            )

    def update(self) -> None:
        with self.world.lock:
            if self.source_model is not self.world.model:
                if self._preloaded:
                    self._restore_model(self.world.model)
                else:
                    self._bind_model(self.world.model)
            mujoco.mj_copyData(self.data, self.world.model, self.world.data)
            obstacles = getattr(self.world, "obstacles", None)
            if obstacles is not None:
                # Copies of the model do not see obstacles added after they were made.
                obstacles.sync(self.model)
            targets = [body.visual_mouth for body, *_ in self.jaws]

        for (_, geoms, positions, quats, pivot), target in zip(self.jaws, targets):
            angle = min(max(target, MOUTH_CLOSED), MOUTH_OPEN) - MOUTH_CLOSED
            rotation = np.array([math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0])
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, rotation)
            for geom, pos, quat in zip(geoms, positions, quats):
                self.model.geom_pos[geom] = pivot + matrix.reshape(3, 3) @ (pos - pivot)
                mujoco.mju_mulQuat(self.model.geom_quat[geom], rotation, quat)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)


class EquipmentVisualWorld:
    """Keep one viewer model with both robots, displaying only the active equipment.

    The physics duck's named joints are mirrored onto both variants. Keeping the viewer model
    fixed avoids rebuilding its window and blocking the control loop on each equipment change.
    """

    def __init__(self, world, model: mujoco.MjModel):
        if world.count != 1 or not world.preloaded_models or len(world.bodies) != 1:
            raise ValueError("the equipment viewer requires one duck with preloaded models")
        self.world = world
        bodies = [
            SimpleNamespace(index=0, prefix="", visual_mouth=MOUTH_CLOSED),
            SimpleNamespace(index=0, prefix="d1_", visual_mouth=MOUTH_CLOSED),
        ]
        mirror = SimpleNamespace(
            model=model, data=mujoco.MjData(model), bodies=bodies,
            lock=threading.Lock(), preloaded_models={},
        )
        self._mirror = mirror
        self._visual = VisualWorld(mirror, own_model=True, upright_cameras=True)
        self.model, self.data = self._visual.model, self._visual.data
        self._geoms = {}
        self._alpha = {}
        for mode, prefix in (("walk", ""), ("roller", "d1_")):
            trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "trunk_base")
            if trunk < 0:
                raise ValueError(f"the viewer model has no {prefix}trunk_base")
            geoms = np.array([
                geom for geom in range(model.ngeom)
                if model.body_rootid[model.geom_bodyid[geom]] == trunk
            ], dtype=int)
            if not len(geoms):
                raise ValueError(f"the viewer model has no {mode} robot geometry")
            self._geoms[mode] = geoms
            self._alpha[mode] = model.geom_rgba[geoms, 3].copy()
        self._mappings = {
            id(source): self._joint_mappings(source)
            for source, _ in world.preloaded_models.values()
        }
        self.mode: str | None = None

    def _joint_mappings(self, source: mujoco.MjModel) -> list[tuple[int, int, int]]:
        trunk = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        if trunk < 0:
            raise ValueError("the physics model has no trunk_base")
        mappings = []
        for joint in range(source.njnt):
            name = mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if name is None:
                raise ValueError("the equipment viewer requires named joints")
            is_duck = source.body_rootid[source.jnt_bodyid[joint]] == trunk
            targets = (name, "d1_" + name) if is_duck else (name,)
            width = 7 if source.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE else (
                4 if source.jnt_type[joint] == mujoco.mjtJoint.mjJNT_BALL else 1
            )
            matched = False
            for target in targets:
                dest = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, target)
                if dest < 0:
                    if is_duck:
                        continue
                    raise ValueError(f"the viewer model has no joint {target!r}")
                if self.model.jnt_type[dest] != source.jnt_type[joint]:
                    raise ValueError(f"the viewer joint {target!r} has an incompatible type")
                mappings.append((
                    int(source.jnt_qposadr[joint]), int(self.model.jnt_qposadr[dest]), width
                ))
                matched = True
            if not matched:
                raise ValueError(f"the viewer model has no matching joint for {name!r}")
        return mappings

    def update(self) -> None:
        with self.world.lock:
            source, data = self.world.model, self.world.data
            for old, new, width in self._mappings[id(source)]:
                self._mirror.data.qpos[new:new + width] = data.qpos[old:old + width]
            self._mirror.data.time = data.time
            obstacles = getattr(self.world, "obstacles", None)
            if obstacles is not None:
                obstacles.sync(self._mirror.model, self._mirror.data)
            mode = self.world.modes[0]
            mouth = self.world.bodies[0].visual_mouth
        for body in self._mirror.bodies:
            body.visual_mouth = mouth
        if mode != self.mode:
            for equipment, geoms in self._geoms.items():
                self.model.geom_rgba[geoms, 3] = self._alpha[equipment] if equipment == mode else 0
            self.mode = mode
        self._visual.update()
