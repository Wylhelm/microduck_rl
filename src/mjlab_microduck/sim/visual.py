"""A render-only copy of the world, including the commanded beak animation."""

from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

MOUTH_CLOSED = math.radians(-5.0)
MOUTH_OPEN = math.radians(30.0)


class VisualWorld:
    """Render from a separate model and data; never move a physics geom."""

    def __init__(self, world):
        self.world = world
        with world.lock:
            self.model = copy.copy(world.model)
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
            mujoco.mj_copyData(self.data, self.world.model, self.world.data)
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
