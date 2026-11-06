import math
import io
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mjlab_microduck.sim.body_server import (
    DEFAULT_SCENE, HOME_TRUNK_Z, JOINT_NAMES, MOUTH_INDEX, Body, Handler, World,
)
from mjlab_microduck.sim.visual import MOUTH_CLOSED, MOUTH_OPEN, VisualWorld


def make_world(count=1, scene=DEFAULT_SCENE):
    world = World(scene, count)
    for index in range(count):
        body = Body(world, index)
        body.place(None, HOME_TRUNK_Z, index * 0.5)
        world.bodies.append(body)
    mujoco.mj_forward(world.model, world.data)
    return world


def target(body, mouth):
    targets = [0.0] * len(JOINT_NAMES)
    for actuator, index in zip(body.actuators, body.to_wire):
        targets[index] = body.world.data.ctrl[actuator]
    targets[MOUTH_INDEX] = mouth
    body.set_targets(targets)


def test_visual_only_tracks_each_duck_and_closes():
    world = make_world(2)
    visual = VisualWorld(world)
    assert len(visual.jaws) == 2
    visual.update()
    a, b = [entry[1] for entry in visual.jaws]
    closed_a = visual.data.geom_xpos[a].copy()
    closed_b = visual.data.geom_xpos[b].copy()
    original_model_positions = world.model.geom_pos.copy()
    original_qpos = world.data.qpos.copy()
    original_ctrl = world.data.ctrl.copy()

    target(world.bodies[0], MOUTH_OPEN)
    visual.update()
    assert not np.allclose(visual.data.geom_xpos[a], closed_a)
    np.testing.assert_allclose(visual.data.geom_xpos[b], closed_b)
    np.testing.assert_array_equal(world.model.geom_pos, original_model_positions)
    np.testing.assert_array_equal(world.data.qpos, original_qpos)
    np.testing.assert_array_equal(world.data.ctrl, original_ctrl)
    sensors = world.bodies[0].sensors()
    for name in ("positions", "velocities", "currents_ma"):
        assert sensors[name][MOUTH_INDEX] == 0.0

    target(world.bodies[0], MOUTH_OPEN + 10)
    visual.update()
    clamped_open = visual.data.geom_xpos[a].copy()
    target(world.bodies[0], MOUTH_OPEN)
    visual.update()
    np.testing.assert_allclose(visual.data.geom_xpos[a], clamped_open)

    target(world.bodies[0], MOUTH_CLOSED)
    visual.update()
    np.testing.assert_allclose(visual.data.geom_xpos[a], closed_a)
    target(world.bodies[0], MOUTH_OPEN)
    world.bodies[0].set_torque(False)
    visual.update()
    np.testing.assert_allclose(visual.data.geom_xpos[a], closed_a)
    target(world.bodies[0], MOUTH_OPEN)
    world.bodies[0].close_visual_mouth()
    visual.update()
    np.testing.assert_allclose(visual.data.geom_xpos[a], closed_a)


def test_physics_rollout_is_identical_with_animated_mouth():
    animated = make_world()
    reference = make_world()
    visual = VisualWorld(animated)
    assert (animated.model.nq, animated.model.nv, animated.model.nu) == (21, 20, 14)
    for step in range(20):
        target(animated.bodies[0], MOUTH_OPEN if step % 2 else MOUTH_CLOSED)
        target(reference.bodies[0], MOUTH_CLOSED)
        animated.step()
        reference.step()
        visual.update()
        for name in ("qpos", "qvel", "ctrl", "qacc", "actuator_force"):
            np.testing.assert_array_equal(
                getattr(animated.data, name), getattr(reference.data, name)
            )
        assert animated.data.ncon == reference.data.ncon
    np.testing.assert_array_equal(animated.model.body_inertia, reference.model.body_inertia)
    np.testing.assert_array_equal(animated.model.geom_pos, reference.model.geom_pos)
    np.testing.assert_array_equal(animated.model.geom_quat, reference.model.geom_quat)


def test_nonfinite_mouth_target_is_rejected_without_state_change():
    world = make_world()
    before = world.data.ctrl.copy()
    with pytest.raises(ValueError, match="finite"):
        target(world.bodies[0], math.nan)
    np.testing.assert_array_equal(world.data.ctrl, before)
    assert world.bodies[0].visual_mouth == MOUTH_CLOSED


@pytest.mark.parametrize("scene_name", ["scene.xml", "scene_backlash.xml", "scene_walk.xml"])
def test_visual_mouth_in_supported_models(scene_name):
    world = make_world(scene=DEFAULT_SCENE.with_name(scene_name))
    visual = VisualWorld(world)
    assert len(visual.jaws) == 1
    visual.update()
    geom = visual.jaws[0][1][0]
    original = visual.data.geom_xpos[geom].copy()
    target(world.bodies[0], MOUTH_OPEN)
    visual.update()
    assert not np.allclose(visual.data.geom_xpos[geom], original)


def test_viewer_and_camera_scene_uses_visual_geometry():
    world = make_world()
    visual = VisualWorld(world)
    scene = mujoco.MjvScene(visual.model, maxgeom=1000)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    option = mujoco.MjvOption()
    geom = visual.jaws[0][1][0]

    def scene_position(mouth):
        target(world.bodies[0], mouth)
        visual.update()
        mujoco.mjv_updateScene(
            visual.model, visual.data, option, None, camera,
            mujoco.mjtCatBit.mjCAT_ALL, scene,
        )
        return next(
            g.pos.copy() for g in scene.geoms[:scene.ngeom]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == geom
        )

    closed = scene_position(MOUTH_CLOSED)
    opened = scene_position(MOUTH_OPEN)
    assert not np.allclose(closed, opened)
    np.testing.assert_allclose(scene_position(MOUTH_CLOSED), closed)


def test_disconnect_closes_visual_mouth():
    world = make_world()
    body = world.bodies[0]
    targets = [0.0] * len(JOINT_NAMES)
    targets[MOUTH_INDEX] = MOUTH_OPEN
    handler = object.__new__(Handler)
    handler.connection = SimpleNamespace(setsockopt=lambda *args: None)
    handler.server = SimpleNamespace(body=body)
    handler.client_address = ("offline", 0)
    handler.rfile = io.BytesIO((json.dumps({"op": "write", "targets": targets}) + "\n").encode())
    handler.wfile = io.BytesIO()
    handler.handle()
    assert json.loads(handler.wfile.getvalue()) == {}
    assert body.visual_mouth == MOUTH_CLOSED
