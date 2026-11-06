import os
import threading

import mujoco
import numpy as np
import pytest

from mjlab_microduck.sim.body_server import (
    HOME_TRUNK_Z,
    SCENES,
    Body,
    Handler,
    World,
    floor_clearance,
)
from mjlab_microduck.sim.camera import Camera
from mjlab_microduck.sim.visual import VisualWorld


def make_world(count=1):
    world = World(SCENES / "scene_backlash.xml", count)
    for index in range(count):
        body = Body(world, index)
        body.place(None, HOME_TRUNK_Z, index * 0.5)
        world.bodies.append(body)
    mujoco.mj_forward(world.model, world.data)
    return world


def wheels(world, prefix=""):
    return [
        joint
        for joint in range(world.model.njnt)
        if (name := mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_JOINT, joint))
        and name.startswith(prefix + "passive_")
        and "wheel" in name
    ]


def test_equipment_is_reversible_and_preserves_pose_velocity_and_clearance():
    world = make_world()
    body = world.bodies[0]
    original = body.sensors()
    clearance = body.floor_clearance()
    world.data.qvel[body.trunk_dof] = 0.2
    body.set_gain(125)
    visual = VisualWorld(world)
    assert not wheels(world)
    for mode in ("roller", "walk", "roller", "walk"):
        world.set_mode(0, mode)
        assert bool(wheels(world)) == (mode == "roller")
        current = body.sensors()
        assert current["mode"] == mode
        np.testing.assert_array_equal(current["positions"], original["positions"])
        np.testing.assert_allclose(current["trunk"][:2], original["trunk"][:2])
        assert world.data.qvel[body.trunk_dof] == pytest.approx(0.2)
        assert body.floor_clearance() == pytest.approx(clearance)
        assert body.kp == 125
        visual.update()
        assert visual.source_model is world.model
        assert len(visual.jaws) == 1
        assert visual.model.nq == world.model.nq
        model = world.model
        world.set_mode(0, mode)
        assert world.model is model
    np.testing.assert_allclose(body.sensors()["trunk"], original["trunk"])


def test_other_ducks_keep_their_state_and_equipment():
    world = make_world(2)
    other = world.bodies[1]
    original = other.sensors()
    world.set_mode(0, "roller")
    assert wheels(world)
    assert not wheels(world, "d1_")
    np.testing.assert_array_equal(other.sensors()["positions"], original["positions"])
    np.testing.assert_array_equal(other.sensors()["trunk"], original["trunk"])
    world.set_mode(1, "roller")
    assert wheels(world, "d1_")
    world.set_mode(0, "walk")
    assert not wheels(world)
    assert wheels(world, "d1_")
    assert other.sensors()["mode"] == "roller"


def test_held_duck_and_control_targets_survive_equipment_change():
    world = make_world()
    body = world.bodies[0]
    before = world.data.ctrl.copy()
    world.set_mode(0, "roller")
    np.testing.assert_array_equal(world.data.ctrl, before)
    held = body.sensors()["trunk"]
    world.step(20)
    np.testing.assert_array_equal(body.sensors()["trunk"], held)
    assert not body.released


def test_skates_roll_in_contact_with_the_floor():
    world = make_world()
    body = world.bodies[0]
    world.set_mode(0, "roller")
    world.data.qpos[body.trunk + 2] -= body.floor_clearance()
    mujoco.mj_forward(world.model, world.data)
    body.set_torque(True)
    world.data.qvel[body.trunk_dof] = 0.3
    start = body.sensors()["trunk"][0]
    world.step(20)
    assert body.sensors()["trunk"][0] - start > 0.01
    wheel_rates = world.data.qvel[world.model.jnt_dofadr[wheels(world)]]
    assert np.max(np.abs(wheel_rates)) > 1
    assert np.all(np.isfinite(world.data.qpos))


def test_bad_mode_does_not_change_physics():
    world = make_world()
    model = world.model
    with pytest.raises(ValueError, match="mode must"):
        world.set_mode(0, "ice")
    assert world.model is model
    assert world.modes == ["walk"]


def test_transition_support_leaves_joints_free_and_expires_before_gliding():
    world = make_world(2)
    body = world.bodies[0]
    other = world.bodies[1].sensors()
    body.set_torque(True)
    handler = object.__new__(Handler)
    assert handler.dispatch(body, {"op": "mode", "mode": "walk", "support": True}) == {}
    held = body.support_pose.copy()
    world.data.ctrl[body.actuators[3]] += 0.1
    before = world.data.qpos[body.qpos_adr[3]]
    world.step(40)
    np.testing.assert_array_equal(world.data.qpos[body.trunk : body.trunk + 7], held)
    assert abs(world.data.qpos[body.qpos_adr[3]] - before) > 0.001
    np.testing.assert_array_equal(world.bodies[1].sensors()["trunk"], other["trunk"])
    world.set_mode(0, "roller")
    assert body.sensors()["supported"]
    assert body.torque_on and body.released
    body.set_mode_support(False)
    world.step(61)
    assert not body.sensors()["supported"]
    world.data.qvel[body.trunk_dof] = 0.3
    before = body.sensors()["trunk"][0]
    world.step(20)
    assert body.sensors()["trunk"][0] - before > 0.01


def test_torque_off_clears_support_and_bad_support_is_refused():
    world = make_world()
    body = world.bodies[0]
    handler = object.__new__(Handler)
    body.set_torque(True)
    body.set_mode_support(True)
    assert body.sensors()["supported"]
    body.set_torque(False)
    assert not body.sensors()["supported"]
    assert body.support_until is None
    with pytest.raises(ValueError, match="support must be a boolean"):
        handler.dispatch(body, {"op": "mode", "mode": "roller", "support": "yes"})
    assert body.sensors()["mode"] == "walk"


@pytest.mark.skipif(
    os.environ.get("DUCK_TEST_RENDER") != "1", reason="requires an OpenGL context"
)
def test_head_camera_rebinds_and_renders_after_repeated_equipment_changes():
    world = make_world()
    visual = VisualWorld(world)
    visual.update()
    camera = Camera(visual.model, "head_camera", width=64, height=36)
    try:
        for mode in ("walk", "roller", "walk"):
            world.set_mode(0, mode)
            visual.update()
            camera.render(visual)
            assert camera.model is visual.model
            assert len(camera.frame()) == 64 * 36 * 2
    finally:
        camera.renderer.close()


def test_unsupported_floor_refuses_without_replacing_model():
    world = make_world()
    floor = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    world.model.geom_type[floor] = mujoco.mjtGeom.mjGEOM_BOX
    model = world.model
    with pytest.raises(ValueError, match="floor plane"):
        world.set_mode(0, "roller")
    assert world.model is model
    assert world.modes == ["walk"]


def test_protocol_dispatch_and_sensor_reads_during_switch():
    world = make_world()
    body = world.bodies[0]
    handler = object.__new__(Handler)
    errors = []
    done = threading.Event()

    def read():
        try:
            while not done.is_set():
                sample = body.sensors()
                assert len(sample["positions"]) == 15
                assert np.all(np.isfinite(sample["positions"]))
        except Exception as error:
            errors.append(error)

    reader = threading.Thread(target=read)
    reader.start()
    try:
        for mode in ("roller", "walk"):
            assert handler.dispatch(body, {"op": "mode", "mode": mode}) == {}
            assert handler.dispatch(body, {"op": "read"})["mode"] == mode
            assert np.isfinite(floor_clearance(world.model, world.data, ""))
    finally:
        done.set()
        reader.join(timeout=2)
    assert not reader.is_alive()
    assert not errors
