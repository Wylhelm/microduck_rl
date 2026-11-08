import os
import threading
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mjlab_microduck.sim import body_server
from mjlab_microduck.sim.body_server import (
    HOME_TRUNK_Z,
    SCENES,
    Body,
    Handler,
    World,
    build_world,
    floor_clearance,
    pose_table,
    run,
)
from mjlab_microduck.sim.camera import Camera
from mjlab_microduck.sim.visual import EquipmentVisualWorld, VisualWorld


def make_world(count=1, *, preload_equipment=False):
    world = World(SCENES / "scene_backlash.xml", count, preload_equipment=preload_equipment)
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


@pytest.mark.parametrize("preload_equipment", [False, True])
def test_equipment_is_reversible_and_preserves_pose_velocity_and_clearance(preload_equipment):
    world = make_world(preload_equipment=preload_equipment)
    body = world.bodies[0]
    original = body.sensors()
    clearance = body.floor_clearance()
    walking_gains = body._gain.copy()
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
        np.testing.assert_array_equal(
            world.model.actuator_gainprm[body.actuator_slice, 0], body._gain * 125 / 200
        )
        np.testing.assert_array_equal(
            world.model.actuator_biasprm[body.actuator_slice, 1], -body._gain * 125 / 200
        )
        if mode == "walk":
            np.testing.assert_array_equal(body._gain, walking_gains)
        visual.update()
        assert visual.source_model is world.model
        assert len(visual.jaws) == 1
        assert visual.model.nq == world.model.nq
        model = world.model
        world.set_mode(0, mode)
        assert world.model is model
    np.testing.assert_allclose(body.sensors()["trunk"], original["trunk"])


def test_preloaded_equipment_reuses_both_models_without_losing_gain_or_torque(monkeypatch):
    calls = []
    build = body_server.build_world

    def track_build(*args, **kwargs):
        calls.append((args, kwargs))
        return build(*args, **kwargs)

    monkeypatch.setattr(body_server, "build_world", track_build)
    world = make_world(preload_equipment=True)
    assert len(calls) == 2
    walking = world.model
    body = world.bodies[0]
    walking_gains = body._gain.copy()
    body.set_gain(80)
    body.set_torque(False)
    for mode in ("roller", "walk", "roller", "walk"):
        world.set_mode(0, mode)
        assert np.all(world.model.actuator_gainprm[body.actuator_slice, 0] == 0)
        body.set_torque(True)
        np.testing.assert_allclose(
            world.model.actuator_gainprm[body.actuator_slice, 0], body._gain * 0.4
        )
        if mode == "walk":
            assert world.model is walking
            np.testing.assert_array_equal(body._gain, walking_gains)
        body.set_torque(False)
    assert len(calls) == 2


def test_preloading_rejects_multiple_ducks():
    with pytest.raises(ValueError, match="exactly one duck"):
        make_world(2, preload_equipment=True)


def test_pose_table_can_reuse_initial_single_duck_model():
    scene = SCENES / "scene_backlash.xml"
    world = make_world(preload_equipment=True)
    assert pose_table(scene, "SIT", model=world.model) == pose_table(scene, "SIT")


def test_preloaded_equipment_can_start_in_roller_mode():
    world = World(SCENES / "scene_rollers.xml", preload_equipment=True)
    body = Body(world, 0)
    body.place(None, HOME_TRUNK_Z, 0)
    world.bodies.append(body)
    mujoco.mj_forward(world.model, world.data)
    assert world.modes == ["roller"]
    assert wheels(world)
    world.set_mode(0, "walk")
    assert not wheels(world)
    world.set_mode(0, "roller")
    assert wheels(world)


def test_equipment_viewer_keeps_one_model_and_mirrors_pose_and_visibility():
    world = make_world(preload_equipment=True)
    body = world.bodies[0]
    model = build_world(world.scene, 2, ["walk", "roller"])
    viewer = EquipmentVisualWorld(world, model)
    viewer_model = viewer.model
    for mode in ("walk", "roller", "walk"):
        world.set_mode(0, mode)
        world.data.qpos[body.trunk] = 0.23
        world.data.qpos[body.qpos_adr[0]] = 0.12
        if mode == "roller":
            wheel = wheels(world)[0]
            world.data.qpos[world.model.jnt_qposadr[wheel]] = 0.37
        mujoco.mj_forward(world.model, world.data)
        viewer.update()

        assert viewer.model is viewer_model
        source_trunk = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        for prefix in ("", "d1_"):
            target_trunk = mujoco.mj_name2id(
                viewer.model, mujoco.mjtObj.mjOBJ_BODY, prefix + "trunk_base"
            )
            np.testing.assert_allclose(
                viewer.data.xpos[target_trunk], world.data.xpos[source_trunk], atol=1e-10
            )
        prefix = "" if mode == "walk" else "d1_"
        for source in range(world.model.nbody):
            if world.model.body_rootid[source] != source_trunk:
                continue
            name = mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_BODY, source)
            if name is None:
                continue
            target = mujoco.mj_name2id(viewer.model, mujoco.mjtObj.mjOBJ_BODY, prefix + name)
            assert target >= 0
            np.testing.assert_allclose(
                viewer.data.xpos[target], world.data.xpos[source], atol=1e-10
            )
            np.testing.assert_allclose(
                viewer.data.xmat[target], world.data.xmat[source], atol=1e-10
            )
        for equipment in ("walk", "roller"):
            expected = viewer._alpha[equipment] if equipment == mode else 0
            np.testing.assert_array_equal(
                viewer.model.geom_rgba[viewer._geoms[equipment], 3], expected
            )
        if mode == "roller":
            target = mujoco.mj_name2id(
                viewer.model, mujoco.mjtObj.mjOBJ_JOINT,
                "d1_" + mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_JOINT, wheel),
            )
            assert viewer.data.qpos[viewer.model.jnt_qposadr[target]] == pytest.approx(0.37)


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
@pytest.mark.parametrize("preload_equipment", [False, True])
def test_head_camera_rebinds_and_renders_after_repeated_equipment_changes(preload_equipment):
    world = make_world(preload_equipment=preload_equipment)
    visual = VisualWorld(world)
    viewer = VisualWorld(world, upright_cameras=True)
    visual.update()
    camera = Camera(
        visual.model, "head_camera", width=64, height=36,
        preloaded_models=visual.preloaded_models,
    )
    initial_renderer = camera.renderer
    initial_visual_model = visual.model
    try:
        for mode in ("walk", "roller", "walk"):
            world.set_mode(0, mode)
            visual.update()
            camera.render(visual)
            viewer.update()
            assert camera.model is visual.model
            assert len(camera.frame()) == 64 * 36 * 2
            stream_axes = visual.data.cam_xmat[camera.camera].reshape(3, 3)
            viewer_camera = mujoco.mj_name2id(
                viewer.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera"
            )
            viewer_axes = viewer.data.cam_xmat[viewer_camera].reshape(3, 3)
            np.testing.assert_allclose(-stream_axes[:, 2], -viewer_axes[:, 2], atol=1e-10)
            np.testing.assert_allclose(stream_axes[:, 0], -viewer_axes[:, 1], atol=1e-10)
            if preload_equipment and mode == "walk":
                assert camera.renderer is initial_renderer
                assert visual.model is initial_visual_model
    finally:
        camera.close()


@pytest.mark.parametrize("preload_equipment", [False, True])
def test_viewer_head_camera_faces_forward_upright_after_equipment_changes(preload_equipment):
    world = make_world(preload_equipment=preload_equipment)
    stream = VisualWorld(world)
    viewer = VisualWorld(world, upright_cameras=True)
    for mode in ("walk", "roller", "walk"):
        world.set_mode(0, mode)
        for yaw in (0.0, 0.7):
            body = world.bodies[0]
            body.place(None, HOME_TRUNK_Z, 0)
            world.data.qpos[body.trunk + 3:body.trunk + 7] = (
                np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)
            )
            mujoco.mj_forward(world.model, world.data)
            original = world.model.cam_quat.copy()
            stream.update()
            viewer.update()
            camera = mujoco.mj_name2id(viewer.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
            axes = viewer.data.cam_xmat[camera].reshape(3, 3)
            np.testing.assert_allclose(
                -axes[:, 2], [np.cos(yaw), np.sin(yaw), 0], atol=1e-10
            )
            np.testing.assert_allclose(axes[:, 1], [0, 0, 1], atol=1e-10)
            np.testing.assert_array_equal(world.model.cam_quat, original)
            np.testing.assert_array_equal(stream.model.cam_quat, original)


def test_equipment_viewer_has_upright_cameras_for_both_robots():
    world = make_world(preload_equipment=True)
    viewer = EquipmentVisualWorld(world, build_world(world.scene, 2, ["walk", "roller"]))
    for mode in ("walk", "roller", "walk"):
        world.set_mode(0, mode)
        world.bodies[0].place(None, HOME_TRUNK_Z, 0)
        viewer.update()
        for name in ("head_camera", "d1_head_camera"):
            camera = mujoco.mj_name2id(viewer.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            axes = viewer.data.cam_xmat[camera].reshape(3, 3)
            np.testing.assert_allclose(-axes[:, 2], [1, 0, 0], atol=1e-10)
            np.testing.assert_allclose(axes[:, 1], [0, 0, 1], atol=1e-10)


def test_camera_closes_preloaded_renderers_if_initialization_fails(monkeypatch):
    world = make_world(preload_equipment=True)
    closed = []
    created = []

    def renderer(*_args, **_kwargs):
        if created:
            raise RuntimeError("renderer initialization failed")
        result = SimpleNamespace(close=lambda: closed.append(True))
        created.append(result)
        return result

    monkeypatch.setattr(mujoco, "Renderer", renderer)
    with pytest.raises(RuntimeError, match="renderer initialization failed"):
        Camera(
            world.model, "head_camera", preloaded_models=(world.preloaded_models["roller"][0],)
        )
    assert closed == [True]


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


def test_slow_camera_does_not_slow_physics():
    world = make_world()
    frames = []
    stop = threading.Event()
    errors = []

    def render(*args):
        time.sleep(0.1)
        frames.append(world.data.time)

    world.bodies[0].camera = SimpleNamespace(render=render)
    visual = SimpleNamespace(model=world.model, update=lambda: None)

    def drive():
        try:
            run(world, headless=True, visual=visual, stop=stop)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=drive)
    thread.start()
    started = time.perf_counter()
    try:
        time.sleep(0.8)
    finally:
        stop.set()
        thread.join(timeout=3)
    elapsed = time.perf_counter() - started
    assert not thread.is_alive()
    assert not errors
    assert len(frames) >= 3
    assert world.data.time >= elapsed * 0.8


def test_physics_worker_errors_reach_the_main_loop():
    def fail(*args):
        raise ValueError("physics failed")

    world = SimpleNamespace(
        model=SimpleNamespace(opt=SimpleNamespace(timestep=0.005)),
        bodies=[],
        step=fail,
    )
    with pytest.raises(ValueError, match="physics failed"):
        run(world, headless=True)
