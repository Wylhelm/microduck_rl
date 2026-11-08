"""Realism profiles and no-assist mode of the simulated body (`sim/realism.py`, `body_server.py`)."""

import dataclasses
import math

import mujoco
import numpy as np
import pytest

from mjlab_microduck.sim import realism
from mjlab_microduck.sim.body_server import (
    HOME_POSE,
    HOME_TRUNK_Z,
    JOINT_NAMES,
    SCENES,
    Body,
    Handler,
    World,
    gravity_in_trunk,
)
from mjlab_microduck.sim.tof import Tof

SCENE = SCENES / "scene_backlash.xml"


def make_world(
    profile="off", seed=0, *, no_assist=False, count=1, limp=False, **kwargs
):
    world = World(
        SCENE,
        count,
        realism_profile=profile,
        realism_seed=seed,
        no_assist=no_assist,
        **kwargs,
    )
    for index in range(count):
        body = Body(world, index, limp=limp)
        body.place(None, HOME_TRUNK_Z, index * 0.5)
        world.bodies.append(body)
    mujoco.mj_forward(world.model, world.data)
    return world


def drive(world, ticks=25):
    """A fixed script of writes, steps and reads; returns every answer."""
    body = world.bodies[0]
    handler = object.__new__(Handler)
    answers = []
    for tick in range(ticks):
        targets = list(HOME_POSE)
        targets[2] += 0.1 * math.sin(tick / 4)
        handler.dispatch(body, {"op": "write", "targets": targets})
        world.step(4)
        answers.append(handler.dispatch(body, {"op": "read"}))
    return answers


# ── off is the old simulator ─────────────────────────────────────────────────


def test_off_profile_leaves_model_and_sensors_untouched():
    world = make_world("off")
    plain = mujoco.MjModel.from_xml_path(str(SCENE))
    assert world.model.nmocap == plain.nmocap == 0
    assert world.model.nbody == plain.nbody
    np.testing.assert_array_equal(world.model.geom_friction, plain.geom_friction)
    np.testing.assert_array_equal(world.model.body_mass, plain.body_mass)
    body = world.bodies[0]
    assert body.sensor_model is None and body.link is None and body.tof_realism is None
    np.testing.assert_array_equal(
        body._gain, plain.actuator_gainprm[body.actuator_slice, 0]
    )

    world.step(10)
    answer = Handler.dispatch(object.__new__(Handler), body, {"op": "read"})
    data = world.data
    quat = data.qpos[body.trunk + 3 : body.trunk + 7]
    for slot, wire in enumerate(body.to_wire):
        assert answer["positions"][wire] == float(data.qpos[body.qpos_adr[slot]])
        assert answer["velocities"][wire] == float(data.qvel[body.qvel_adr[slot]])
    assert answer["imu"]["quat"] == [float(v) for v in quat]
    assert answer["imu"]["gravity"] == [float(v) for v in gravity_in_trunk(quat)]
    assert answer["imu"]["gyro"] == [
        float(v) for v in data.qvel[body.trunk_dof + 3 : body.trunk_dof + 6]
    ]
    assert "commanded_mode" not in answer
    assert set(answer) == {
        "positions",
        "velocities",
        "currents_ma",
        "trunk_z",
        "trunk",
        "sim_time",
        "mode",
        "supported",
        "imu",
    }

    # A write lands immediately, as it always did.
    targets = list(HOME_POSE)
    targets[2] = 0.1
    body.set_targets(targets)
    assert world.data.ctrl[body.actuators[body.to_wire.index(2)]] == 0.1


def test_off_tof_is_the_plain_sensor():
    world = make_world("off")
    body = world.bodies[0]
    plain = Tof(world.model, body.tof.site, seed=0)
    assert body.depth()["distance_mm"] == plain.frame(world.data)[0]


def test_off_trajectory_is_unchanged_by_the_new_code_paths():
    """Two off worlds, one built through the new keyword arguments: the same physics, bit for bit."""
    a = make_world("off")
    b = World(SCENE, 1)
    body = Body(b, 0)
    body.place(None, HOME_TRUNK_Z, 0.0)
    b.bodies.append(body)
    mujoco.mj_forward(b.model, b.data)
    assert drive(a) == drive(b)


# ── seeded and reproducible ──────────────────────────────────────────────────


@pytest.mark.parametrize("profile", ["real", "hard"])
def test_same_seed_same_run_and_another_seed_another(profile):
    first, second, other = (drive(make_world(profile, seed)) for seed in (7, 7, 8))
    assert first == second
    assert first != other


def test_model_randomization_is_seeded_and_in_range():
    plain = mujoco.MjModel.from_xml_path(str(SCENE))
    prof = realism.PROFILES["real"]
    worlds = [make_world("real", seed) for seed in (1, 1, 2)]
    np.testing.assert_array_equal(worlds[0].model.body_mass, worlds[1].model.body_mass)
    assert not np.array_equal(worlds[0].model.body_mass, worlds[2].model.body_mass)
    for world in worlds:
        ratio = world.model.body_mass[1:] / np.where(
            plain.body_mass[1:] > 0, plain.body_mass[1:], 1
        )
        moved = plain.body_mass[1:] > 0
        assert np.all(ratio[moved] >= prof.mass[0] - 1e-9) and np.all(
            ratio[moved] <= prof.mass[1] + 1e-9
        )
        floor = mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        scale = world.model.geom_friction[floor, 0] / plain.geom_friction[floor, 0]
        assert prof.friction[0] <= scale <= prof.friction[1] and scale != 1.0
        body = world.bodies[0]
        gains = body._gain / plain.actuator_gainprm[body.actuator_slice, 0]
        assert np.all((gains >= prof.kp[0]) & (gains <= prof.kp[1]))
        assert len(set(np.round(gains, 6))) > 1


def test_motor_strength_survives_an_equipment_change():
    world = make_world("real", 3)
    body = world.bodies[0]
    before = {JOINT_NAMES[w]: g for w, g in zip(body.to_wire, body._gain)}
    nominal = mujoco.MjModel.from_xml_path(str(SCENE)).actuator_gainprm[
        body.actuator_slice, 0
    ]
    world.set_mode(0, "roller")
    world.set_mode(0, "walk")
    after = {JOINT_NAMES[w]: g for w, g in zip(body.to_wire, body._gain)}
    assert before == pytest.approx(after)
    assert not np.allclose(list(after.values()), nominal)


# ── what the sensors report ──────────────────────────────────────────────────


def test_real_sensors_are_quantized_noisy_and_biased_within_the_training_envelope():
    world = make_world("real", 5)
    body = world.bodies[0]
    errors = []
    for _ in range(50):
        world.step(4)
        answer = body.sensors()
        truth = np.zeros(len(JOINT_NAMES))
        truth[body.to_wire] = world.data.qpos[body.qpos_adr]
        measured = np.array(answer["positions"])
        steps = measured[body.to_wire] / realism.ENCODER_STEP
        np.testing.assert_allclose(steps, np.round(steps), atol=1e-6)
        vel = np.array(answer["velocities"])[body.to_wire] / realism.VELOCITY_STEP
        np.testing.assert_allclose(vel, np.round(vel), atol=1e-6)
        errors.append(measured[body.to_wire] - truth[body.to_wire])
        quat = np.array(answer["imu"]["quat"])
        assert abs(np.linalg.norm(quat) - 1.0) < 1e-3
        assert answer["positions"][JOINT_NAMES.index("mouth")] == 0.0
    errors = np.array(errors)
    # Latency means some error is motion; the bias and quantum bound the rest.
    assert np.abs(errors).max() < 0.1
    assert np.abs(errors).max() > 0.0
    # A per-joint bias: the mean error is not zero but stays inside ENCODER_BIAS_RANGE + a quantum.
    assert np.abs(errors.mean(axis=0)).max() <= 0.015 + realism.ENCODER_STEP + 0.01


def test_gyro_bias_drifts_and_heading_drifts():
    prof = dataclasses.replace(
        realism.PROFILES["real"], gyro_noise=0.0, gravity_noise=0.0, yaw_drift=0.01
    )
    model = realism.SensorModel(prof, 1, 0, JOINT_NAMES)
    zeros = np.zeros(len(JOINT_NAMES))
    upright = np.array([1.0, 0.0, 0.0, 0.0])
    model.record(0.0, zeros, zeros, upright, np.zeros(3), np.array([0, 0, -1.0]))
    start = model.frames[-1][1]
    for step in range(1, 1001):
        model.record(
            step * 0.02, zeros, zeros, upright, np.zeros(3), np.array([0, 0, -1.0])
        )
    end = model.frames[-1][1]
    assert np.any(start["gyro"] != 0)  # power-up bias
    assert not np.allclose(start["gyro"], end["gyro"])  # random walk
    yaw = lambda q: math.atan2(
        2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)
    )
    assert abs(yaw(end["quat"]) - yaw(start["quat"])) == pytest.approx(
        abs(model.yaw_rate) * 20, abs=2e-3
    )


def test_command_latency_delays_targets_by_15_to_30_ms():
    world = make_world("real", 2)
    body = world.bodies[0]
    actuator = body.actuators[body.to_wire.index(2)]
    before = float(world.data.ctrl[actuator])
    targets = list(HOME_POSE)
    targets[2] = before + 0.2
    body.set_targets(targets)
    assert world.data.ctrl[actuator] == before  # in flight
    world.step(2)  # 10 ms
    assert world.data.ctrl[actuator] == before
    world.step(5)  # 35 ms
    assert world.data.ctrl[actuator] == pytest.approx(before + 0.2)


def test_torque_change_discards_targets_in_flight():
    world = make_world("real", 2)
    body = world.bodies[0]
    targets = list(HOME_POSE)
    targets[2] += 0.3
    body.set_targets(targets)
    body.set_torque(True)
    world.step(10)
    actuator = body.actuators[body.to_wire.index(2)]
    assert world.data.ctrl[actuator] != pytest.approx(targets[2])


def test_dropped_reads_repeat_the_previous_frame_and_dropped_writes_vanish():
    world = World(SCENE, 1, realism_profile="real", realism_seed=0)
    world.profile = dataclasses.replace(world.profile, read_drop=1.0, write_drop=1.0)
    body = Body(world, 0)
    body.place(None, HOME_TRUNK_Z, 0.0)
    world.bodies.append(body)
    mujoco.mj_forward(world.model, world.data)
    first = body.sensors()
    world.step(10)
    assert body.sensors() is first
    actuator = body.actuators[body.to_wire.index(2)]
    before = float(world.data.ctrl[actuator])
    targets = list(HOME_POSE)
    targets[2] = before + 0.3
    body.set_targets(targets)
    world.step(10)
    assert world.data.ctrl[actuator] == before


def test_sensor_latency_returns_an_older_frame():
    prof = dataclasses.replace(realism.PROFILES["real"], sensor_latency_s=(0.02, 0.02))
    model = realism.SensorModel(prof, 0, 0, JOINT_NAMES)
    zeros = np.zeros(len(JOINT_NAMES))
    for step in range(10):
        model.record(
            step * 0.005,
            zeros + step,
            zeros,
            np.array([1.0, 0, 0, 0]),
            np.zeros(3),
            np.array([0, 0, -1.0]),
        )
    frame = model.frame_at(0.045 - 0.02)
    assert np.median(frame["positions"]) == pytest.approx(5, abs=0.05)


def test_hard_tof_has_failed_zones_and_off_has_none():
    hard, off = make_world("hard", 1), make_world("off")
    statuses = [s for _ in range(20) for s in hard.bodies[0].depth()["status"]]
    assert realism.STATUS_CONSISTENCY_FAILED in statuses
    assert realism.STATUS_CONSISTENCY_FAILED not in off.bodies[0].depth()["status"]


# ── no assist ────────────────────────────────────────────────────────────────


def test_no_assist_mode_request_neither_swaps_nor_supports():
    world = make_world(no_assist=True)
    body = world.bodies[0]
    body.set_torque(True)
    handler = object.__new__(Handler)
    assert (
        handler.dispatch(body, {"op": "mode", "mode": "roller", "support": True}) == {}
    )
    assert world.modes[0] == "walk"
    assert body.support_pose is None
    answer = handler.dispatch(body, {"op": "read"})
    assert answer["mode"] == "walk" and answer["commanded_mode"] == "roller"
    assert answer["supported"] is False
    # The human fits the skates; nothing steadies the trunk while they do.
    assert handler.dispatch(body, {"op": "equipment.fit", "equipment": "roller"}) == {
        "mode": "roller"
    }
    assert world.modes[0] == "roller" and body.support_pose is None
    world.step(20)
    assert body.support_pose is None
    with pytest.raises(ValueError):
        handler.dispatch(body, {"op": "mode", "mode": "skis"})


def test_no_assist_never_applies_trunk_support():
    world = make_world(no_assist=True)
    body = world.bodies[0]
    body.set_torque(True)
    body.set_mode_support(True)
    assert body.support_pose is None
    world.step(5)
    assert body.support_pose is None


def test_assisted_mode_still_supports_and_swaps():
    world = make_world()
    body = world.bodies[0]
    body.set_torque(True)
    handler = object.__new__(Handler)
    handler.dispatch(body, {"op": "mode", "mode": "roller", "support": True})
    assert world.modes[0] == "roller" and body.support_pose is not None


def test_no_assist_does_not_hold_a_duck_before_it_is_enabled():
    held, free = make_world(), make_world(no_assist=True)
    for world in (held, free):
        world.data.qpos[world.bodies[0].trunk + 2] = 0.25  # in the air
        world.bodies[0].remember()
        world.step(40)
    assert held.data.qpos[held.bodies[0].trunk + 2] == pytest.approx(0.25)
    assert free.data.qpos[free.bodies[0].trunk + 2] < 0.2


def test_equipment_fit_works_with_assist_too():
    world = make_world()
    body = world.bodies[0]
    handler = object.__new__(Handler)
    assert (
        handler.dispatch(body, {"op": "equipment.fit", "equipment": "roller"})["mode"]
        == "roller"
    )
    assert body.support_pose is None


# ── the oracle ───────────────────────────────────────────────────────────────


def test_oracle_pose_is_the_truth_and_says_so():
    world = make_world("hard", 4)
    body = world.bodies[0]
    world.data.qpos[body.trunk : body.trunk + 2] = [1.5, -0.5]
    yaw = 0.7
    world.data.qpos[body.trunk + 3 : body.trunk + 7] = [
        math.cos(yaw / 2),
        0,
        0,
        math.sin(yaw / 2),
    ]
    pose = Handler.dispatch(object.__new__(Handler), body, {"op": "oracle.pose"})
    assert pose["oracle"] is True
    assert (pose["x"], pose["y"]) == (1.5, -0.5)
    assert pose["yaw"] == pytest.approx(yaw)
    info = Handler.dispatch(object.__new__(Handler), body, {"op": "sim.info"})
    assert (
        info["realism"] == "hard"
        and info["realism_seed"] == 4
        and info["no_assist"] is False
    )


def test_unknown_profile_is_refused():
    with pytest.raises(ValueError, match="realism must be one of"):
        World(SCENE, 1, realism_profile="extreme")
