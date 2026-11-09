"""Obstacles placed at launch and moved at runtime (`sim/obstacles.py`)."""

import math

import mujoco
import numpy as np
import pytest

from mjlab_microduck.sim import obstacles
from mjlab_microduck.sim.body_server import HOME_TRUNK_Z, SCENES, Body, Handler, World
from mjlab_microduck.sim.tof import STATUS_VALID
from mjlab_microduck.sim.visual import VisualWorld

SCENE = SCENES / "scene_backlash.xml"
WALL = {"name": "wall", "shape": "box", "size": [0.1, 1.5, 0.6], "pose": {"x": 0.8, "y": 0.0}}


def make_world(slots=4, *, profile="off", **kwargs):
    world = World(SCENE, 1, obstacle_slots=slots, realism_profile=profile, **kwargs)
    body = Body(world, 0)
    body.place(None, HOME_TRUNK_Z, 0.0)
    world.bodies.append(body)
    mujoco.mj_forward(world.model, world.data)
    return world


def valid_ranges(world, zones=slice(None)):
    frame = world.bodies[0].depth()
    pairs = list(zip(frame["distance_mm"], frame["status"]))[zones]
    return [d / 1000 for d, s in pairs if s == STATUS_VALID]


# The three lower rows of the 8x8 grid see the floor (~0.74-1.7 m); the upper five see
# nothing in the empty scene, so any return there is an obstacle.
ABOVE_FLOOR = slice(0, 40)


def wall_ranges(world):
    return valid_ranges(world, ABOVE_FLOOR)


def tof_x(world):
    return float(world.data.site_xpos[world.bodies[0].tof.site][0])


def test_no_slots_by_default_and_obstacle_ops_say_how_to_get_them():
    world = World(SCENE, 1)
    assert world.model.nmocap == 0 and world.obstacles.capacity == 0
    body = Body(world, 0)
    world.bodies.append(body)
    with pytest.raises(ValueError, match="--obstacle-slots"):
        Handler.dispatch(object.__new__(Handler), body, {"op": "obstacle.add", "obstacle": WALL})


def test_slots_are_parked_and_change_nothing_until_used():
    world = make_world()
    assert world.model.nmocap == 4
    before = valid_ranges(world)
    plain = make_world(0)
    assert before == valid_ranges(plain)
    world.step(20)
    plain.step(20)
    np.testing.assert_allclose(world.data.qpos[:7], plain.data.qpos[:7], atol=1e-9)


def test_tof_rays_hit_an_added_obstacle_and_stop_after_it_is_removed():
    world = make_world()
    assert wall_ranges(world) == []
    handler = object.__new__(Handler)
    body = world.bodies[0]
    answer = handler.dispatch(body, {"op": "obstacle.add", "obstacle": WALL})
    assert answer["obstacle"]["name"] == "wall"
    expected = 0.8 - 0.05 - tof_x(world)
    hits = [r for r in wall_ranges(world) if r < 1.0]
    assert len(hits) >= 8
    assert min(hits) == pytest.approx(expected, abs=0.03)

    handler.dispatch(body, {"op": "obstacle.move", "name": "wall", "pose": {"x": 1.2}})
    hits = [r for r in wall_ranges(world) if r < 1.5]
    assert min(hits) == pytest.approx(1.2 - 0.05 - tof_x(world), abs=0.03)

    listed = handler.dispatch(body, {"op": "obstacle.list"})
    assert listed["slots"] == 4
    assert listed["obstacles"][0]["pose"]["x"] == 1.2

    assert handler.dispatch(body, {"op": "obstacle.remove", "name": "wall"}) == {}
    assert not any(r < 1.5 for r in wall_ranges(world))
    assert handler.dispatch(body, {"op": "obstacle.list"})["obstacles"] == []


def test_obstacles_collide_with_the_duck():
    world = make_world()
    body = world.bodies[0]
    body.set_torque(True)  # released, so nothing holds it in place
    trunk = world.data.qpos[body.trunk:body.trunk + 3].copy()
    world.add_obstacle({"name": "ram", "shape": "box", "size": [0.3, 0.6, 0.4],
                        "pose": {"x": trunk[0] - 0.3, "y": trunk[1]}})
    mujoco.mj_forward(world.model, world.data)
    first = world.model.body_geomadr[mujoco.mj_name2id(
        world.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_0")]
    duck = world.model.body_rootid[mujoco.mj_name2id(
        world.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")]
    # Push the ram through where the duck stands: it must shove the duck, not pass through it.
    for step in range(50):
        world.move_obstacle("ram", {"x": trunk[0] - 0.3 + 0.004 * step})
        world.step(1)
    contacts = {
        (int(c.geom1), int(c.geom2)) for c in world.data.contact[:world.data.ncon]
    }
    shoved = world.data.qpos[body.trunk] - trunk[0]
    assert shoved > 0.02, f"the duck moved {shoved:.3f} m"
    ram = world.model.geom_bodyid[first]
    assert world.model.body_contype[ram] != 0
    touching = [
        (a, b) for a, b in contacts
        if {world.model.geom_bodyid[a], world.model.geom_bodyid[b]} & {ram}
        and duck in {world.model.body_rootid[world.model.geom_bodyid[a]],
                     world.model.body_rootid[world.model.geom_bodyid[b]]}
    ]
    assert touching


def test_obstacles_are_rendered_by_the_visual_world():
    world = make_world()
    visual = VisualWorld(world)
    visual.update()
    geom = mujoco.mj_name2id(visual.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_0_0")
    assert visual.model.geom_rgba[geom, 3] == 0
    world.add_obstacle(WALL)
    visual.update()
    assert visual.model.geom_rgba[geom, 3] == 1.0
    assert visual.model.geom_group[geom] == 0  # drawn by default, unlike collision group 3
    mocap = visual.model.body_mocapid[visual.model.geom_bodyid[geom]]
    np.testing.assert_allclose(visual.data.mocap_pos[mocap][:2], [0.8, 0.0])
    try:
        renderer = mujoco.Renderer(visual.model, 64, 64)
    except Exception as error:  # no GL context on this machine
        pytest.skip(f"no renderer: {error}")
    with renderer:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.8, 0.0, 0.3]
        camera.distance, camera.azimuth, camera.elevation = 1.5, 180.0, -10.0
        renderer.enable_segmentation_rendering()
        renderer.update_scene(visual.data, camera)
        segmentation = renderer.render()
    seen = (segmentation[..., 0] == geom) & (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
    assert seen.any()


def test_obstacles_survive_an_equipment_change():
    world = make_world()
    world.add_obstacle(WALL)
    world.set_mode(0, "roller")
    geom = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_0_0")
    assert world.model.geom_contype[geom] != 0
    assert any(r < 1.0 for r in wall_ranges(world))


def test_shapes_and_validation():
    chair = obstacles.parse({"name": "c", "shape": "chair"})
    assert len(chair.parts) == 6
    door = obstacles.parse({"name": "d", "shape": "door", "size": [0.8, 0.04, 2.0]})
    assert door.parts[0].size == pytest.approx((0.02, 0.4, 1.0))
    for bad in (
        {"shape": "box", "size": [1, 1, 1]},
        {"name": "x", "shape": "sofa"},
        {"name": "x", "shape": "box", "size": [1, 1]},
        {"name": "x", "shape": "box", "size": [5, 1, 1]},
        {"name": "x", "shape": "box", "size": [1, 1, 1], "rgba": [1, 1, 1, 0]},
    ):
        with pytest.raises(ValueError):
            obstacles.parse(bad)
    world = make_world(slots=1)
    world.add_obstacle(WALL)
    with pytest.raises(ValueError, match="slots are in use"):
        world.add_obstacle({**WALL, "name": "other"})
    world.remove_obstacle("wall")
    world.add_obstacle(WALL)
    with pytest.raises(ValueError, match="already exists"):
        world.obstacles.add(WALL)


def test_multi_part_shapes_keep_their_parts_apart():
    world = make_world()
    world.add_obstacle({"name": "c", "shape": "chair", "pose": {"x": 1.0}})
    mujoco.mj_forward(world.model, world.data)
    ids = [mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_0_{j}") for j in range(6)]
    legs = world.data.geom_xpos[ids[:4]]
    assert {tuple(np.round(leg[:2], 3)) for leg in legs} == {
    (round(1.0 + sx * 0.195, 3), sy * 0.195) for sx in (-1, 1) for sy in (-1, 1)
    }
    assert np.allclose(legs[:, 2], 0.21)
    assert world.data.geom_xpos[ids[4]][2] == pytest.approx(0.435)
    # A ray under the seat between the legs passes; one at seat height stops on the chair.
    geom = np.zeros(1, np.int32)
    low = mujoco.mj_ray(world.model, world.data, np.array([0.3, 0, 0.2]), np.array([1.0, 0, 0]),
                    None, 1, -1, geom)
    assert low < 0 or geom[0] not in ids
    seat = mujoco.mj_ray(world.model, world.data, np.array([0.3, 0, 0.44]), np.array([1.0, 0, 0]),
                     None, 1, -1, geom)
    assert seat == pytest.approx(0.49, abs=1e-6) and geom[0] == ids[4]


def test_moving_obstacle_paces_its_beat_deterministically():
    pet = {"name": "cat", "shape": "cylinder", "size": [0.2, 0.2],
           "motion": {"waypoints": [[1.0, 0.0], [2.0, 0.0]], "speed": 0.5}}
    world = make_world()
    world.add_obstacle(pet)
    mocap = 0
    world.step(200)  # 1 s: half way
    assert world.data.mocap_pos[mocap][0] == pytest.approx(1.5, abs=0.02)
    world.step(400)  # 3 s: on the way back
    assert world.data.mocap_pos[mocap][0] == pytest.approx(1.5, abs=0.02)
    assert abs(world.obstacles.slots[0].yaw) == pytest.approx(math.pi)


def test_glass_is_seen_through_or_ghosts_under_realism_and_is_solid_when_off():
    pane = {"name": "glass", "shape": "box", "size": [0.02, 1.5, 0.6],
            "rgba": [0.8, 0.9, 1.0, 0.3], "pose": {"x": 0.6}}
    off = make_world()
    off.add_obstacle(pane)
    off.add_obstacle({**WALL, "pose": {"x": 1.4}})
    assert min(wall_ranges(off)) == pytest.approx(0.59 - tof_x(off), abs=0.03)
    real = make_world(profile="real", realism_seed=3)
    real.add_obstacle(pane)
    real.add_obstacle({**WALL, "pose": {"x": 1.4}})
    ranges = [r for _ in range(5) for r in wall_ranges(real) if r < 2.0]
    beyond = [r for r in ranges if r > 1.2]
    assert beyond, "realism should see through glass at least sometimes"


def test_dark_obstacles_vanish_beyond_the_dark_range():
    dark = {**WALL, "rgba": [0.05, 0.05, 0.05, 1.0], "pose": {"x": 2.3}}
    off, real = make_world(), make_world(profile="real")
    for world in (off, real):
        world.add_obstacle(dark)
    assert any(1.8 < r < 2.4 for r in wall_ranges(off))
    assert not any(1.8 < r < 2.4 for r in wall_ranges(real))


