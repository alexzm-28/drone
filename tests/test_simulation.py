from dataclasses import replace
import json
from pathlib import Path
import math
import cv2
import mujoco
import numpy as np
import pytest
from drone_sim.config import Config
from drone_sim.control import Stage
from drone_sim.simulation import Simulation


def run_until_terminal(sim, duration=80):
    while sim.now < duration:
        sim.step()
        if sim.controller.stage == Stage.COMPLETE:
            break
        if sim.controller.stage == Stage.ABORT and sim.landed:
            break
    return sim.result()


@pytest.mark.parametrize('seed', [1, 7, 42, 123])
def test_complete_cycle_in_pose_mode(seed):
    with Simulation(replace(Config(), seed=seed), 'pose') as sim:
        result = run_until_terminal(sim)
        assert result['success'], result
        assert result['landing']['vertical_speed'] < .2
        stages = [event['to'] for event in result['events']]
        assert all(stage in stages for stage in ['ALIGN', 'DESCEND', 'LAND', 'DOCK', 'CHARGE', 'RELEASE', 'REFLY', 'COMPLETE'])
        assert any(row['charging'] and row['landed'] for row in sim.history)
        assert not any(row['charging'] and not row['landed'] for row in sim.history)


def test_reproducible_without_wall_clock():
    with Simulation(vision_mode='pose') as a, Simulation(vision_mode='pose') as b:
        for _ in range(120):
            a.step()
            b.step()
        np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
        assert a.history == b.history


@pytest.mark.parametrize('position,yaw', [((.3,-.18,1.2),.25), ((-.2,.1,.8),-.6), ((0,0,.4),.8)])
def test_rendered_camera_pose_accuracy(position, yaw):
    with Simulation() as sim:
        sim.data.qpos[:3] = position
        sim.data.qpos[3:7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
        mujoco.mj_forward(sim.model, sim.data)
        pose = sim.vision.estimate(sim.render_camera(), sim.now)
        assert pose is not None
        assert np.linalg.norm(np.array([pose.x,pose.y,pose.z])-position) < .035
        assert abs(pose.yaw-yaw) < .06


def test_occlusion_and_outside_fov_cannot_leak_truth():
    with Simulation() as sim:
        assert sim.vision.estimate(np.full_like(sim.rgb, 32), 0) is None
        sim.data.qpos[0] = 3
        mujoco.mj_forward(sim.model, sim.data)
        assert sim.vision.estimate(sim.render_camera(), 0) is None


def test_complete_camera_cycle():
    with Simulation(vision_mode='camera') as sim:
        result = run_until_terminal(sim)
        assert result['success'], result
        assert sim.vision_mode == 'camera'
        assert result['landing']['xy_error'] < .045


def test_persistent_occlusion_aborts_and_lands():
    with Simulation(vision_mode='camera') as sim:
        while sim.now < 25:
            sim.force_vision_loss = sim.now >= 3
            sim.step()
            if sim.controller.stage == Stage.ABORT and sim.landed:
                break
        assert sim.controller.stage == Stage.ABORT
        assert sim.controller.reason == 'vision_timeout'
        assert sim.landed
        assert not sim.result()['success']


def test_battery_charge_clock_does_not_speed_up_flight_drain():
    cfg = replace(Config(), charge_time_scale=120)
    with Simulation(cfg, 'pose') as sim:
        for _ in range(25):
            sim.step()
        expected = cfg.initial_soc - cfg.flight_power_w / (3600*cfg.battery_capacity_wh)
        assert abs(sim.soc-expected) < 1e-10


def test_no_charging_without_dock_or_with_motors_running():
    with Simulation(vision_mode='pose') as sim:
        for _ in range(40):
            sim.step()
            assert sim.motors_on and not sim.charging and not sim.dock_contact


def test_dashboard_is_renderable():
    from drone_sim.app import draw_dashboard
    with Simulation() as sim:
        sim.step()
        image = draw_dashboard(sim)
        assert image.shape == (940,1500,3)
        assert np.std(image) > 10


@pytest.mark.parametrize('case',json.loads((Path(__file__).parent/'regression_cases.json').read_text()))
def test_wind_and_electrode_overlap_regressions(case):
    with Simulation(replace(Config(),**case),'pose') as sim:
        result = run_until_terminal(sim)
        assert result['success'],result
