import math
from dataclasses import replace
import mujoco
import numpy as np
from drone_sim.config import Config
from drone_sim.simulation import Simulation


def test_no_thrust_means_gravity_not_scripted_trajectory():
    with Simulation(vision_mode='pose') as s:
        s.motors_on = False
        s.motor_force = 0
        initial = s.position[2]
        for _ in range(50):
            s._autopilot()
            mujoco.mj_step(s.model,s.data)
            mujoco.mj_forward(s.model,s.data)
        t = 50*s.cfg.physics_dt
        assert abs(s.position[2]-(initial-.5*9.81*t*t)) < .012
        assert s.data.qvel[2] < -2.3


def test_motor_shutdown_has_exponential_lag():
    with Simulation(vision_mode='pose') as s:
        start = s.motor_force
        s.motors_on = False
        s._autopilot()
        assert 0 < s.motor_force < start
        assert abs(s.motor_force-start*math.exp(-s.cfg.physics_dt/s.cfg.motor_tau)) < 1e-12


def test_all_feet_must_overlap_electrodes():
    # The centre-radius criterion alone must not claim a valid electrical dock.
    with Simulation(vision_mode='pose') as s:
        s.data.qpos[:3] = [.04,0,s.cfg.pad_top-s.cfg.feet_z-.0002]
        s.data.qpos[3:7] = [1,0,0,0]
        s.motors_on = False
        s.motor_force = 0
        s.mode = 'ground'
        for _ in range(300):
            s._autopilot()
            vz = s.data.qvel[2]
            mujoco.mj_step(s.model,s.data)
            mujoco.mj_forward(s.model,s.data)
            s._update_contact_and_battery(vz)
        assert s.landed
        assert np.linalg.norm(s.position[:2]) < s.cfg.dock_radius
        assert not s.dock_contact
        assert not s.charging
