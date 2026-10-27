"""MuJoCo dynamics, hidden onboard autopilot and independently simulated sensors.

The onboard velocity/attitude loops are a surrogate, not reverse-engineered Tello
firmware. The outer landing controller only gets Pose and Telemetry objects.
"""
from collections import deque
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import mujoco
import numpy as np
from .config import Config
from .control import Command, LandingController, Pose, Telemetry, Stage
from .scene import scene_xml
from .vision import MarkerVision


class Simulation:
    def __init__(self, cfg=None, vision_mode='camera'):
        self.cfg = cfg or Config()
        if vision_mode not in ('camera', 'pose'):
            raise ValueError('vision_mode must be camera or pose')
        self.vision_mode = vision_mode
        self.model = mujoco.MjModel.from_xml_string(scene_xml(self.cfg))
        self.data = mujoco.MjData(self.model)
        self.body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'drone')
        self.pad = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'pad')
        self.vision = MarkerVision(self.cfg)
        self.renderer = None
        self.overview_renderer = None
        self.view_camera = mujoco.MjvCamera()
        self.view_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.view_camera.lookat[:] = [0, 0, 0.5]
        self.view_camera.distance = 1.8
        self.view_camera.azimuth = -55
        self.view_camera.elevation = -22
        self.follow_view = False
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.controller = LandingController(self.cfg)
        self.command_queue = deque()
        self.pose_queue = deque()
        self.pose = None
        self.command = Command()
        self.applied_command = Command()
        self.rgb = np.zeros((self.cfg.camera_height, self.cfg.camera_width, 3), np.uint8)
        self.last_detection_ok = False
        self.force_vision_loss = False
        self.wind_enabled = True
        self.wind_force = np.zeros(3)
        self.mode = 'fly'
        self.motor_force = self.cfg.mass * 9.81
        self.velocity_integral = np.zeros(3)
        self.motors_on = True
        self.landed = False
        self.contact_since = None
        self.dock_since = None
        self.dock_contact = False
        self.charging = False
        self.released = False
        self.release_at = None
        self.soc = self.cfg.initial_soc
        self.takeoff_anchor = np.zeros(2)
        self.land_anchor = np.zeros(2)
        self.next_camera_time = 0.0
        self.history = []
        self.touchdowns = []
        self.last_contact = False
        self.land_requested_at = None
        self.path = deque(maxlen=600)

    @property
    def now(self):
        return float(self.data.time)

    @property
    def position(self):
        return self.data.xpos[self.body].copy()

    @property
    def rotation(self):
        return self.data.xmat[self.body].reshape(3, 3).copy()

    @property
    def yaw(self):
        r = self.rotation
        return math.atan2(r[1, 0], r[0, 0])

    def _contact(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            a, b = int(contact.geom1), int(contact.geom2)
            if (a == self.pad and self.model.geom_bodyid[b] == self.body or
                    b == self.pad and self.model.geom_bodyid[a] == self.body):
                if contact.dist < 0.001:
                    return True
        return False

    def _any_floor_contact(self):
        for i in range(self.data.ncon):
            ct = self.data.contact[i]
            a, b = int(ct.geom1), int(ct.geom2)
            if ((self.model.geom_bodyid[a] == self.body and self.model.geom_bodyid[b] == 0)
                    or (self.model.geom_bodyid[b] == self.body and self.model.geom_bodyid[a] == 0)):
                if ct.dist < 0.001:
                    return True
        return False

    def telemetry(self):
        return Telemetry(self.landed, self.dock_contact, self.charging, self.soc,
                         float(self.position[2]), self.released)

    def render_camera(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.cfg.camera_height,
                                            width=self.cfg.camera_width)
        self.renderer.update_scene(self.data, camera='ground_camera')
        return self.renderer.render().copy()

    def render_overview(self):
        if self.overview_renderer is None:
            self.overview_renderer = mujoco.Renderer(self.model, height=600, width=900)
        if self.follow_view:
            self.view_camera.lookat[:] = self.position
        self.overview_renderer.update_scene(self.data, camera=self.view_camera)
        scene = self.overview_renderer.scene
        # Visual-only 3D trajectory. It never participates in collisions or sensing.
        points = list(self.path)[::6]
        for a, b in zip(points, points[1:]):
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                               np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                               np.array([0.2,0.75,0.9,0.65],dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .0018, a, b)
            scene.ngeom += 1
        return self.overview_renderer.render().copy()

    def _observe(self):
        c, now = self.cfg, self.now
        if now + 1e-8 >= self.next_camera_time:
            self.next_camera_time += 1 / c.camera_hz
            if self.vision_mode == 'camera':
                self.rgb = self.render_camera()
                if self.force_vision_loss:
                    self.rgb[:] = 30  # Actual blank image sent to detector.
                observation = self.vision.estimate(self.rgb, now)
            else:
                # Fast control tests: geometric FOV + synthetic pose, explicitly labelled.
                p = self.position
                d = p[2] + c.marker_z - c.camera_z
                half_y = d * math.tan(math.radians(c.camera_fovy) / 2)
                half_x = half_y * c.camera_width / c.camera_height
                visible = (d > 0.09 and abs(p[0]) + c.marker_size * 0.7 < half_x
                           and abs(p[1]) + c.marker_size * 0.7 < half_y)
                observation = None
                if visible and not self.force_vision_loss:
                    noisy = p + self.rng.normal(0, c.pose_noise, 3)
                    observation = Pose(*noisy, self.yaw + self.rng.normal(0, 0.006), now)
            if self.rng.random() < c.vision_dropout:
                observation = None
            self.last_detection_ok = observation is not None
            self.pose_queue.append((now + c.camera_latency, observation))
        while self.pose_queue and self.pose_queue[0][0] <= now + 1e-8:
            _, self.pose = self.pose_queue.popleft()

    def _execute_action(self, action):
        if action == 'land' and self.mode not in ('land', 'ground'):
            self.mode = 'land'
            self.land_anchor = self.position[:2]
            self.land_requested_at = self.now
        elif action == 'release' and self.landed and self.dock_contact:
            self.release_at = self.now + 0.5
        elif action == 'takeoff' and self.landed and self.released:
            self.motors_on = True
            self.landed = False
            self.contact_since = None
            self.mode = 'takeoff'
            self.takeoff_anchor = self.position[:2]

    def _autopilot(self):
        c, p, r = self.cfg, self.position, self.rotation
        v = self.data.qvel[:3].copy()
        command = self.applied_command
        desired_local = np.clip([command.right, command.forward, command.up],
                                -c.max_speed, c.max_speed)
        cs, sn = math.cos(self.yaw), math.sin(self.yaw)
        desired_velocity = np.array([cs*desired_local[0]-sn*desired_local[1],
                                     sn*desired_local[0]+cs*desired_local[1], desired_local[2]])
        yaw_rate = float(np.clip(command.yaw_ccw, -c.max_yaw_rate, c.max_yaw_rate))
        if self.mode == 'land':
            desired_velocity[:] = (0, 0, -0.10)
            # Ideal onboard horizontal position sensor; separate from the ground
            # camera. Represents the inner autopilot's position-hold capability.
            desired_velocity[:2] = np.clip(1.8*(self.land_anchor-p[:2]),-.15,.15)
            yaw_rate = 0.0
        elif self.mode == 'takeoff':
            desired_velocity[:2] = np.clip(1.5*(self.takeoff_anchor-p[:2]), -0.2, 0.2)
            desired_velocity[2] = np.clip(1.6*(c.reflight_height-p[2]), 0, 0.35)
            yaw_rate = 0.0
        elif self.mode == 'ground':
            desired_velocity[:] = 0
            yaw_rate = 0.0
        self.velocity_integral = np.clip(self.velocity_integral + (desired_velocity-v)*c.physics_dt, -0.3, 0.3)
        acceleration = np.clip((desired_velocity-v)*np.array([c.velocity_kp_xy,c.velocity_kp_xy,c.velocity_kp_z])
                               + self.velocity_integral*np.array([c.velocity_ki_xy,c.velocity_ki_xy,c.velocity_ki_z]),-2.5,2.5)
        max_horizontal = 9.81 * math.tan(c.max_tilt)
        length = np.linalg.norm(acceleration[:2])
        if length > max_horizontal:
            acceleration[:2] *= max_horizontal / length
        force_vector = c.mass * (acceleration + np.array([0, 0, 9.81]))
        desired_z = force_vector / np.linalg.norm(force_vector)
        desired_x_heading = np.array([cs, sn, 0.0])
        desired_y = np.cross(desired_z, desired_x_heading)
        desired_y /= np.linalg.norm(desired_y)
        desired_x = np.cross(desired_y, desired_z)
        desired_r = np.column_stack([desired_x, desired_y, desired_z])
        attitude_error = 0.5 * sum((np.cross(r[:, i], desired_r[:, i]) for i in range(3)), np.zeros(3))
        desired_omega_world = np.array([0.0, 0.0, yaw_rate])
        omega_body = self.data.qvel[3:6].copy()
        angular_accel_body = r.T @ (100*attitude_error + 16*desired_omega_world) - 16*omega_body
        torque = r @ (self.model.body_inertia[self.body] * angular_accel_body)
        target_thrust = float(np.clip(np.dot(force_vector, r[:, 2]), 0, c.mass * 16)) if self.motors_on else 0
        self.motor_force += (1-math.exp(-c.physics_dt/c.motor_tau))*(target_thrust-self.motor_force)
        surface_z = c.pad_top if np.linalg.norm(p[:2]) < .24 else 0.0
        height = max(0, p[2] + c.feet_z - surface_z)
        # Explicit simplified ground-effect term, compensated by the velocity loop.
        ge = 1 + 0.06 * math.exp(-height/0.045) if self.motors_on else 1
        wind = np.zeros(3)
        if self.wind_enabled and self.motors_on:
            wind = c.wind_accel*np.array([math.sin(self.now*1.7), math.cos(self.now*1.3), 0])
        self.wind_force = c.mass * wind
        self.data.xfrc_applied[:] = 0
        self.data.xfrc_applied[self.body, :3] = r[:, 2]*self.motor_force*ge + self.wind_force - c.mass*0.10*v
        self.data.xfrc_applied[self.body, 3:] = torque if self.motors_on else -0.0005*(r@omega_body)

    def _update_contact_and_battery(self, pre_step_vz):
        c, now, p = self.cfg, self.now, self.position
        contact = self._any_floor_contact()
        if contact and not self.last_contact and self.motors_on:
            self.touchdowns.append({'time': now, 'x': float(p[0]), 'y': float(p[1]),
                                    'xy_error': float(np.linalg.norm(p[:2])),
                                    'vertical_speed': abs(float(pre_step_vz)), 'on_pad': self._contact()})
        self.last_contact = contact
        if contact and self.mode == 'land':
            if self.contact_since is None:
                self.contact_since = now
            if now-self.contact_since >= 0.12:
                self.motors_on = False
                self.mode = 'ground'
        elif self.mode != 'ground':
            self.contact_since = None
        self.landed = (contact and not self.motors_on and np.linalg.norm(self.data.qvel[:3]) < 0.04)
        electrode_offsets = []
        for x in (-0.065,0.065):
            for y in (-0.065,0.065):
                foot = p + self.rotation @ np.array([x,y,c.feet_z+0.006])
                electrode_offsets.append(np.linalg.norm(foot[:2]-[x,y]))
        dock_ok = (self.landed and self._contact() and np.linalg.norm(p[:2]) <= c.dock_radius
                   and max(electrode_offsets) <= c.electrode_radius
                   and abs(self.yaw) < c.dock_yaw_tolerance
                   and self.rotation[2, 2] > math.cos(math.radians(6)))
        if dock_ok and not self.released:
            if self.dock_since is None:
                self.dock_since = now
            self.dock_contact = now-self.dock_since >= c.dock_dwell
        else:
            self.dock_since = None
            self.dock_contact = False
        if self.release_at is not None and now >= self.release_at:
            self.released = True
            self.dock_contact = False
        self.charging = self.dock_contact and not self.released and not self.motors_on
        # Only battery time is accelerated; dynamics and simulation clock stay physical.
        power = c.charge_power_w if self.charging else -c.flight_power_w if self.motors_on else -0.1
        scale = c.charge_time_scale if self.charging else 1.0
        self.soc = float(np.clip(self.soc + power*c.physics_dt*scale/(3600*c.battery_capacity_wh), 0, 1))

    def step(self):
        c = self.cfg
        self._observe()
        self.command = self.controller.update(self.now, self.pose, self.telemetry())
        self.command_queue.append((self.now+c.command_latency, self.command))
        for _ in range(round(c.control_dt/c.physics_dt)):
            while self.command_queue and self.command_queue[0][0] <= self.now + 1e-8:
                _, self.applied_command = self.command_queue.popleft()
                self._execute_action(self.applied_command.action)
            self._autopilot()
            pre_step_vz = self.data.qvel[2]
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self._update_contact_and_battery(pre_step_vz)
        p = self.position
        pose = self.pose
        self.history.append({'time': self.now, 'stage': self.controller.stage.value,
                             'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2]),
                             'yaw': self.yaw, 'vx': float(self.data.qvel[0]),
                             'vy': float(self.data.qvel[1]), 'vz': float(self.data.qvel[2]),
                             'meas_x': pose.x if pose else None, 'meas_y': pose.y if pose else None,
                             'meas_z': pose.z if pose else None, 'meas_time': pose.captured_at if pose else None,
                             'vision_ok': self.last_detection_ok,
                             'right': self.command.right, 'forward': self.command.forward,
                             'up': self.command.up, 'soc': self.soc, 'landed': self.landed,
                             'dock_contact': self.dock_contact, 'charging': self.charging})
        self.path.append(p)

    def result(self):
        landing = self.touchdowns[0] if self.touchdowns else None
        success = (self.controller.stage == Stage.COMPLETE and landing is not None
                   and landing['on_pad'] and landing['xy_error'] <= self.cfg.dock_radius
                   and landing['vertical_speed'] <= 0.20)
        return {'success': bool(success), 'stage': self.controller.stage.value,
                'reason': self.controller.reason, 'simulation_seconds': self.now,
                'vision_mode': self.vision_mode, 'seed': self.cfg.seed,
                'landing': landing, 'soc': self.soc, 'events': self.controller.events,
                'config': self.cfg.to_dict()}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'result.json').write_text(json.dumps(self.result(), indent=2), encoding='utf-8')
        if self.history:
            with (directory/'telemetry.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.history[0])
                writer.writeheader()
                writer.writerows(self.history)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.overview_renderer is not None:
            self.overview_renderer.close()
            self.overview_renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
