"""Shared outer loop: observations in metres/radians, body velocity commands.

World X=right, Y=forward, Z=up; yaw is counterclockwise about +Z.
No access to MuJoCo state or ground truth is allowed in this module.
"""
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
from .config import Config


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Stage(str, Enum):
    ACQUIRE = 'ACQUIRE'
    ALIGN = 'ALIGN'
    DESCEND = 'DESCEND'
    LAND = 'LAND'
    DOCK = 'DOCK'
    CHARGE = 'CHARGE'
    RELEASE = 'RELEASE'
    REFLY = 'REFLY'
    COMPLETE = 'COMPLETE'
    ABORT = 'ABORT'


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    yaw: float
    captured_at: float
    reprojection_error: float = 0.0

    def valid(self, now, max_age):
        return (all(math.isfinite(v) for v in (self.x, self.y, self.z, self.yaw, self.captured_at))
                and 0 <= now - self.captured_at <= max_age + 1e-8 and self.z > 0)


@dataclass(frozen=True)
class Telemetry:
    landed: bool = False
    dock_contact: bool = False
    charging: bool = False
    soc: float = 0.6
    height: float = 1.2
    released: bool = False


@dataclass(frozen=True)
class Command:
    right: float = 0.0
    forward: float = 0.0
    up: float = 0.0
    yaw_ccw: float = 0.0
    action: str = ''


class PID:
    def __init__(self, kp, ki, kd, limit=0.35):
        self.kp, self.ki, self.kd, self.limit = kp, ki, kd, limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous = None
        self.derivative = 0.0

    def update(self, error, dt):
        # Filter the derivative and inhibit integration while output saturates.
        raw_d = 0.0 if self.previous is None else (error - self.previous) / dt
        self.derivative += dt / (0.15 + dt) * (raw_d - self.derivative)
        self.previous = error
        trial = float(np.clip(self.integral + error * dt, -0.15, 0.15))
        raw = self.kp * error + self.ki * trial + self.kd * self.derivative
        if abs(raw) <= self.limit or raw * error < 0:
            self.integral = trial
        return float(np.clip(raw, -self.limit, self.limit))


class LandingController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stage = Stage.ACQUIRE
        self.entered = 0.0
        self.stable_since = None
        self.last_seen = 0.0
        self.last_update = None
        self.last_capture = None
        self.events = []
        self.reason = ''
        self.x_pid = PID(cfg.kp_xy, cfg.ki_xy, cfg.kd_xy)
        self.y_pid = PID(cfg.kp_xy, cfg.ki_xy, cfg.kd_xy)

    def transition(self, stage, now, reason=''):
        previous_stage = self.stage
        self.events.append({'time': round(now, 4), 'from': self.stage.value,
                            'to': stage.value, 'reason': reason})
        self.stage, self.entered, self.reason = stage, now, reason
        self.stable_since = None
        # ALIGN and DESCEND have the same horizontal target. Keeping their
        # feedback history prevents command jumps and preserves disturbance
        # rejection when a frame is lost or descent is paused.
        if previous_stage == Stage.ACQUIRE or stage == Stage.ABORT:
            self.x_pid.reset()
            self.y_pid.reset()

    def abort(self, now, reason):
        self.transition(Stage.ABORT, now, reason)
        return Command(action='land')

    def update(self, now: float, pose: Pose | None, telemetry: Telemetry):
        if self.last_update is not None and now <= self.last_update:
            raise ValueError('Control clock must increase monotonically')
        dt = self.cfg.control_dt if self.last_update is None else now - self.last_update
        self.last_update = now
        c = self.cfg
        if self.stage in (Stage.COMPLETE, Stage.ABORT):
            return Command()
        if telemetry.soc < 0.15 and not telemetry.landed:
            return self.abort(now, 'low_battery')
        if self.stage in (Stage.ACQUIRE,Stage.ALIGN,Stage.DESCEND) and now > c.acquisition_timeout:
            return self.abort(now,'alignment_timeout')
        # Terminal landing/charging must progress even when the marker is invisible.
        if self.stage == Stage.LAND:
            if telemetry.landed:
                self.transition(Stage.DOCK, now, 'contact_and_motors_off')
            elif now - self.entered > c.land_timeout:
                return self.abort(now, 'landing_timeout')
            return Command()
        if self.stage == Stage.DOCK:
            if telemetry.dock_contact:
                self.transition(Stage.CHARGE, now, 'dock_contact')
            elif now - self.entered > 5.0:
                return self.abort(now, 'dock_missed')
            return Command()
        if self.stage == Stage.CHARGE:
            if not telemetry.dock_contact:
                return self.abort(now, 'dock_contact_lost')
            if telemetry.charging and telemetry.soc >= c.reflight_soc:
                self.transition(Stage.RELEASE, now, 'charge_target_reached')
                return Command(action='release')
            if now - self.entered > c.charge_timeout:
                return self.abort(now, 'charge_timeout')
            return Command()
        if self.stage == Stage.RELEASE:
            if telemetry.released:
                self.transition(Stage.REFLY, now, 'dock_released')
                return Command(action='takeoff')
            if now - self.entered > 3.0:
                return self.abort(now, 'release_timeout')
            return Command()
        if self.stage == Stage.REFLY:
            if telemetry.height >= c.reflight_height - 0.04:
                self.transition(Stage.COMPLETE, now, 'reflight_height_reached')
            elif now - self.entered > 12:
                return self.abort(now, 'takeoff_timeout')
            return Command()
        usable = pose is not None and pose.valid(now, c.pose_max_age)
        if not usable:
            self.stable_since = None
            self.x_pid.previous = None
            self.y_pid.previous = None
            self.x_pid.derivative = 0.0
            self.y_pid.derivative = 0.0
            self.last_capture = None
            if self.stage == Stage.DESCEND:
                self.transition(Stage.ALIGN, now, 'vision_lost_stop_descent')
            if now - self.last_seen > c.vision_abort_sec:
                return self.abort(now, 'vision_timeout')
            return Command()  # Explicit zero velocity; never retain the old descent command.
        self.last_seen = now
        if self.stage == Stage.ACQUIRE:
            self.transition(Stage.ALIGN, now, 'marker_acquired')
        # Only integrate/filter once per distinct camera observation.
        if self.last_capture != pose.captured_at:
            sample_dt = dt if self.last_capture is None else max(0.001, pose.captured_at - self.last_capture)
            self.vx = self.x_pid.update(-pose.x, sample_dt)
            self.vy = self.y_pid.update(-pose.y, sample_dt)
            self.last_capture = pose.captured_at
        error = math.hypot(pose.x, pose.y)
        aligned = error < c.xy_tolerance and abs(wrap(pose.yaw)) < c.yaw_tolerance
        up = 0.0
        if self.stage == Stage.ALIGN:
            if aligned:
                if self.stable_since is None:
                    self.stable_since = now
                elif now - self.stable_since >= c.stable_time:
                    self.transition(Stage.DESCEND, now, 'alignment_stable')
            else:
                self.stable_since = None
        elif self.stage == Stage.DESCEND:
            if error > c.reacquire_error or abs(wrap(pose.yaw)) > 2 * c.yaw_tolerance:
                self.transition(Stage.ALIGN, now, 'descent_drift')
            elif pose.z <= c.handoff_height:
                # Stop vertically above the camera blind zone and settle more
                # precisely than the cruise alignment threshold. This reserves
                # electrode overlap margin for drift during onboard landing.
                settled = (error < c.handoff_xy_tolerance and abs(wrap(pose.yaw)) < math.radians(4)
                           and math.hypot(self.x_pid.derivative,self.y_pid.derivative) < .05)
                if settled:
                    if self.stable_since is None:
                        self.stable_since = now
                    elif now-self.stable_since >= c.handoff_stable_sec:
                        self.transition(Stage.LAND, now, 'handoff_to_onboard_landing')
                        return Command(action='land')
                else:
                    self.stable_since = None
            else:
                self.stable_since = None
                up = -c.descent_speed
        cs, sn = math.cos(pose.yaw), math.sin(pose.yaw)
        return Command(right=cs * self.vx + sn * self.vy,
                       forward=-sn * self.vx + cs * self.vy, up=up,
                       yaw_ccw=float(np.clip(-c.kp_yaw * wrap(pose.yaw), -0.5, 0.5)))
