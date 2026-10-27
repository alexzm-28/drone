from dataclasses import dataclass, asdict, fields
import math


@dataclass(frozen=True)
class Config:
    physics_dt: float = 0.005
    control_dt: float = 0.04
    mass: float = 0.087
    motor_tau: float = 0.055
    max_speed: float = 0.65
    max_yaw_rate: float = 1.0
    max_tilt: float = 0.32
    velocity_kp_xy: float = 8.0
    velocity_ki_xy: float = 4.0
    velocity_kp_z: float = 5.0
    velocity_ki_z: float = 2.0
    command_latency: float = 0.08
    camera_latency: float = 0.08
    camera_hz: float = 25.0
    camera_width: int = 640
    camera_height: int = 480
    camera_fovy: float = 70.0
    camera_z: float = 0.045
    marker_size: float = 0.10
    marker_id: int = 7
    marker_z: float = -0.024
    pad_top: float = 0.04
    feet_z: float = -0.052
    initial_x: float = 0.30
    initial_y: float = -0.18
    initial_z: float = 1.20
    initial_yaw: float = 0.25
    wind_accel: float = 0.045
    vision_dropout: float = 0.015
    pose_noise: float = 0.002
    xy_tolerance: float = 0.025
    yaw_tolerance: float = math.radians(6)
    stable_time: float = 0.8
    descent_speed: float = 0.12
    handoff_height: float = 0.34
    handoff_xy_tolerance: float = 0.018
    handoff_stable_sec: float = 0.4
    reacquire_error: float = 0.055
    pose_max_age: float = 0.25
    vision_abort_sec: float = 2.0
    acquisition_timeout: float = 60.0
    land_timeout: float = 12.0
    abort_land_timeout: float = 30.0
    dock_radius: float = 0.045
    electrode_radius: float = 0.035
    dock_yaw_tolerance: float = math.radians(12)
    dock_dwell: float = 1.0
    battery_capacity_wh: float = 4.18
    charge_power_w: float = 2.5
    flight_power_w: float = 15.0
    charge_time_scale: float = 60.0
    initial_soc: float = 0.60
    reflight_soc: float = 0.64
    charge_timeout: float = 60.0
    reflight_height: float = 0.9
    kp_xy: float = 1.25
    ki_xy: float = 0.08
    kd_xy: float = 0.30
    kp_yaw: float = 1.6
    seed: int = 42

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self,item.name)
            if isinstance(value,(int,float)) and not math.isfinite(value):
                raise ValueError(f'{item.name} must be finite')
        if self.physics_dt <= 0 or self.control_dt < self.physics_dt:
            raise ValueError('Invalid simulation timestep')
        if abs(round(self.control_dt / self.physics_dt) * self.physics_dt - self.control_dt) > 1e-9:
            raise ValueError('control_dt must be an integer multiple of physics_dt')
        for name in ('mass', 'motor_tau', 'max_speed', 'max_yaw_rate', 'camera_hz',
                     'marker_size', 'battery_capacity_wh', 'charge_power_w', 'charge_time_scale'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if not 0 <= self.vision_dropout < 1 or not 0 <= self.initial_soc < self.reflight_soc <= 1:
            raise ValueError('Invalid dropout or battery configuration')
        if min(self.command_latency, self.camera_latency, self.pose_noise) < 0:
            raise ValueError('Negative latency/noise')
        if not 10 < self.camera_fovy < 160 or min(self.camera_width,self.camera_height) < 64:
            raise ValueError('Invalid camera geometry')
        if not 0 <= self.marker_id < 50 or self.electrode_radius <= 0:
            raise ValueError('Invalid marker/electrode configuration')
        if min(self.kp_xy,self.ki_xy,self.kd_xy,self.kp_yaw,self.wind_accel) < 0:
            raise ValueError('Negative gain/disturbance')

    def to_dict(self):
        return asdict(self)
