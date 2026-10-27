import math
from dataclasses import replace
import pytest
from drone_sim.config import Config
from drone_sim.control import Command, LandingController, PID, Pose, Stage, Telemetry


def test_pose_age_nan_future_rejected():
    assert Pose(0, 0, 1, 0, 1).valid(1.2, .25)
    assert not Pose(0, 0, 1, 0, 1).valid(1.3, .25)
    assert not Pose(float('nan'), 0, 1, 0, 1).valid(1.2, .25)
    assert not Pose(0, 0, 1, 0, 2).valid(1.2, .25)


def test_coordinate_transform_at_90_degrees():
    cfg = Config()
    controller = LandingController(cfg)
    # A positive world-X offset needs negative world-X motion. At yaw +90,
    # this must be a positive body-forward command, not body-right.
    command = controller.update(.1, Pose(.2, 0, 1, math.pi/2, .1), Telemetry())
    assert abs(command.right) < 1e-8
    assert command.forward > 0
    assert command.yaw_ccw < 0


def test_descent_stops_on_first_missing_observation():
    controller = LandingController(Config())
    controller.stage = Stage.DESCEND
    controller.last_seen = 1
    command = controller.update(1.04, None, Telemetry())
    assert command == Command()
    assert controller.stage == Stage.ALIGN
    assert controller.stable_since is None


def test_stale_pose_cannot_drive_descent():
    controller = LandingController(Config())
    controller.stage = Stage.DESCEND
    controller.last_seen = 1
    command = controller.update(1.4, Pose(.01, 0, .5, 0, 1), Telemetry())
    assert command.up == 0
    assert controller.stage == Stage.ALIGN


def test_land_continues_without_vision_and_never_invents_contact():
    c = LandingController(Config())
    c.stage = Stage.DESCEND
    c.update(.5, Pose(0,0,.3,0,.5), Telemetry())
    command = c.update(1, Pose(0, 0, .3, 0, 1), Telemetry())
    assert command.action == 'land'
    assert c.stage == Stage.LAND
    c.update(2, None, Telemetry())
    assert c.stage == Stage.LAND
    c.update(3, None, Telemetry(landed=True))
    assert c.stage == Stage.DOCK


def test_low_altitude_handoff_waits_for_electrode_alignment_margin():
    c = LandingController(Config())
    c.stage = Stage.DESCEND
    for i in range(20):
        now = .1+i*.04
        command = c.update(now,Pose(.02,0,.3,0,now),Telemetry())
        assert command.up == 0 and command.action == ''
    assert c.stage == Stage.DESCEND


def test_charge_requires_contact_and_power_then_release_before_takeoff():
    c = LandingController(Config())
    c.stage = Stage.CHARGE
    command = c.update(1, None, Telemetry(landed=True, dock_contact=True, charging=True, soc=.65))
    assert c.stage == Stage.RELEASE and command.action == 'release'
    command = c.update(1.1, None, Telemetry(landed=True, soc=.65))
    assert command.action == ''
    command = c.update(1.2, None, Telemetry(landed=True, released=True, soc=.65))
    assert c.stage == Stage.REFLY and command.action == 'takeoff'


def test_long_occlusion_aborts_with_land_and_no_emergency():
    c = LandingController(Config())
    c.update(.04, Pose(0, 0, 1, 0, 0), Telemetry())
    command = c.update(2.2, None, Telemetry())
    assert c.stage == Stage.ABORT
    assert command.action == 'land'
    assert c.reason == 'vision_timeout'


def test_repeated_capture_does_not_integrate_twice():
    c = LandingController(Config())
    pose = Pose(.1, 0, 1, 0, .01)
    c.update(.04, pose, Telemetry())
    integral = c.x_pid.integral
    c.update(.08, pose, Telemetry())
    assert c.x_pid.integral == integral


def test_short_dropout_preserves_integral_but_stops_output():
    c = LandingController(Config())
    c.update(.04,Pose(.1,0,1,0,.04),Telemetry())
    c.update(.08,Pose(.1,0,1,0,.08),Telemetry())
    integral = c.x_pid.integral
    c.stage = Stage.DESCEND
    assert c.update(.12,None,Telemetry()) == Command()
    assert c.x_pid.integral == integral
    assert c.x_pid.previous is None


def test_pid_integrator_does_not_wind_up_when_saturated():
    pid = PID(10, 10, 0, limit=.3)
    for _ in range(500):
        assert pid.update(1, .04) == .3
    assert pid.integral == 0


def test_monotonic_clock_required():
    c = LandingController(Config())
    c.update(.1, None, Telemetry())
    with pytest.raises(ValueError):
        c.update(.1, None, Telemetry())


def test_alignment_cannot_hover_forever():
    c = LandingController(Config())
    cmd = c.update(61,Pose(.02,0,.3,0,61),Telemetry())
    assert c.stage == Stage.ABORT and cmd.action == 'land'
    assert c.reason == 'alignment_timeout'


def test_invalid_timestep_rejected():
    with pytest.raises(ValueError):
        replace(Config(), control_dt=.043)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1])
def test_invalid_gains_rejected(value):
    with pytest.raises(ValueError):
        replace(Config(),kp_xy=value)
