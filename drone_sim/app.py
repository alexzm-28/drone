"""Interactive viewer, replayable CLI experiments and report export."""
import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import time
from contextlib import ExitStack
import cv2
import numpy as np
from .config import Config
from .control import Stage
from .simulation import Simulation

BG = (22, 18, 13)
PANEL = (37, 31, 24)
TEXT = (229, 235, 240)
MUTED = (149, 158, 170)
GREEN = (157, 224, 79)
CYAN = (239, 197, 65)
ORANGE = (74, 158, 250)


def label(img, text, x, y, size=0.55, color=TEXT, weight=1):
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, weight, cv2.LINE_AA)


def draw_dashboard(sim, paused=False, speed=1.0):
    frame = np.full((940, 1500, 3), BG, dtype=np.uint8)
    label(frame, 'LANDING LAB', 24, 38, 0.95, GREEN, 2)
    label(frame, 'Closed-loop 3D simulation / NO HARDWARE', 285, 36, 0.55, MUTED)
    label(frame, f'{sim.vision_mode.upper()}  |  t {sim.now:05.1f}s  |  {speed:.1f}x' + ('  PAUSED' if paused else ''),
          1020, 36, 0.56, CYAN)
    stages = list(Stage)[:-1]
    for i, st in enumerate(stages):
        x = 24 + i*146
        active = sim.controller.stage == st
        cv2.rectangle(frame, (x, 56), (x+136, 87), GREEN if active else PANEL, -1)
        label(frame, st.value, x+8, 77, 0.44, BG if active else MUTED, 1)
    scene = cv2.cvtColor(sim.render_overview(), cv2.COLOR_RGB2BGR)
    frame[104:704, 24:924] = scene
    label(frame, 'WORLD VIEW', 40, 132, 0.55, CYAN, 2)
    label(frame, 'Drag: orbit / wheel: zoom / 1-3: views', 40, 156, 0.43, MUTED)
    p = sim.position
    label(frame, f'X {p[0]:+.3f}m   Y {p[1]:+.3f}m   Z {p[2]:.3f}m', 40, 680, 0.62)
    cv2.rectangle(frame, (948, 104), (1476, 485), PANEL, -1)
    image = cv2.cvtColor(sim.rgb, cv2.COLOR_RGB2BGR)
    if sim.vision_mode == 'camera' and sim.vision.corners is not None and not sim.force_vision_loss:
        cv2.polylines(image, [sim.vision.corners.astype(np.int32)], True, GREEN, 2, cv2.LINE_AA)
    frame[134:476, 960:1464] = cv2.resize(image, (504, 342))
    label(frame, 'GROUND CAMERA / ArUco 4x4 ID 7' if sim.vision_mode == 'camera' else 'POSE MODE / SYNTHETIC SENSOR',
          960, 124, 0.48, CYAN)
    cv2.rectangle(frame, (948, 500), (1476, 888), PANEL, -1)
    good = sim.pose is not None and sim.pose.valid(sim.now, sim.cfg.pose_max_age)
    label(frame, sim.controller.stage.value, 968, 535, 0.9, GREEN if sim.controller.stage != Stage.ABORT else ORANGE, 2)
    lines = [
        ('Vision', 'TRACKING' if good else 'LOST / BRAKE'),
        ('XY error (truth)', f'{np.linalg.norm(p[:2])*100:.1f} cm'),
        ('Yaw', f'{math.degrees(sim.yaw):+.1f} deg'),
        ('Velocity Z', f'{sim.data.qvel[2]:+.2f} m/s'),
        ('Contact / motors', f'{sim.landed} / {sim.motors_on}'),
        ('Dock / charging', f'{sim.dock_contact} / {sim.charging}'),
        ('Battery', f'{sim.soc*100:.1f}%  (target {sim.cfg.reflight_soc*100:.0f}%)'),
        ('Charge clock', f'{sim.cfg.charge_time_scale:g}x; flight clock 1x'),
        ('Wind / loss', f'{sim.wind_enabled} / {sim.force_vision_loss}'),
        ('PID P / I / D', f'{sim.cfg.kp_xy:.2f} / {sim.cfg.ki_xy:.2f} / {sim.cfg.kd_xy:.2f}'),
        ('Camera / command delay', f'{sim.cfg.camera_latency*1000:.0f} / {sim.cfg.command_latency*1000:.0f} ms'),
    ]
    for i, (name, value) in enumerate(lines):
        label(frame, name, 968, 567+i*26, 0.45, MUTED)
        label(frame, value, 1190, 567+i*26, 0.45)
    # Fixed-scale plots: elapsed seconds on X, metres on Y; no misleading autoscale.
    cv2.rectangle(frame, (24, 720), (924, 888), PANEL, -1)
    label(frame, 'LAST 20s   altitude 0..1.5m / horizontal error 0..0.5m', 40, 744, 0.47, MUTED)
    history = [row for row in sim.history if row['time'] >= sim.now-20]
    for i in range(4):
        y = 769 + i*33
        cv2.line(frame, (44, y), (900, y), (55, 48, 40), 1)
    for key, maximum, color in [('z', 1.5, GREEN), ('error', 0.5, CYAN)]:
        points = []
        for row in history:
            value = math.hypot(row['x'], row['y']) if key == 'error' else row[key]
            x = 44+int((row['time']-max(0, sim.now-20))/20*856)
            y = 871-int(np.clip(value/maximum, 0, 1)*105)
            points.append((x, y))
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points, np.int32)], False, color, 2, cv2.LINE_AA)
    label(frame, 'SPACE pause | R reset | V occlude | W wind | arrows gust | +/- speed | S export | Q quit',
          24, 920, 0.50, MUTED)
    if sim.controller.stage in (Stage.COMPLETE, Stage.ABORT):
        label(frame, f'{sim.controller.stage.value}: {sim.controller.reason}', 40, 190, 0.6,
              GREEN if sim.result()['success'] else ORANGE, 2)
    return frame


def parser():
    p = argparse.ArgumentParser(description='Indoor landing and charging simulation. No hardware access.')
    p.add_argument('--headless', action='store_true', help='No GUI; camera mode still renders images offscreen')
    p.add_argument('--vision', choices=['camera', 'pose'], default='camera')
    p.add_argument('--duration', type=float, default=80)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--snapshot', action='store_true')
    p.add_argument('--record', type=Path, default=None, help='Export actual dashboard frames to MP4')
    p.add_argument('--snapshot-stages', action='store_true', help='Save one dashboard per mission stage')
    p.add_argument('--smoke-test', action='store_true', help='Open viewer, render 3 frames, then exit')
    p.add_argument('--kp', type=float, default=1.25)
    p.add_argument('--ki', type=float, default=0.08)
    p.add_argument('--kd', type=float, default=0.30)
    p.add_argument('--wind', type=float, default=0.045, help='Wind acceleration amplitude m/s^2')
    p.add_argument('--latency', type=float, default=0.08, help='Camera and command delay in seconds')
    p.add_argument('--charge-scale', type=float, default=60)
    p.add_argument('--loss-start', type=float, default=-1)
    p.add_argument('--loss-duration', type=float, default=0)
    return p


def main():
    args = parser().parse_args()
    if args.duration <= 0 or args.loss_duration < 0:
        raise SystemExit('duration must be positive and loss-duration nonnegative')
    cfg = replace(Config(), seed=args.seed, kp_xy=args.kp, ki_xy=args.ki, kd_xy=args.kd,
                  wind_accel=args.wind, camera_latency=args.latency, command_latency=args.latency,
                  charge_time_scale=args.charge_scale)
    output = args.output or Path('outputs')/datetime.now().strftime('%Y%m%d_%H%M%S')
    paused, speed, quit_requested = False, 1.0, False
    writer, saved_stages, frames = None, set(), 0
    if args.record:
        args.record.parent.mkdir(parents=True,exist_ok=True)
        writer = cv2.VideoWriter(str(args.record),cv2.VideoWriter_fourcc(*'mp4v'),1/cfg.control_dt,(1500,940))
        if not writer.isOpened():
            raise SystemExit('Could not create MP4 writer')
    with ExitStack() as resources, Simulation(cfg, args.vision) as sim:
        if writer is not None:
            resources.callback(writer.release)
        if not args.headless:
            cv2.namedWindow('Landing Lab', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Landing Lab', 1350, 846)
            drag = [None]
            def mouse(event, x, y, flags, _):
                if event == cv2.EVENT_LBUTTONDOWN and 24 <= x < 924 and 104 <= y < 704:
                    drag[0] = (x,y)
                elif event == cv2.EVENT_LBUTTONUP:
                    drag[0] = None
                elif event == cv2.EVENT_MOUSEMOVE and drag[0] is not None:
                    dx,dy = x-drag[0][0],y-drag[0][1]
                    sim.view_camera.azimuth -= dx*.3
                    sim.view_camera.elevation = float(np.clip(sim.view_camera.elevation-dy*.3,-89,-5))
                    drag[0] = (x,y)
                elif event == cv2.EVENT_MOUSEWHEEL:
                    sim.view_camera.distance = float(np.clip(sim.view_camera.distance*(.9 if flags>0 else 1.1),.35,5))
            cv2.setMouseCallback('Landing Lab', mouse)
        next_frame = 0.0
        while sim.now < args.duration:
            start = time.perf_counter()
            if not paused:
                if args.loss_start >= 0:
                    sim.force_vision_loss = args.loss_start <= sim.now < args.loss_start+args.loss_duration
                sim.step()
            terminal = sim.controller.stage == Stage.COMPLETE or (
                sim.controller.stage == Stage.ABORT and (sim.landed or sim.now-sim.controller.entered > cfg.abort_land_timeout))
            if writer is not None or args.snapshot_stages:
                dashboard = draw_dashboard(sim,paused,speed)
                if writer is not None and not paused:
                    writer.write(dashboard)
                stage = sim.controller.stage.value
                if args.snapshot_stages and stage not in saved_stages:
                    output.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(output/f'stage_{stage}.png'),dashboard)
                    saved_stages.add(stage)
            if args.headless:
                if terminal:
                    break
                continue
            if time.perf_counter() >= next_frame or paused or terminal:
                cv2.imshow('Landing Lab', draw_dashboard(sim, paused, speed))
                next_frame = time.perf_counter()+1/30
            key = cv2.waitKeyEx(1)
            frames += 1
            if args.smoke_test and frames >= 3:
                break
            if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty('Landing Lab', cv2.WND_PROP_VISIBLE) < 1:
                quit_requested = True
                break
            if key == ord(' '):
                paused = not paused
            elif key in (ord('r'), ord('R')):
                sim.reset()
                paused = False
                terminal = False
            elif key in (ord('v'), ord('V')):
                sim.force_vision_loss = not sim.force_vision_loss
            elif key in (ord('w'), ord('W')):
                sim.wind_enabled = not sim.wind_enabled
            elif key in (ord('s'), ord('S')):
                sim.save(output)
                cv2.imwrite(str(output/'dashboard.png'), draw_dashboard(sim, paused, speed))
                print(f'Saved {output.resolve()}', flush=True)
            elif key in (ord('+'), ord('=')):
                speed = min(4, speed*2)
            elif key == ord('-'):
                speed = max(0.25, speed/2)
            elif key in (ord('1'),ord('2'),ord('3')):
                sim.follow_view = key == ord('3')
                sim.view_camera.lookat[:] = [0,0,.5 if key != ord('2') else 0]
                sim.view_camera.distance = 1.8
                sim.view_camera.elevation = -89 if key == ord('2') else -22
                sim.view_camera.azimuth = -55
            elif key in (2424832, 2555904, 2490368, 2621440):
                axis = 0 if key in (2424832, 2555904) else 1
                sign = -1 if key in (2424832, 2621440) else 1
                sim.data.qvel[axis] += sign*0.18
            if terminal:
                paused = True  # Keep the result visible; R starts another run.
            time.sleep(max(0, cfg.control_dt/speed - (time.perf_counter()-start)))
        sim.save(output)
        if writer is not None:
            writer.release()
        if args.snapshot:
            cv2.imwrite(str(output/'dashboard.png'), draw_dashboard(sim, paused, speed))
            cv2.imwrite(str(output/'camera.png'), cv2.cvtColor(sim.rgb, cv2.COLOR_RGB2BGR))
        result = sim.result()
        print(json.dumps({k:v for k,v in result.items() if k not in ('config','events')}, indent=2))
        print(f'Artifacts: {output.resolve()}')
        if not args.headless:
            cv2.destroyAllWindows()
        if args.headless and not result['success'] and not args.smoke_test:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
