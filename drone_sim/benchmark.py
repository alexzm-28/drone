"""Repeatable nominal/randomised/fault experiments; exports JSON, CSV and HTML."""
import argparse
from dataclasses import replace
from datetime import datetime
import html
import json
from pathlib import Path
import csv
import time
import numpy as np
from .config import Config
from .control import Stage
from .simulation import Simulation


def experiment(cfg, mode, fault=None, duration=80, export=None):
    with Simulation(cfg, mode) as sim:
        while sim.now < duration:
            if fault == 'short_occlusion':
                sim.force_vision_loss = 5 <= sim.now < 5.6
            elif fault == 'long_occlusion':
                sim.force_vision_loss = sim.now >= 3
            sim.step()
            if sim.controller.stage == Stage.COMPLETE:
                break
            if sim.controller.stage == Stage.ABORT and (
                    sim.landed or sim.now-sim.controller.entered > cfg.abort_land_timeout):
                break
        result = sim.result()
        result['safe_abort'] = (sim.controller.stage == Stage.ABORT and sim.landed
                                and not sim.motors_on)
        if export:
            sim.save(export)
        return result


def write_report(rows, path, elapsed):
    path.mkdir(parents=True, exist_ok=True)
    (path/'benchmark.json').write_text(json.dumps({'wall_seconds': elapsed, 'runs': rows}, indent=2), encoding='utf-8')
    flat = []
    for r in rows:
        landing = r['landing'] or {}
        flat.append({k:r[k] for k in ('case','vision_mode','seed','expected','success','safe_abort','passed','stage','reason','simulation_seconds')} |
                    {'landing_error_m': landing.get('xy_error'), 'touchdown_speed_mps': landing.get('vertical_speed')})
    with (path/'benchmark.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=flat[0])
        writer.writeheader()
        writer.writerows(flat)
    summaries = []
    for mode in ('pose','camera'):
        group = [r for r in rows if r['vision_mode']==mode and r['expected']=='complete']
        if group:
            success = [r for r in group if r['success']]
            errors = [r['landing']['xy_error'] for r in success]
            summaries.append(f'{mode}: {len(success)}/{len(group)} completed; '
                             f'mean landing error {np.mean(errors)*100:.2f} cm' if errors else f'{mode}: 0/{len(group)} completed')
    passed = sum(r['passed'] for r in rows)
    headers = ''.join(f'<th>{html.escape(k)}</th>' for k in flat[0])
    table = ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row.values())+'</tr>' for row in flat)
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Landing Lab 实验报告</title>
<style>body{{background:#101820;color:#e7edf3;font:15px system-ui;margin:40px}}h1{{color:#8edd9b}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{padding:9px;border-bottom:1px solid #33424d;text-align:left}}th{{background:#20323b}}p{{line-height:1.8}}.cards{{display:flex;gap:20px}}.card{{padding:20px;background:#20323b;border-radius:12px}}</style>
<h1>Landing Lab / 自动着陆与充电仿真实验</h1>
<p>生成于 {datetime.now().isoformat(timespec='seconds')}。物理引擎：MuJoCo；只运行仿真，不连接实物。</p>
<div class="cards"><div class="card">符合预期：{passed}/{len(rows)}</div><div class="card">运行耗时：{elapsed:.1f}s</div></div>
<p>{'<br>'.join(html.escape(s) for s in summaries)}</p>
<p>camera 使用渲染图像和 OpenCV 识别；pose 使用合成位姿传感器。safe_abort 要求中止任务后实际落地且电机关闭。
成功要求完整经历充电与复飞、首次触地位于平台、误差 ≤ 4.5cm、触地竖直速度 ≤ 0.20m/s。
充电时钟默认加速60倍，飞行耗电与动力学未加速。模型参数未经真机辨识，不能据此声称真机精度。</p>
<div style="overflow:auto"><table><thead><tr>{headers}</tr></thead><tbody>{table}</tbody></table></div>
<p>完整配置与阶段事件见 benchmark.json；每行结果见 benchmark.csv。</p></html>'''
    (path/'report.html').write_text(content, encoding='utf-8')
    return {'runs':len(rows),'passed':passed,'summaries':summaries,'wall_seconds':round(elapsed,2)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=int, default=100, help='Randomised synthetic-sensor trials')
    p.add_argument('--camera-runs', type=int, default=10, help='Full rendered-camera trials')
    p.add_argument('--output', type=Path, default=Path('outputs/benchmark'))
    args = p.parse_args()
    if args.runs < 0 or args.camera_runs < 0:
        raise SystemExit('Run counts must be nonnegative')
    rng = np.random.default_rng(20261027)
    base = Config()
    cases = []
    for i in range(args.runs):
        cfg = replace(base, seed=1000+i, initial_x=float(rng.uniform(-.35,.35)),
                      initial_y=float(rng.uniform(-.25,.25)), initial_yaw=float(rng.uniform(-.65,.65)),
                      initial_z=float(rng.uniform(.95,1.35)), wind_accel=float(rng.uniform(0,.13)),
                      camera_latency=float(rng.uniform(.04,.14)), command_latency=float(rng.uniform(.04,.14)),
                      vision_dropout=float(rng.uniform(0,.04)))
        cases.append((f'random_{i:03}', cfg, 'pose', None, 'complete'))
    for i in range(args.camera_runs):
        cfg = replace(base, seed=2000+i, initial_x=float(rng.uniform(-.30,.30)),
                      initial_y=float(rng.uniform(-.2,.2)), initial_yaw=float(rng.uniform(-.5,.5)),
                      wind_accel=float(rng.uniform(0,.1)))
        cases.append((f'camera_{i:03}',cfg,'camera',None,'complete'))
    cases += [
        ('camera_nominal', base, 'camera', None, 'complete'),
        ('camera_short_occlusion', base, 'camera', 'short_occlusion', 'complete'),
        ('camera_long_occlusion', base, 'camera', 'long_occlusion', 'safe_abort'),
        ('camera_stale_sensor', replace(base,camera_latency=.4), 'camera', None, 'safe_abort'),
        ('outside_camera_fov', replace(base,initial_x=1.8), 'camera', None, 'safe_abort'),
        ('low_battery', replace(base,initial_soc=.12), 'pose', None, 'safe_abort'),
    ]
    rows = []
    start = time.perf_counter()
    for name, cfg, mode, fault, expected in cases:
        row = experiment(cfg, mode, fault, export=args.output/name if name.startswith('camera_') else None)
        row.update(case=name,expected=expected)
        row['passed'] = row['success'] if expected=='complete' else row['safe_abort']
        rows.append(row)
        print(f"{len(rows)}/{len(cases)} {name}: {row['stage']} {row['reason']} expected={expected} pass={row['passed']}",flush=True)
    summary = write_report(rows,args.output,time.perf_counter()-start)
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Report: {(args.output / "report.html").resolve()}')
    if not all(r['passed'] for r in rows):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
