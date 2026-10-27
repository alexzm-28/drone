# Landing Lab：室内无人机自动着陆与充电仿真

Python + MuJoCo + OpenCV 的可交互三维闭环仿真。**只运行仿真，不连接无人机、串口或充电硬件。**

任务流程：识别 → 对准 → 下降 → 机载降落 → 接触确认 → 对接 → 充电 → 释放 → 复飞。

这套实现用刚体动力学推进飞行，用渲染图像估计位置，用实际碰撞判断落地。相机看不到标签时，控制器会停止下降；持续失去视觉则中止任务并执行降落。

## 启动

需要 **Python 3.12 64 位**、可用的 OpenGL 图形驱动。Windows 下在项目目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_sim.ps1
.\run_sim.ps1
```

已安装依赖时，也可以双击 `Run Simulation.cmd`，或者：

```powershell
.\.venv\Scripts\python.exe -m drone_sim
```

Linux：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m drone_sim
```

Linux 无显示器环境可安装 `xvfb`、`xauth`、`libgl1`、`libglfw3`，用 `xvfb-run -a` 执行下文的命令。`--headless` 仅关闭交互窗口，图像模式仍然需要离屏渲染上下文。

## 交互

| 操作 | 功能 |
|---|---|
| 鼠标左键拖动三维区域 / 滚轮 | 环绕观察 / 缩放 |
| `1` / `2` / `3` | 默认视角 / 俯视 / 跟随 |
| 空格 | 暂停 / 继续 |
| `R` | 用同一随机种子重新开始 |
| `V` | 遮挡地面相机，向识别器输入空白图像 |
| `W` | 开关风扰 |
| 方向键 | 注入 0.18m/s 水平速度扰动 |
| `+` / `-` | 调整播放速度，物理步长保持不变 |
| `S` | 导出当前数据、结果和界面截图 |
| `Q` / Esc | 退出并保存数据 |

任务完成或中止后，窗口保持显示结果，按 `R` 重跑。默认结果写入 `outputs/日期_时间/`。

## 两种传感器模式

- **`camera`，默认**：MuJoCo 地面朝上相机 → 渲染 RGB 图像 → ArUco 4×4 ID 7 检测 → PnP 位姿 → 控制器。控制器不读取无人机真实坐标。
- **`pose`，快速实验**：带视场限制、噪声、丢帧和延迟的合成位姿。适合批量验证控制策略，不能作为图像识别效果的证据。

两种模式共用控制器、刚体物理、电池和对接模型。动力学没有预设轨迹；Blender 动画不是本实现的运行依赖。

## 常用实验

```powershell
# 完整图像闭环，导出截图和各阶段截图
.\run_sim.ps1 --headless --snapshot --snapshot-stages --output outputs\camera_run

# 录制实际仿真界面，导出的 MP4 按仿真时间播放
.\run_sim.ps1 --headless --record outputs\demo.mp4 --output outputs\demo

# 改 PID、风扰、图像和指令延迟
.\run_sim.ps1 --kp 1.1 --ki 0.06 --kd 0.35 --wind 0.10 --latency 0.12

# 在第5秒遮挡相机0.6秒，观察暂停下降和重新对准
.\run_sim.ps1 --loss-start 5 --loss-duration 0.6

# 长时遮挡会中止并降落；命令行退出码2表示任务未成功完成
.\run_sim.ps1 --headless --loss-start 3 --loss-duration 50

# 验证交互窗口能创建和绘制，随后自动退出
.\run_sim.ps1 --smoke-test
```

## 测试与报告

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m drone_sim.benchmark --runs 100 --camera-runs 10 --output outputs\validation
```

批量实验默认包含 100 次随机化合成传感器实验、10 次随机化完整图像实验，以及正常图像、短时遮挡、持续遮挡、过期图像、目标出视场、低电量等 6 个用例。

- `report.html`：可用浏览器打开的中文报告。
- `benchmark.csv`：每次实验的结果与落地指标。
- `benchmark.json`：完整模型配置、随机种子和阶段事件。
- 各图像实验子目录中的 `telemetry.csv` / `result.json`：逐帧数据与任务结果。

**成功判定**：完成充电和复飞；首次触地位于平台；落点距离中心 ≤4.5cm；首次触地竖直速度 ≤0.20m/s。故障用例另行检查是否在中止后落地并关闭电机，故障处理成功不计为任务成功。

## 模型边界

这是带视觉闭环的工程简化模型。机载速度/姿态控制器为自行实现的代理模型，质量、惯量、推力滞后和风扰均可配置，**没有做真机参数辨识**。报告中的厘米或毫米级误差只适用于所列仿真条件。

默认充电时钟加速 **60 倍**，飞行动力学和飞行耗电保持正常仿真时间；可用 `--charge-scale 1 --duration 500` 观察不加速的充电。充电采用恒定有效功率的能量模型，没有模拟电池温度和完整 CC/CV 充电曲线。对接电极和释放机构使用几何条件与状态信号建模。

详细方程、坐标系、传感器约束及验收要求见 [仿真设计](docs/simulation_design.md)。

## 结构

```text
drone_sim/config.py       模型和任务参数
drone_sim/scene.py        三维场景、机体、标签、平台
drone_sim/vision.py       图像检测与位姿求解
drone_sim/control.py      独立的外环 PID 与任务状态机
drone_sim/simulation.py   刚体物理、机载控制、延迟、接触、电池
drone_sim/app.py          交互窗口、数据导出、视频录制
drone_sim/benchmark.py    随机化和故障实验、报告
tests/                   控制、视觉、动力学、任务验证
```

旧的 `DOC/`、`REDEME/` 与 `Images/` 是本地历史资料；当前可运行仿真入口是本 README。历史硬件程序与动画不属于当前仿真交付，也不需要安装。

实现依据：[MuJoCo Python 接口](https://mujoco.readthedocs.io/en/stable/python.html)、[MuJoCo 动力学](https://mujoco.readthedocs.io/en/stable/computation/index.html)、[OpenCV ArUco 检测与位姿估计](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)。
