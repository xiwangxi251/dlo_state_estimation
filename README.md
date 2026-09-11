# DLO state estimation

独立的线缆（DLO）视觉状态识别仓库。它从 RGB-D 图像估计线缆上按弧长均匀分布的三维节点，当前默认输出 14 个节点。仓库与 `panda_cable_grasp` 主项目分开，便于在另一台电脑上单独安装、测试和复现实验。

## 仓库内容

```text
dlo_state_estimation/
├─ trackdlo_standalone/       # TrackDLO C++17/pybind11 核心和 Python API
├─ benchmark/                 # 当前改进版算法、评测、可视化和单元测试
│  ├─ dlo_position/           # HSV/骨架/排序/几何/时序/融合模块
│  ├─ run_trackdlo_current.py # 当前双视角 TrackDLO 评测入口
│  ├─ run_recorded_benchmark.py
│  ├─ make_*visualization*.py
│  └─ tests/
├─ requirements.txt
└─ .gitignore
```

数据集、视频、评测结果、`__pycache__` 和 C++ 编译产物均不提交，避免仓库膨胀。原始说明分别保存在 `benchmark/README_original.md` 和 `trackdlo_standalone/README_original.md`。

## 在另一台电脑安装

推荐把两个仓库并列放置（主项目仓库已经迁移完成）：

```powershell
git clone <DLO仓库地址> dlo_state_estimation
git clone git@github.com:xiwangxi251/mujoco_dynamic_DLO.git panda_cable_grasp

cd dlo_state_estimation
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

编译 TrackDLO 的 C++ 扩展还需要 Eigen3 和 C++17 编译器。Windows 例子：

```powershell
$env:EIGEN3_INCLUDE_DIR = "C:\path\to\eigen-3.4.0"
python -m pip install -e .\trackdlo_standalone
```

Linux 可以使用 `EIGEN3_INCLUDE_DIR=/usr/include/eigen3`。如果只想运行不依赖 MuJoCo 的 RGB-D 识别器，安装 TrackDLO 后即可；MuJoCo 和主项目只在“回放主项目日志、重新渲染相机、计算仿真真值”时需要。

## 运行当前改进版评测

`run_trackdlo_current.py` 默认值来自原电脑路径，跨电脑时请显式传入三个路径：

```powershell
python .\benchmark\run_trackdlo_current.py `
  --run-root "D:\data\linux_log\expert_grasp_fix_4x50\run_20260824_113325" `
  --project-src "..\panda_cable_grasp\src" `
  --trackdlo-root ".\trackdlo_standalone" `
  --cameras opst wrist `
  --episodes-per-scenario 5 `
  --frame-stride 5 `
  --output ".\results\trackdlo_current"
```

该入口包含当前使用的改进：全局/夹爪双视角、邻域点云可见性判断、可见节点优先使用当前观测、遮挡节点用时序状态补全，以及 CPD 未完全收敛时仍使用本次结果。算法默认只输出位置，不计算速度。

如果只测试 TrackDLO 原生离线序列：

```powershell
python .\trackdlo_standalone\scripts\run_offline.py <sequence_dir> --max-frames 20
```

## 直接接入实时 RGB-D 相机

```python
import numpy as np
from trackdlo_standalone import TrackDLOTracker

tracker = TrackDLOTracker(K)          # K 为 RGB 相机内参
first = tracker.initialize(rgb0, depth0)
state = tracker.update(rgb1, depth1)
nodes_xyz = state.nodes_camera       # (45, 3)，TrackDLO 原生节点
visible = state.visible_nodes
```

上层策略需要的 14 个均匀节点由 `benchmark/dlo_position` 中的重采样/融合流程生成。RGB-D 相机必须提供与 RGB 对齐的深度；深度可以是毫米 `uint16` 或米制浮点数组。

## 测试

```powershell
python -m unittest discover -s .\benchmark\tests -t .\benchmark -v
python -m unittest discover -s .\trackdlo_standalone\tests -v
```

## 与主项目的关系

| 内容 | 所在仓库 |
|---|---|
| RGB-D 分割、骨架排序、时序/双视角融合、14 点重采样、误差和视频 | 本仓库 `benchmark/` |
| TrackDLO CPD/C++ 核心 | 本仓库 `trackdlo_standalone/` |
| MuJoCo 环境、相机定义、仿真真值、RL/规则策略 | `panda_cable_grasp` 主项目 |

因此不会把主项目复制进来，也不会把日志和视频提交到 Git。拿到另一台电脑后，克隆本仓库并安装依赖，再把主项目路径通过 `--project-src` 指给它即可。

## 发布到远程 Git

当前仓库已在本地初始化并提交。拿到 GitHub/GitLab 空仓库地址后执行：

```powershell
git remote add origin <DLO仓库地址>
git branch -M main
git push -u origin main
```
