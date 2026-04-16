# yolo26-multitask

在 **Ultralytics 官方 YOLO26** 代码基础上扩展：**共享 backbone + PAN-FPN**，在 **检测头侧** 并行挂载 **检测（Detect）**、**姿态（Pose26）**、**实例分割（Segment26）** 三个分支，使同一模型在训练与推理中同时输出三类结果。

设计要点：检测 / 姿态 / 分割对应**同一物体的不同组件**，因此在模型里使用 **三套独立类别数**（`nc_det`、`nc_pose`、`nc_seg`），**不**把三类合并为单一 `nc`。

## 仓库结构

| 路径 | 说明 |
|------|------|
| `third_party/ultralytics/` | 官方 [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) 克隆；**本项目的修改都在此树内**（便于 `git pull` 跟进上游） |
| `third_party/ultralytics/ultralytics/cfg/models/26/yolo26-multitask.yaml` | 多任务模型配置模板 |
| `third_party/ultralytics/ultralytics/nn/modules/head.py` | 新增 `MultiTask26` 组合头 |
| `third_party/ultralytics/ultralytics/nn/tasks.py` | `MultiTaskModel`、`parse_model` / stride 初始化、`guess_model_task` |
| `third_party/ultralytics/ultralytics/utils/loss.py` | `E2EMultiTaskLoss`、`TaskViewModel` |
| `ymt/` | 薄封装：`load_multitask_model()` |
| `docs/PROJECT_PLAN.md` | **项目计划**与待办 |
| `tests/test_multitask26.py` | 构建与前向 smoke 测试 |

## 上游更新

本仓库中的 `third_party/ultralytics` 为**源码快照**（不含嵌套 `.git`）。要与官方仓库同步，可重新克隆或覆盖：

```bash
rm -rf third_party/ultralytics
git clone --depth 1 https://github.com/ultralytics/ultralytics.git third_party/ultralytics
rm -rf third_party/ultralytics/.git   # 若希望继续作为单仓库子目录提交
# 然后重新应用本项目的补丁文件（或使用你自己的合并流程）
```

若上游改动与本地补丁冲突，需手动合并后重新跑测试：

```bash
PYTHONPATH=third_party/ultralytics python3 -m pytest tests/test_multitask26.py -v
```

## 安装

推荐将 vendored 包以可编辑方式安装（会拉取 `opencv-python`、`pillow` 等 Ultralytics 依赖）：

```bash
python3 -m pip install -e "third_party/ultralytics"
python3 -m pip install -e ".[dev]"
```

## 快速使用（PyTorch）

```python
from pathlib import Path
from ultralytics.nn.tasks import MultiTaskModel

yaml_path = Path("third_party/ultralytics/ultralytics/cfg/models/26/yolo26-multitask.yaml")
model = MultiTaskModel(cfg=str(yaml_path), ch=3, verbose=False)

# 训练态：最后一层返回 dict，键为 det / pose / seg，结构与各单任务 head 一致（含 E2E one2many / one2one）
model.train()
import torch
x = torch.randn(2, 3, 640, 640)
out = model(x)
```

推理态下各分支仍返回 Ultralytics 原有格式（元组或张量）；若需要统一封装，可在业务层解析 `out["det"]` 等。

### 类别与关键点

编辑 `yolo26-multitask.yaml` 顶部字段：

- `nc_det` / `nc_pose` / `nc_seg`：三套类别数  
- `kpt_shape: [K, 3]`：姿态分支关键点（与官方 pose 数据格式一致）

`nc` 字段保留用于兼容 Ultralytics 的 `names` 字典（默认取 `max(nc_det, nc_pose, nc_seg)` 也可在 YAML 里手写）。

## 损失与训练

- `MultiTaskModel.init_criterion()` 返回 `E2EMultiTaskLoss`：对 det / pose / seg 分别构造 `v8DetectionLoss`、`PoseLoss26`、`v8SegmentationLoss`，并通过 `TaskViewModel` 让每个损失只「看到」对应子头（`model.model[-1]` 代理）。
- 返回值形状为 **14 维向量**（3+5+6：det + pose + seg 各分支 loss 分量拼接），与 `Trainer` 中 `loss.sum()` 用法兼容。

**重要**：当前实现假定 batch 内仍提供 Ultralytics 标准键（`batch_idx`、`bboxes`、`keypoints`、`masks` 等）。

- **类别**：若检测 / 姿态 / 分割的类别 ID **互不共用**，请在 batch 中提供 `cls_det`、`cls_pose`、`cls_seg`（张量格式与官方 `cls` 相同）；未提供时三任务都会回退使用同一个 `cls`（与旧行为兼容）。
- 若某图像缺少某任务的标注，需在数据层给出空张量或填充策略；多任务联合 assigner 的细粒度行为仍需在你自己的数据集上验证。

完整任务拆解见 `docs/PROJECT_PLAN.md`。

## 测试（本环境已跑通）

```bash
PYTHONPATH=third_party/ultralytics python3 -m pytest tests/test_multitask26.py -v
```

## 许可

- 对 `third_party/ultralytics` 的修改与上游相同，遵循 **AGPL-3.0**（见上游 `LICENSE`）。  
- 仓库根目录的 `ymt/`、`tests/`、`docs/` 等补充文件以仓库内声明为准。
