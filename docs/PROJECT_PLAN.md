# 项目计划（yolo26-multitask）

本文档与根目录 `README.md` 同步维护；较大里程碑在 Slack 同步。

## 目标

在 **Ultralytics YOLO26 官方主干**（共享 backbone + PAN-FPN）上，通过 **检测头侧** 扩展，使同一模型在训练与推理中同时输出：

- **检测**：部件级 bbox + `nc_det` 类（与姿态/分割类表独立）
- **姿态**：`nc_pose` 类 + `kpt_shape` 关键点（YOLO26 `Pose26`）
- **分割**：`nc_seg` 类 + mask 系数 + `Proto26`（YOLO26 `Segment26`）

语义约束：三类来自同一物体的不同组件，但 **不作为同一检测类别** 合并。

## 已完成

| 任务 | 说明 |
|------|------|
| 引入上游 | `third_party/ultralytics` 为官方源码目录（快照，无嵌套 `.git`）；同步方式见根 `README.md` |
| 组合头 `MultiTask26` | `ultralytics/nn/modules/head.py`：三支头共享 `x`（P3/P4/P5） |
| 解析与 stride | `parse_model` 支持 `nc_det`/`nc_pose`/`nc_seg`；`DetectionModel` 对 `MultiTask26` 同步子头 stride |
| 联合损失 `E2EMultiTaskLoss` | `ultralytics/utils/loss.py`：`TaskViewModel` 包装使各 `v8*Loss` 读到对应子头 |
| 模型类 `MultiTaskModel` | `ultralytics/nn/tasks.py`：`init_criterion` → `E2EMultiTaskLoss` |
| YAML 模板 | `ultralytics/cfg/models/26/yolo26-multitask.yaml` |
| 逻辑自测 | `tests/test_multitask26.py`（构建 + 训练态前向 smoke） |

## 待办（你本地 / 下一迭代）

| 任务 | 说明 |
|------|------|
| 数据管线 | 将标注转为 Ultralytics 训练 `batch`：`bboxes` + **`cls_det` / `cls_pose` / `cls_seg`**（或与旧版兼容的共享 `cls`）、`keypoints`（pose）、`masks`（seg）；三类可不同实例数 |
| 训练入口 | 基于 `ultralytics` 的 `Trainer` 扩展或自定义 loop，确保 `criterion.update()` 与官方 E2E 一致 |
| 推理与导出 | 统一后处理：三份输出；ONNX/TensorRT 需分别处理各分支 |
| 权重初始化 | 从 `yolo26n.pt` 等加载 backbone+neck，头部分支按需初始化或微调 |

## 风险与假设

- 联合损失将 **检测、姿态、分割的 GT 共用同一 `batch_idx`/`bboxes`/`cls` 张量**；若某任务无标注，需在数据层填零并 mask，否则 assigner 行为需再验证。
- `E2EMultiTaskLoss` 中 `o2m/o2o` 调度使用 `hyp.epochs`（来自 pose 子损失），与单任务 E2E 行为对齐；若需按 det 调度可再改。
