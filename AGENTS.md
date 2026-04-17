# Agent guide（yolo26-multitask）

## 项目目标

在 **vendored Ultralytics YOLO26**（`third_party/ultralytics`）上实现 **单模型多任务**：共享 backbone+neck，**检测 / 姿态 / 分割** 三支头并行；三类为同一物体不同组件 → **独立 `nc_det` / `nc_pose` / `nc_seg`**。

## 关键文件（优先读这些）

| 主题 | 路径 |
|------|------|
| 组合检测头 `MultiTask26` | `third_party/ultralytics/ultralytics/nn/modules/head.py` |
| 模型类 `MultiTaskModel`、YAML 解析、`parse_model` 中 `MultiTask26` 分支 | `third_party/ultralytics/ultralytics/nn/tasks.py` |
| 联合损失 `E2EMultiTaskLoss`、`TaskViewModel` | `third_party/ultralytics/ultralytics/utils/loss.py` |
| 模型 YAML 模板 | `third_party/ultralytics/ultralytics/cfg/models/26/yolo26-multitask.yaml` |
| 项目计划 / 待办 | `docs/PROJECT_PLAN.md` |
| 训练 batch / 标注对齐 | `docs/BATCH_SPEC.md`，`ymt/batch.py` |
| Smoke 测试 | `tests/test_multitask26.py` |
| 本地封装 | `ymt/model.py` |

## 对本 Agent 的约束

1. **以官方代码为基**：新逻辑优先写在 `third_party/ultralytics` 内与现有 YOLO26 模块一致的位置；避免在仓库根目录重写一套 YOLO。
2. **最小侵入**：只改多任务必需的分支；不要顺手格式化整个上游树。
3. **上游同步**：`third_party/ultralytics` 为**无 `.git` 的快照**；更新上游请按 `README.md` 重新克隆后再合并本地改动；每次合并后跑 `tests/test_multitask26.py` 并更新 `docs/PROJECT_PLAN.md` 风险说明。
4. **训练契约**：修改损失时注意 `Trainer` 对 `loss.sum()` 的假设；`E2EMultiTaskLoss` 返回 **1D loss 向量**。
5. **沟通**：较大里程碑向用户 **Slack** 简短报告（代码路径 + 完成项 + 下一步）；最终交付一个合并到 `main` 的正式 PR（按用户流程）。

## 常用命令

```bash
# 测试
PYTHONPATH=third_party/ultralytics python3 -m pytest tests/test_multitask26.py -v

# 可编辑安装上游（开发机）
python3 -m pip install -e "third_party/ultralytics"
```

## 与组织级说明冲突时

以组织 / 用户明确指令为准，并回写本文件去除过时约束。
