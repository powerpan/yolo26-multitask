# 多任务训练 batch 约定（与 Ultralytics 对齐）

本项目的 `MultiTaskModel` + `E2EMultiTaskLoss` 在 **不修改** Ultralytics `v8DetectionLoss` / `PoseLoss26` / `v8SegmentationLoss` 的前提下，复用其张量约定。下列字段若与官方不一致，**最容易在静默中训错**。

## 1. 所有分支共用的张量

| 键 | 形状 / 含义 | 注意 |
|----|----------------|------|
| `img` | `(B, 3, H, W)` | 与单任务相同 |
| `batch_idx` | `(N, 1)` 或 `(N,)` | 每条 GT 实例所属图像下标，与官方相同 |
| `bboxes` | `(N, 4)` | **xywh 归一化到 0–1**（相对整图宽高），与 `v8DetectionLoss.preprocess` 一致；**三个分支的检测/分配都基于这一套框** |
| `keypoints` | `(N_kpt_rows, K, D)` | `K,D` = YAML `kpt_shape`；坐标为 **相对整图宽高的 0–1**，与 pose 任务一致；`D==3` 时第三维为可见性（0=不可见） |
| `masks` | 见下 | 实例分割 GT；格式与 Ultralytics `overlap_mask` 设置一致 |
| `sem_masks` | `(B, H, W)`，`long` | **YOLO26 `Segment26` 训练必需**：`Proto26` 前向在 `training` 时会返回 `(proto, pred_semseg)`，`v8SegmentationLoss` 在存在正 anchor 时会读取 `batch["sem_masks"]` 计算语义分割辅助项。`H,W` 通常与 `masks` 一致；值为 **像素级类别 id**（含背景 0），类数不超过 YAML 的 `nc_seg` |

**对齐含义**：当前实现里，pose 与 seg 的 assigner 仍用 **同一组** `bboxes` + `batch_idx` 做目标分配。若你的数据里「姿态实例」与「检测框」不是一一对应，需要你在 **数据转换阶段** 为每条 pose 记录提供与之对齐的 `bboxes`（例如部件 tight box 或父物体框），否则 assigner 会把关键点配到错误的 anchor 上。

## 2. 三套类别 ID（推荐）

| 键 | 用途 | 形状 |
|----|------|------|
| `cls_det` | 检测头 `nc_det` 类 | `(N, 1)`，与 `bboxes` 行一一对应 |
| `cls_pose` | 姿态头 `nc_pose` 类 | 与 `keypoints` 的行对齐：同一 `batch_idx` 下，第 `i` 个 pose 实例应对应 `cls_pose` 的第 `i` 行（与 Ultralytics pose dataloader 堆叠方式一致） |
| `cls_seg` | 分割头 `nc_seg` 类 | 与实例 mask / `bboxes` 行对齐 |

若 **不提供** `cls_det` / `cls_pose` / `cls_seg`，三个分支都会使用同一个 **`cls`**（旧行为，仅当三任务类表可共享时安全）。

`E2EMultiTaskLoss` 对每个分支使用 **batch 的浅拷贝** 并写入对应 `cls`，避免分支间意外改写字典。

## 3. `masks` 与 `overlap_mask`

Ultralytics 的 `v8SegmentationLoss` 依赖 `model.args.overlap_mask`：

- **`overlap_mask=True`**：`masks` 常为 **`(B, max_instances, H, W)`**，实例用 **1-based** 整数 id 与 `target_gt_idx+1` 匹配。
- **`overlap_mask=False`**：`masks` 常为 **`(total_instances, H, W)`**，与 `batch_idx` 一起索引实例。

与官方 **segment** dataloader 保持一致即可；不要混用两种布局。

若暂时没有像素级语义标注，可用 **全零** `sem_masks`（表示全图背景）占位以满足张量键存在，但会引入无信息语义分支梯度，长期建议改为真实 `sem_masks` 或在上游为「无 semseg」路径加开关后再训练。

## 4. 与官方 `YOLODataset` 的差异

官方 `ultralytics/data/dataset.py` 中 **`use_segments` 与 `use_keypoints` 互斥**。多任务训练一般需要 **自定义 Dataset**，在一次 `__getitem__` 中同时返回 segments + keypoints，并在 `collate_fn` 里拼成上述 batch。

## 5. 辅助校验

Python 包 `ymt.batch` 提供 `validate_multitask_batch` 与 `assert_multitask_batch`，在训练循环前调用可减少维度/键名错误。
