# T03b highD 标准数据管线与基线复现（关卡）

- 执行：Codex　审查：Claude Code
- 状态：待做
- 关联：`tasks/T03_prediction_design.md`；依赖 T03a

## 为什么做

假设一在 highD 上的结论要能和文献对比，前提是数据管线和评估与文献一致。复现不了文献数字，后面的比较就没有参照。

## 问题

按 PiP（Song 等，ECCV 2020）的 highD 流程，自己的管线能否复现文献报告的 V-LSTM 和 CS-LSTM 的 RMSE（相差 ≤ 10%）？

## 方法

1. 数据：`common/paths.py` 的 `highd_full`（原始发布版，60 段录像）。
2. 预处理：按 PiP 仓库 `preprocess/preprocess_highD.m` 的逻辑用 Python 重写为 `prediction/bench/highd_std.py`（PiP 仓库没有许可证文件，只参考不复制，文件头注明出处）：
   - 3 s 历史、5 s 未来，5 Hz（核对 PiP 对 highD 先降采样再取 `t_h=30, t_f=50, d_s=2` 的具体做法）；只保留历史 ≥ 3 s、未来 ≥ 5 s 的样本；
   - 邻车网格 25 × 5（每格 8 × 7 英尺，与 PiP 相同）；机动标签（横向：保持、左换道、右换道；纵向：正常、制动）；
   - 训练、验证、测试的划分与 PiP 相同（读脚本确认是按录像还是按车辆划分，写进记录）。
   - 输出到 `prediction/bench/work/highd_std/`（不进仓库），记录各划分的样本数，并与 PiP 论文或其公开的处理后数据对照（若用户下载了 PiP 提供的 `.mat`，逐项比较样本数）。
3. 基线：
   - 不需训练：匀速、匀加速（用历史最后 0.4 s 差分得到速度、加速度）。
   - V-LSTM：编码器—解码器 LSTM，只用自车历史。
   - CS-LSTM：用 `nachiket92/conv-social-pooling` 的模型代码（先确认许可证；没有许可证就按论文重写）。
   - 各 3 个种子，经 T03a 运行器。
4. 指标：1、2、3、4、5 s 的 RMSE（m）；ADE、FDE（5 s）。分车道保持和换道两类另报。
5. 目标数字：从 PiP 论文的 highD 表中读出 V-LSTM、CS-LSTM 在 1–5 s 的 RMSE，写进执行记录；C2F-TP（arXiv 2412.13231）表 1 汇总的 CS-LSTM 为 0.24 / 1.26 / 3.31 m（1 / 3 / 5 s），作交叉核对。

## 判据（跑之前写好）

- 通过：V-LSTM 和 CS-LSTM 在 1–5 s 的 RMSE 与 PiP 报告的值相对差都 ≤ 10%；匀速、匀加速的误差随时刻单调上升且明显高于两个网络。
- 部分通过：只有 4–5 s 超出 10%：检查坐标单位（英尺/米）、降采样和划分后再判；仍超出就记录差异，由用户决定是否继续。
- 不通过：1–3 s 就超出 10%：管线有问题，不进入 T03d。

## 产出

`prediction/bench/highd_std.py`、配置 `prediction/bench/configs/highd_baseline/`、`prediction/bench/results/highd_baseline.csv`；`experiment_log/<日期>_highd_baseline.md`；`EXPERIMENT_LOG.md` 一行；`STATUS.md` 第 8 节。

## 审查要点

- 坐标单位和方向：highD 原始是米、行驶方向两个，PiP 是否做了方向统一和单位换算；
- 划分是否与 PiP 一致，测试集车辆是否出现在训练集；
- 目标数字是否确实出自 PiP 论文的 highD 表（不是 NGSIM 表）；
- RMSE 的定义（PiP 用的是每个时刻的 RMSE，不是 ADE）。

## 执行记录

## 审查记录
