# T03a 预测运行器与 GPU 环境

- 执行：Claude Code　审查：Codex
- 状态：待做
- 关联：`tasks/T03_prediction_design.md` 共同规则；依赖 T07（`common/`）

## 为什么做

T03b–T03g 约 120 次训练，要在云 GPU 和本机之间来回，必须能批量运行、断点续跑、统一汇总；同一套代码还是学位论文“系统设计与实现”一章的内容。没有它，后面每个任务都会各写一套脚本。

## 问题

能否用一份配置文件描述一个设置，一条命令批量跑完、中断后续跑，并从结果文件生成汇总表？

## 方法

1. 新建 `prediction/bench/`：
   - `run.py <配置目录或文件>…`：按配置逐个训练和评估；每个（设置，种子）是一个单元，输出 `prediction/bench/work/<任务>/<设置>/<种子>/`（检查点、`result.json`）；已有 `result.json` 的单元跳过；检查点按轮次原子写入，复用 `prediction/resume.py` 的做法（模型、优化器、早停计数、全部随机数状态）。
   - `summarise.py <任务>`：从 `result.json` 生成 `prediction/bench/results/<任务>.csv`（每行一个单元，另加每个设置的均值、最小值、最大值），记录参数量、推理时间、GPU 型号、PyTorch/CUDA 版本、代码提交号。
   - 配置用 YAML：数据集、输入组合、历史长度、模型、隐藏层宽度、种子列表、训练预算。种子按 `zlib.crc32(f'{任务}|{设置}|{r}'.encode())`。
   - `params.py`：给定模型和目标参数量，求隐藏层宽度，使各设置参数量相差 ≤ 5%。
2. 推理时间：同一 GPU、batch 固定（例如 256）、预热 10 次后取 100 次的中位数；CPU 单线程再测一次。
3. 环境：
   - `prediction/bench/env/requirements_cloud.txt`（云 GPU，Python 3.9、PyTorch 1.12 起，与 `mie-lab/location-prediction` 兼容）；本机仍用 `prediction/scripts/local_python.bat`。
   - `prediction/bench/CLOUD.md`：租用实例、上传数据（只传任务需要的文件；highD 用完删除）、`MEMFX_*` 环境变量、断点续跑、取回 `work/` 和 `results/` 的步骤。
4. `.gitignore` 加 `prediction/bench/work/`、`external/*/data/`、`external/*/outputs/`。
5. `external/location-prediction` 以 git submodule 固定在提交 `3417149`（MIT，保留许可证）。
6. 冒烟测试：用一个 10 秒能跑完的小配置（合成数据、2 轮），验证：中途杀掉进程再运行，结果与不中断一致（Transformer 逐位一致；LSTM 允许 `resume.py` 文件头说明的 cuDNN dropout 差异）。

## 判据（跑之前写好）

- 通过：冒烟测试中断续跑结果一致；`summarise.py` 输出的表包含上面列的全部字段；同一配置在本机 CPU 和云 GPU 上都能运行。
- 不通过：任一项不满足，记录原因，不开始 T03d–T03g。

## 产出

`prediction/bench/` 代码与 `CLOUD.md`；`prediction/README.md` 加一节；`experiment_log/<日期>_bench_runner.md`；`EXPERIMENT_LOG.md` 加一行；`STATUS.md` 第 8 节更新。

## 审查要点

- 续跑时是否真的恢复了全部随机数状态；跳过逻辑是否只认完整的 `result.json`（不认 `.tmp`）；
- 参数量对齐是否对每个设置都写进结果；
- 新的中间文件是否都在 `.gitignore` 里（仓库是公开的）。

## 执行记录

## 审查记录
