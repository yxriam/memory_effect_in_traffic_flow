# 2026-10-22 P0-5 阳性对照的结果；神经网络训练改为可断点续跑

## 目的

1. 检验历史长度消融能否检出长程依赖（待办清单 P0-5）：论文 §8.2 的“可用记忆止于三四步”成立的前提，是看 128 步的模型在确有长依赖时能学到它。
2. 按用户 10-22 的要求，让所有任务都能打断后继续、可复现。本次先改神经网络训练（唯一还会被打断的长任务），并把这条要求写进 `CLAUDE.md`。

## 数据

| 数据 | 位置 | 规模 / 版本 | 备注 |
|---|---|---|---|
| 合成序列 5 组 | `prediction/data/pc_L<L>_I<10×bits>.npz`（git 忽略）+ `.json`（入库） | 每组与 `taxi_traj20` 相同的 6,804 段、相同划分；测试集 155,033 个目标 | 由 `positive_control.py make` 生成（10-20 云端）；本机 `.json` 与仓库逐字节相同（md5 见 `.json`） |

植入：L = 8、16、32 各约 0.2 bits/步，L = 16 另有 0.1 和 0.3 bits/步（实际 0.102–0.303）；生成方法见 `positive_control.py` 开头和 2026-10-19 记录第二部分。

## 实验

| 命令 | 参数与种子 | 环境 | 耗时 |
|---|---|---|---|
| `prediction\scripts\run_positive_control.bat`（本机） | 5 组 × 种子 0–2 × {LSTM, Transformer} × {k = 4（`neural_hist.py`，每轮抽 3×10⁵ 个目标），128 步窗口（`neural_ch5.py`）}，其余参数为两个脚本的默认值 | Windows，Python 3.8.18，torch 1.10.0+cu102，GeForce MX250（`results/neural/env_check.txt`） | 10-20 21:00 – 10-21 04:03（依次运行；之前白天的并行运行因休眠和争用 GPU 几乎没有进展，见 10-20 记录） |
| `python prediction/positive_control.py summary`（云端，用本机 60 个结果 JSON） | — | 云端 Linux | 秒级 |

## 结果

60 个模型全部完成。测试交叉熵（bits/步，3 个种子的均值；“增益” = k = 4 减 128 步，正值表示 128 步更好）：

| 数据 | 植入 | 可得增益 | LSTM k = 4 | LSTM 128 | LSTM 增益（sd） | Transformer k = 4 | Transformer 128 | Transformer 增益（sd） | 生成器只看 4 步 |
|---|---|---|---|---|---|---|---|---|---|
| L = 8 | 0.204 | 0.202 | 2.898 | 2.908 | −0.010（0.003） | 2.906 | 3.117 | −0.212（0.008） | 2.582 |
| L = 16 | 0.102 | 0.101 | 2.871 | 2.877 | −0.007（0.006） | 2.871 | 3.088 | −0.217（0.005） | 2.556 |
| L = 16 | 0.204 | 0.203 | 2.880 | 2.883 | −0.003（0.010） | 2.876 | 3.096 | −0.220（0.013） | 2.571 |
| L = 16 | 0.303 | 0.300 | 2.851 | 2.862 | −0.011（0.003） | 2.849 | 3.080 | −0.231（0.002） | 2.551 |
| L = 32 | 0.217 | 0.216 | 2.915 | 2.923 | −0.008（0.008） | 2.917 | 3.134 | −0.217（0.007） | 2.600 |

- 两种网络都没有学到植入的依赖：128 步 LSTM 比 4 步差 0.003–0.011 bits，128 步 Transformer 差 0.21–0.23 bits（平均 11 轮就早停），而可得增益是 0.10–0.30 bits。回收率（增益 / 可得增益）LSTM −0.02 至 −0.06，Transformer −0.8 至 −2.1。
- 连只看 4 步的模型都比“生成器只看 4 步”的交叉熵高 0.30–0.32 bits：在这个数据量（约 78 万个符号、400 个状态）下，网络连短程结构都没学全，更学不到稀疏的长程交互。与 10-20 的 CPU 检查一致（显式以 L + 1 步前格子为条件的计数模型也只拿到 −0.008 至 +0.024 bits）。
- 结果文件：`prediction/results/neural/positive_control/*.json`（60 个，本机训练）、`prediction/results/ngram/positive_control_summary.csv`、`positive_control_runs.csv`；图 `controls/figures/output/fig_s_positive_control.pdf`（论文附录 B 图 17），快速版 `prediction/figures/output/fig_positive_control.*`。

论文数字核对：`python tools/check_paper_numbers.py` → `summary: {'match': 293}`（新增 P0-5 的 11 项）。

## 结论

按清单的判据（“若检不出，消融实验对长记忆没有判别力，该结论必须删除或改进训练后重做”）：检不出。论文 §8.2 删去“三四步之外没有可用记忆”，改为“模型在三四步之外利用不到信息，但阳性对照表明在这个数据量下历史长度消融检不出 0.1–0.3 bits/步的长依赖，所以对几步之外的记忆不下结论”。摘要、引言贡献、讨论第二点、局限性同步修改，附录 B 加一段说明和图 17。“改进训练后重做”需要多得多的数据或专门的模型，不在本文范围。

## 问题与改动

1. 真实出租车数据上的 Transformer 历史长度消融（同一批脚本的第二部分）10-21 04:38 停在 k = 16 的第 10 轮：之后没有输出、日志文件不再被占用，说明窗口已不在运行（可能是夜间重启或关机）。已完成 k = 1、2、3、4、8；剩 k = 16、32 和种子 1、2 的 k = 4、128 步共 6 个。用户重新运行 `run_positive_control.bat` 即可续跑。
2. 断点续跑（用户要求）：新增 `prediction/resume.py`。`neural_hist.py`、`neural_ch5.py` 每轮结束后把模型、优化器、早停状态、历史、已用时间和所有随机数状态（torch CPU / CUDA、numpy 全局和 Generator）原子写入 `<tag>.resume.pt`；重新启动时若参数相同就从下一轮继续，随机数序列与不中断时相同；写完结果 JSON 后删除续跑文件。结果 JSON 新增 `env`（Python、torch、CUDA、cuDNN、显卡）。不中断时训练与以前完全相同。
   - 云端没有 PyTorch（PyPI 和 download.pytorch.org 都被拒绝），用一个替身 torch 模块验证了逻辑：连续中断两次再续跑，与不中断的结果逐位相同。
   - 本机 GPU 上的验证：新增 `prediction/scripts/test_resume.bat`（短 LSTM：一次跑完 vs 第 2 轮后中断再续跑，比较测试交叉熵），`run_positive_control.bat` 第一次运行时自动执行，结果写到 `results/neural/resume_test/RESULT.txt`。
   - 兼容 torch 1.10（没有 `weights_only` 参数时自动退回旧调用）。
   - **本机 GPU 测试结果（10-22 00:30，`results/neural/resume_test/`）**：`A 4.876345  B 4.878325  diff 2.0e-03  PASS`。中断前的两轮 A 与 B 逐位相同（GPU 计算在这里是确定的）；续跑后第 3 轮的训练损失相差约 1×10⁻⁴。原因是 cuDNN 的 LSTM 层间 dropout 在进程内部自己保存随机状态（首次使用时播种），无法存盘，新进程会重新播种。所以 LSTM 续跑后与不中断时差约 0.002 bits/步（种子间差异约 0.01）；Transformer 没有这种隐藏状态，续跑应与不中断时一致。已写进 `resume.py` 的说明。
   - 用户 00:30 重新运行后，Transformer 消融 k = 16 已从头开始，并在每轮后写续跑文件（`hist_transformer_taxi_traj20_k16_s0.resume.pt`）。
3. `CLAUDE.md`“约定”新增：每个任务都必须能打断后继续、可复现。
4. `controls/figures/fig_supplement_checks.py` 新增 `fig_s_positive_control`；`tools/check_paper_numbers.py` 第 5 组加 P0-5 的 11 项。
5. 论文（本机 `D:\desktop\paper\paper_latex_v2`）：修改前的版本另存为 `main_v3_before_1022.tex`、`main_zh_v3_before_1022.tex`；改动见 `修改说明_v3.md` 第六节。

## 下一步

- [x] 本机重新运行 `prediction\scripts\run_positive_control.bat`（10-22 00:30 已启动，续跑测试 PASS）。
- [x] 等剩下的 6 个 Transformer 消融跑完和自动汇总（10-22 02:06 完成，见下面第二部分）。
- [x] 把真实数据上的 Transformer 历史长度结果补进论文表 4 和 §8.2（10-22）。
- [ ] 其余同 `EXPERIMENT_LOG.md`“当前的下一步”。


---

# 第二部分：真实出租车数据上的 Transformer 历史长度消融（10-22 00:30 – 02:06）

## 目的

清单 P0-5 的第二项：用 Transformer 重复真实数据上的历史长度消融（原来只有 LSTM），并补齐 128 步 Transformer 的另外两个种子。

## 数据

`prediction/data/taxi_traj20.npz`（6,804 段、782,034 个符号、400 个状态；与论文所用相同，元数据见 `taxi_traj20.json`）。

## 实验

| 命令 | 参数与种子 | 环境 | 时间 |
|---|---|---|---|
| `run_positive_control.bat` 的第 2 步 `lane_transformer_ablation.bat` | `neural_hist.py --model transformer --hist k`，k = 1、2、3、4、8、16、32（种子 0），k = 4 另跑种子 1、2；`neural_ch5.py --model transformer` 种子 1、2（种子 0 是 10-17 的已有结果） | Windows 10.0.26100，Python 3.8.18，torch 1.10.0+cu102，numpy 1.24.3，CUDA 10.2，cuDNN 7605，GeForce MX250（结果 JSON 的 `env`） | k = 1–8 在 10-20 跑完；10-21 04:38 中断后，10-22 00:30 重启：k = 16 00:56，k = 32 01:39，k = 4 种子 1 01:44，128 步种子 1 01:50，k = 4 种子 2 01:58，128 步种子 2 02:05；随后自动运行两个汇总脚本 |

k = 16 是重启后从头训练的（中断时用的是旧代码，没有续跑文件）；之后的运行都带续跑文件，没有再中断。

## 结果

测试交叉熵（bits/步），`results/ngram/ch5_hist_ablation_transformer.csv` 和 `results/neural/neural_taxi_traj20_transformer{,_s1,_s2}.json`：

| k | 1 | 2 | 3 | 4 | 8 | 16 | 32 | 128 步 |
|---|---|---|---|---|---|---|---|---|
| Transformer | 4.442 | 3.047 | 2.830 | 2.779 / 2.844 / 2.781（均值 2.801） | 2.769 | 2.790 | 2.820 | 3.246 / 3.194 / 3.214（均值 3.218） |
| LSTM（已有） | 4.455 | 2.973 | 2.866 | 2.839 | 2.849 | — | — | 2.904 / 2.919 / 2.896（均值 2.906） |

- Transformer 与 LSTM 的图景相同：增益几乎都在前 3–4 步；k = 8、16、32 与 k = 4 的差别（−0.03 到 +0.02）都在 k = 4 三个种子的范围（2.78–2.84）之内。
- 看完整 128 步的 Transformer 只有 3.22 bits，比受限版本差 0.4 bits；与阳性对照中 128 步 Transformer 学不好（比 4 步差 0.21–0.23 bits、约 11 轮就早停）一致。

汇总文件的新旧对比（本机 02:06 自动重新生成）：

| 文件 | 旧 | 新 | 说明 |
|---|---|---|---|
| `results/ngram/ch5_hist_ablation.csv`（LSTM 消融） | 已提交版本 | 本机重新生成 | 去掉 Windows 换行符（CRLF）后逐字节相同；云端用同样的 JSON 重算也相同 |
| `results/neural/ch5_neural_summary.csv` | 已提交版本 | 本机重新生成 | 同上，内容相同（汇总只用每个模型的种子 0 文件） |
| `results/ngram/positive_control_summary.csv`、`_runs.csv` | 10-22 云端用同样 60 个 JSON 算出 | 本机重新生成 | 去掉 CRLF 后逐字节相同 |
| `results/ngram/ch5_hist_ablation_transformer.csv` | — | 新文件 | 云端用同样的 JSON 重算，内容相同；仓库存 LF 版本 |

论文数字核对：`python tools/check_paper_numbers.py` → `summary: {'match': 304}`（新增 Transformer 消融 11 项）。

## 结论

Transformer 的历史长度消融复现了 LSTM 的结论：可用的信息集中在前 3–4 步，再往后的差别小于随机种子之间的波动。结合第一部分的阳性对照，这只说明网络用不到几步之外的信息，不能说明那里没有信息。

## 问题与改动

1. 论文：表 4 加“出租车，Transformer”一行，表注说明 k = 4 为三个种子的均值、k = 16 为 2.79 bits；§8.2 补一句 Transformer 的结果（英文、中文两版）。修改前的版本另存为 `main_v3_before_1022b.tex`、`main_zh_v3_before_1022b.tex`；改动见 `修改说明_v3.md` 第七节。
2. `tools/check_paper_numbers.py`：第 5 组加 Transformer 消融 11 项。
3. 本机 pandas 在 Windows 上写出的 CSV 用 CRLF 换行，与仓库里的 LF 版本只差换行符；不影响内容，推送时由 git 处理。

## 下一步

- [x] P0-5 全部完成（阳性对照 + Transformer 消融），待办清单 P0-5 打勾。
- [ ] 其余同 `EXPERIMENT_LOG.md`“当前的下一步”。
