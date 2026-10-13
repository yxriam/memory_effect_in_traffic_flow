# highD 记忆效应实验记录

记录日期: 2026-05-28  
工作目录: `highD/`（原机器上的本地路径已移除）  
主脚本: `highd_memory_analysis.py`  
结果目录: `memory_analysis_output_8.0`

## 1. 文件整理

运行前已将旧实验文件移动到 `archive_old_markov/`，避免旧的准确率/返回数实验与当前 highD 似然/BIC/LRT/VACF 主线混用。当前顶层仅保留:

- `data/`: highD `*_tracks.csv` 原始轨迹文件，共 60 个。
- `highd_memory_analysis.py`: 当前主线实验脚本。
- `memory_analysis_output_8.0/`: 当前实验输出目录。

旧文件归档内容包括 `memory_analysis_output/`、`combined_markov_results.csv`、`simulation_return_counts.csv`、`markov_accuracy_log.txt`、两个旧训练/验证 CSV 和旧 notebook。

## 2. 运行环境

命令:

```powershell
python highd_memory_analysis.py
```

Python 环境检查:

- `pandas`: 2.2.3
- `numpy`: 2.1.3
- `matplotlib`: 3.9.3
- `scipy`: 当前环境未安装

由于 `scipy` 不可用，脚本使用以下 fallback:

- LRT 的 chi-square survival probability 使用 Wilson-Hilferty 正态近似。
- Wilcoxon signed-rank 使用带平均秩 tie 处理的正态近似。
- 记忆核指数拟合使用 NumPy tau-grid least squares。

## 3. 执行过程

第一次完整运行在方法 C 打印 `R²` 时触发 Windows 控制台 GBK 编码错误:

```text
UnicodeEncodeError: 'gbk' codec can't encode character '\xb2'
```

该错误只发生在终端打印阶段，不是计算错误。随后将终端输出中的 `R²` 改为 ASCII 的 `R^2`，并重新运行完整脚本。

正式运行开始时间为脚本记录的 `2026-05-28T13:57:36.164230`，运行成功，退出码为 0。第二次完整运行耗时约 559.5 秒。

## 4. 实验参数

`memory_analysis_output_8.0/analysis_config.json` 记录了本次运行参数:

```json
{
  "velocity_mode": "signed_longitudinal_velocity",
  "raw_dt_sec": 0.04,
  "downsample": 5,
  "dt_eff_sec": 0.2,
  "n_states": 20,
  "max_order": 3,
  "min_frames_raw": 50,
  "n_splits": 5,
  "train_frac": 0.8,
  "laplace_alpha": 1.0,
  "vacf_max_lag_sec": 8.0,
  "n_track_files": 60,
  "method_a_quantile_edges_fit_on_train_only": true,
  "method_b_lrt_uses_full_sample_for_inference_not_validation": true
}
```

重要说明: 当前实验使用的是有向纵向速度 `signed_longitudinal_velocity`，即中心差分 `x` 坐标并保留正负号。因此论文表述应写为“有向纵向速度”或“纵向速度过程”。若要分析 speed magnitude，需要改为 `abs(v)` 后重新运行全部实验。

## 5. 数据规模

脚本读取 `data/*_tracks.csv` 共 60 个文件。

有效轨迹与步数:

- 轨迹数: 110093
- 5 倍降采样后总帧数: 7926563
- 典型序列长度中位数: 67
- 有效时间步长: 0.20 s

## 6. 方法 A: Markov 似然 / BIC

设置:

- 状态数: 20 个全局等频 quantile 状态
- 比较阶数: k = 0, 1, 2, 3
- 5 次随机 train/val 划分
- train fraction = 0.8
- Laplace smoothing alpha = 1.0
- quantile edges 仅在训练集拟合，再应用到验证集

汇总结果:

| k | 有效参数量 | train LL/step | val LL/step | val perplexity | BIC |
|---:|---:|---:|---:|---:|---:|
| 0 | 19.0 | -2.9952 +/- 0.0002 | -2.9953 +/- 0.0002 | 19.9910 | 37.9756e6 |
| 1 | 380.0 | -0.3972 +/- 0.0007 | -0.3965 +/- 0.0014 | 1.4866 | 4.9718e6 |
| 2 | 2755.0 | -0.2934 +/- 0.0005 | -0.2931 +/- 0.0009 | 1.3406 | 3.6595e6 |
| 3 | 10579.2 | -0.2643 +/- 0.0004 | -0.2639 +/- 0.0005 | 1.3020 | 3.3767e6 |

验证集 per-step LL 提升:

- 0 -> 1: +2.5988 nats
- 1 -> 2: +0.1034 nats
- 2 -> 3: +0.0292 nats

配对显著性检验基于 split 0 的 per-sequence validation LL:

| from k | to k | mean delta | p-value | method |
|---:|---:|---:|---:|---|
| 0 | 1 | +2.5947 | 0.0 | normal_approx_average_ranks |
| 1 | 2 | +0.1054 | 0.0 | normal_approx_average_ranks |
| 2 | 3 | +0.0287 | 0.0 | normal_approx_average_ranks |

解释: 验证集 LL 随 k 单调改善，BIC 到 k=3 仍继续下降，说明一阶 Markov 近似不足。

## 7. 方法 B: 似然比检验

检验目标:

- H0: 一阶 Markov 足够，即 `P(s[t+1] | s[t], s[t-1]) = P(s[t+1] | s[t])`
- H1: 需要二阶历史

结果:

```json
{
  "G_statistic": 1078644.3419492925,
  "ll_1order": -2795987.2698721383,
  "ll_2order": -2256665.098897492,
  "n_steps_compared": 7706377,
  "delta_ll_per_step": 0.06998388100850066,
  "df_empirical": 2451,
  "df_nominal": 7220,
  "p_value_empirical_df": 0.0,
  "p_value_nominal_df": 0.0
}
```

解释: G 统计量远大于经验自由度和名义自由度下的常规临界值，拒绝一阶 Markov 零假设。每步平均信息增益约 +0.0700 nats。

## 8. 方法 C: VACF 与 GLE 记忆核

设置:

- 输入: 有向纵向速度序列
- dt_eff = 0.20 s
- 最大滞后时间 = 8.0 s
- 每条轨迹先减去自身均值，再做 ensemble VACF
- 使用 Volterra 离散反演重建记忆核

核心结果:

```json
{
  "tau_C_sec": 4.2,
  "C0_var": 1.634140235320143,
  "gamma_inst": 0.3521603629317808,
  "gamma_mem": 0.14548075242379915,
  "gamma_total": 0.49764111535558,
  "memory_fraction": 0.29234070082784186
}
```

记忆核指数拟合:

- 单指数: `A = -11.2347`, `tau = 0.0891 s`, `c = 0.0542`, `R^2 = 0.9979`
- 双指数: `R^2 = 0.9985`，但参数表现为相近时间尺度下的补偿，解释上仍以单一约 89 ms 时标为主。

解释:

- VACF 衰减到 1/e 的相关时间约 4.2 s。
- 瞬时摩擦 `gamma_inst = 0.3522 1/s`。
- 非瞬时记忆贡献 `gamma_mem = 0.1455 1/s`。
- 记忆贡献占总摩擦约 29.2%，不能近似为纯 delta 型无记忆核。

## 9. 输出文件

当前 `memory_analysis_output_8.0/` 包含:

- `analysis_config.json`: 参数与数据使用说明。
- `markov_likelihood_results.csv`: 5 次 split 的逐阶 Markov 结果。
- `markov_likelihood_summary.csv`: Markov 汇总表。
- `markov_significance_results.csv`: per-sequence 显著性检验结果。
- `markov_significance_results.json`: 显著性检验 JSON 版本。
- `lrt_test_result.json`: LRT 结果。
- `vacf_kernel_data.npz`: VACF 与 K(t) 数组。
- `vacf_kernel_summary.json`: VACF 与记忆核汇总。
- `vacf_and_kernel.png`: VACF/K(t)/Kmem 合并图。
- `fig1_likelihood_vs_order.png`: 论文图 1。
- `fig2_vacf.png`: 论文图 2。
- `fig3_memory_kernel.png`: 论文图 3。
- `fig4_bic_complexity.png`: 论文图 4。

图像文件已检查为非空 PNG。

## 10. 数据泄露与复现性检查

已确认:

- 方法 A 按车辆轨迹序列划分 train/val，不在同一条序列内部切分。
- 方法 A 的 quantile edges 只在训练集拟合，再用于验证集。
- 方法 A 验证集 LL 未参与模型拟合。
- 方法 B 是全样本统计推断，不是样本外预测评估，已在配置中明确。
- 方法 C 是无监督物理统计量估计，不涉及训练/验证泄露。
- 旧准确率/返回数实验已归档，不再参与当前论文 A.1 主线结果。

注意:

- 当前输出不包含个人身份字段，只包含聚合统计、模型指标和图像。
- 原始 `data/*_tracks.csv` 仍保留在本地工作区，用于复现实验；不要随论文或公开仓库发布原始轨迹数据，除非确认 highD 数据许可允许。

## 11. 结论

三种方法给出一致证据:

1. Markov 似然/BIC: k=2 相对 k=1 的验证集 per-step LL 提升约 +0.1034 nats，k=3 仍继续提升约 +0.0292 nats，BIC 未在 k=3 前饱和。
2. LRT: 一阶 Markov 零假设被强烈拒绝，`G = 1.0786e6`，`delta_ll_per_step = +0.0700 nats`。
3. VACF/GLE: 记忆核包含非瞬时尾部，非瞬时记忆贡献约占总摩擦 29.2%，主时标约 89 ms。

因此，在当前有向纵向速度定义下，highD 高速公路车辆速度过程表现出显著的非 Markov 记忆效应。

