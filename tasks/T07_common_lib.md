# T07 把通用计算收进 `common/`

- 执行：Codex　审查：Claude Code
- 状态：待做。
- 关联：`AGENTS.md` 的“目录约定”；`STATUS.md` 第 8 节

## 为什么做

同一套计算现在散在各个脚本里，各写各的：

- `cmi`：3 份；
- `cmi_with_null`：2 份；
- `Chain`：2 份，另有 `Backoff` 1 份；
- `atomic_json`：2 份；
- 另有约 12 个脚本各自写了 `.tmp` 再改名的逻辑。

第二部分的预测实验要用这些计算。如果继续复制，口径迟早会出现差异：β 怎样调、在哪批测试步上比、置换怎样分层。收进一处以后，新代码只 import，审查也只需要审一处。

## 问题

能不能把这些计算收成一个小模块，并且在已有的航空结果上**逐位复现**？

## 方法

- 新建 `common/memtest.py`。以 `airports/airport_memory.py` 为参考实现，它最新，也经过审查。收进以下函数：
  - `entropy_of_keys`、`cond_entropy`；
  - `encode(cols)`：安全的组合编码，每组合两列就用 `np.unique(..., return_inverse=True)` 重新编号，不会溢出 int64。它替代 `key()`；
  - `cmi_with_null(y, cond_cols, extra, perms, rng)`：在条件的每一层内置换，保持一阶结构；
  - `Backoff`、`tune`（β 按阶、在训练部分的内层划分上调）、`ce`；
  - `same_steps(mask_list)`：取各设置都有定义的那批测试步，返回索引；
  - 返程率：`return_model_ce`。
- 新建 `common/io.py`：
  - `atomic_write_json`、`atomic_write_csv`（先写 `.tmp` 再 `os.replace`）；
  - `ResumeCache(path, params)`：按键缓存结果，参数变了就拒绝复用并报错；
  - `environment()`：从 `prediction/resume.py` 移过来，`resume.py` 改成从这里 import。
- **旧脚本一律不改**（`controls/`、`prediction/` 投稿前冻结），只有两个例外：
  1. `airports/airport_memory.py` 改成从 `common/memtest.py` import，用来证明两者等价；
  2. `prediction/resume.py` 的 `environment()`。
- 新增 `tools/test_common.py`，用 numpy 生成小规模合成序列，固定种子：
  - `encode` 与旧 `key()` 在不溢出的范围内编号一一对应，CMI 逐位相同；
  - `cmi_with_null` 与旧实现在同一个种子下逐位相同；
  - `Backoff` + `tune` + `ce` 与旧实现逐位相同；
  - 运行时间 < 30 秒，不需要任何原始数据。
- 中断与续跑：本任务只有小测试和一次航空重跑，航空重跑沿用 `airport_memory.py` 已有的块缓存。

## 判据（跑之前写好）

- 通过：
  - `python tools/test_common.py` 全部通过；
  - 用改过的 `airports/airport_memory.py` 重跑 `airports/results/` 下的结果，每个文件逐位相同（逐个文件做新旧对比，写进实验记录）；
  - `python tools/check_paper_numbers.py` 全部一致。
- 不通过：任何一项不一致，就先查原因，不要改阈值或舍入去凑。查不清就把 `airport_memory.py` 退回原样，只保留 `common/` 和测试，在执行记录里写明差在哪。

## 产出

- `common/memtest.py`、`common/io.py`、`tools/test_common.py`；
- `common/README.md`：每个函数一行，写清输入、输出、单位；
- 实验记录、`EXPERIMENT_LOG.md` 总表的一行、`STATUS.md` 第 8 节。

## 审查要点

- 置换是不是在条件的每一层内做的（保持一阶结构），种子和置换顺序是否与旧实现一致；
- `tune` 的内层划分有没有用到测试数据；
- `encode` 在列数多、状态数大时是否真的不溢出（加一个故意会溢出旧 `key()` 的测试）；
- 新旧航空结果的逐文件对比是否真的逐位相同。

## 执行记录

## 审查记录
