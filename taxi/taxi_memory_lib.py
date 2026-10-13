"""
taxi_memory_lib.py —— 出租车轨迹记忆深度检验的计算核心
(由 taxi_memory_analysis.ipynb 自动生成；单独放在文件里是为了让 multiprocessing 在 Windows 上能 import 到 worker 函数)

每条轨迹 -> 网格状态序列 -> (可选) run-length 压缩 -> 按时间 80/20 划分
  - 一/二/三阶马尔科夫模型的测试集 per-step 对数似然
  - 二阶 CMI  I(S_{t+1}; S_{t-1} | S_t)        (置换零模型, 偏差校正)
  - 三阶 CMI  I(S_{t+1}; S_{t-2} | S_{t-1}, S_t)
"""
import os
import zlib
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------- 数据与状态化
def read_lat_lon(file_path):
    """读取 "id,time,lon,lat,..." 格式的 GPS 文件，返回 (lat, lon) 列表。"""
    lat, lon = [], []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                cols = line.strip().split(",")
                if len(cols) >= 4:
                    try:
                        lon.append(float(cols[2]))
                        lat.append(float(cols[3]))
                    except ValueError:
                        continue
    except (FileNotFoundError, UnicodeDecodeError):
        return [], []
    return lat, lon


def scan_global_bounds(file_paths, lo_q=0.5, hi_q=99.5):
    """全局网格的经纬度边界：取所有点的 0.5%-99.5% 分位数，避免 GPS 漂移点把网格拉大。"""
    lats, lons = [], []
    for fp in file_paths:
        la, lo = read_lat_lon(fp)
        if la:
            lats.append(np.asarray(la))
            lons.append(np.asarray(lo))
    lat, lon = np.concatenate(lats), np.concatenate(lons)
    return (float(np.percentile(lat, lo_q)), float(np.percentile(lat, hi_q)),
            float(np.percentile(lon, lo_q)), float(np.percentile(lon, hi_q)))


def map_to_states(lat_list, lon_list, grid_size, bounds=None):
    """经纬度 -> g×g 网格编号。bounds=None: 每条轨迹按自身包围盒归一化; 否则用全局固定边界。"""
    if not lat_list:
        return []
    if bounds is None:
        a, b, c, d = min(lat_list), max(lat_list), min(lon_list), max(lon_list)
    else:
        a, b, c, d = bounds
    if abs(b - a) < 1e-9 or abs(d - c) < 1e-9:
        return [0] * len(lat_list)
    states = []
    for la, lo in zip(lat_list, lon_list):
        r = min(max(int((la - a) / (b - a) * grid_size), 0), grid_size - 1)
        co = min(max(int((lo - c) / (d - c) * grid_size), 0), grid_size - 1)
        states.append(r * grid_size + co)
    return states


def compress_repeats(states):
    """run-length 压缩：合并连续重复状态，剥离停留。"""
    out = states[:1]
    for s in states[1:]:
        if s != out[-1]:
            out.append(s)
    return out


def stable_seed(file_path):
    """只依赖文件名的种子（内置 hash() 受 PYTHONHASHSEED 影响，跨进程不可复现）。"""
    return zlib.crc32(os.path.basename(file_path).encode("utf-8")) & 0xFFFFFFFF


# ---------------------------------------------------------------- 马尔科夫模型
def fit_context_counts(states, order):
    """order 阶上下文 -> 下一状态计数。"""
    counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(states) - order):
        counts[tuple(states[i:i + order])][states[i + order]] += 1
    return counts


def _smoothed(d, nxt, n_states, alpha, beta, smoothing, lower_p):
    """上下文计数 d 下 nxt 的概率。
    additive: (c + alpha) / (n + alpha*|S|)        —— 与早期结果一致；|S| 很大时先验极其分散
    backoff : (c + beta*p_lower) / (n + beta)       —— 向低阶分布收缩，样本少时自动退化为低阶模型
    d 为 None 表示该上下文在训练集中未出现：additive 退化为均匀分布，backoff 退化为低阶分布。
    """
    if d is None:
        return 1.0 / n_states if smoothing == "additive" else lower_p
    total = sum(d.values())
    if smoothing == "additive":
        return (d.get(nxt, 0) + alpha) / (total + alpha * n_states)
    return (d.get(nxt, 0) + beta * lower_p) / (total + beta)


def test_logliks(test, train, M1, M2, M3, n_states, alpha, beta=1.0, smoothing="additive"):
    """一/二/三阶模型在测试集上的总对数似然。
    三个阶的模型用**同一种**平滑：additive 时每阶独立 add-alpha；backoff 时 P0=均匀 -> P1 -> P2 -> P3 逐级收缩。
    （若一阶用 add-alpha、高阶用 backoff，ΔLL 会把"一阶先验过于分散"误记为高阶收益——见 notebook 第 3 节。）"""
    ll1 = ll2 = ll3 = 0.0
    p0 = 1.0 / n_states
    p2_, p1_, curr = train[-3], train[-2], train[-1]
    for s in test:
        p1 = _smoothed(M1.get((curr,)), s, n_states, alpha, beta, smoothing, p0)
        p2 = _smoothed(M2.get((p1_, curr)), s, n_states, alpha, beta, smoothing, p1) if (p1_, curr) in M2 else p1
        p3 = _smoothed(M3.get((p2_, p1_, curr)), s, n_states, alpha, beta, smoothing, p2) if (p2_, p1_, curr) in M3 else p2
        ll1 += np.log(p1)
        ll2 += np.log(p2)
        ll3 += np.log(p3)
        p2_, p1_, curr = p1_, curr, s
    return ll1, ll2, ll3, len(test)


# ---------------------------------------------------------------- 条件互信息 + 置换检验
def _H(codes, n):
    """整数编码序列的 plug-in 熵。"""
    counts = np.unique(codes, return_counts=True)[1]
    p = counts / n
    return -float(np.sum(p * np.log(p)))


def _perm_within_strata(x, strata, rng):
    """在 strata 相同的位置之间置换 x（保持 P(x|strata) 的边际结构）。"""
    order = np.argsort(strata, kind="stable")
    xs = x[order].copy()
    ss = strata[order]
    bnd = np.concatenate(([0], np.flatnonzero(np.diff(ss)) + 1, [len(ss)]))
    for a, b in zip(bnd[:-1], bnd[1:]):
        if b - a > 1:
            xs[a:b] = rng.permutation(xs[a:b])
    out = np.empty_like(x)
    out[order] = xs
    return out


def cmi_perm(states, order, n_perm, rng_seed):
    """order=2: I(S_{t+1}; S_{t-1} | S_t);  order=3: I(S_{t+1}; S_{t-2} | S_{t-1}, S_t).
    返回 real / excess(=real - 置换均值, 偏差校正) / p_value；序列过短返回 None。"""
    min_len = 30 if order == 2 else 50
    arr = np.asarray(states, dtype=np.int64)
    if len(arr) < min_len:
        return None
    lag = order - 1
    n = len(arr) - order
    hist = arr[:n]                                   # 被检验的远历史 (S_{t-1} 或 S_{t-2})
    nxt = arr[order:]                                # S_{t+1}
    K = int(arr.max()) + 1
    if order == 2:
        cond = arr[1:n + 1]                          # S_t
    else:
        cond = arr[1:n + 1] * K + arr[2:n + 2]       # (S_{t-1}, S_t) 联合编码
    Kc = int(cond.max()) + 1

    H_c = _H(cond, n)
    H_cn = _H(cond * K + nxt, n)

    def cmi(h):
        return _H(h * Kc + cond, n) + H_cn - H_c - _H((h * Kc + cond) * K + nxt, n)

    real = cmi(hist)
    rng = np.random.default_rng(rng_seed)
    null = np.array([cmi(_perm_within_strata(hist, cond, rng)) for _ in range(n_perm)])
    return {"real": float(real), "excess": float(real - null.mean()),
            "p_value": float((np.sum(null >= real) + 1) / (n_perm + 1))}


# ---------------------------------------------------------------- back-off 收缩强度 beta 的敏感性与内部验证调参
def _ll_orders(test, train, n_states, beta):
    """给定 beta 的 back-off 链 (P0 均匀 -> P1 -> P2 -> P3) 在 test 上的一/二/三阶 LL/step。"""
    M1, M2, M3 = (fit_context_counts(train, k) for k in (1, 2, 3))
    ll1, ll2, ll3, n = test_logliks(test, train, M1, M2, M3, n_states, 0.0, beta, "backoff")
    return ll1 / n, ll2 / n, ll3 / n


def backoff_sweep(args):
    """对一条轨迹：
    (1) beta 敏感性：对每个 beta 直接在测试段上算三阶 LL（用于画 ΔLL-beta 曲线）；
    (2) 调参：把训练段再按 75/25 切开，在内部验证段上为每个阶各选使其 LL 最大的 beta，
        然后用完整训练段重新拟合、在测试段上评估。这样 beta 的选择不看测试数据，ΔLL 是真正的样本外量。"""
    file_path, grid_size, do_compression, bounds, betas = args
    lat, lon = read_lat_lon(file_path)
    states = map_to_states(lat, lon, grid_size, bounds)
    if len(states) < 30:
        return None
    if do_compression:
        states = compress_repeats(states)
    if len(states) < 50:
        return None
    n_states = grid_size * grid_size
    cutoff = int(len(states) * 0.8)
    train, test = states[:cutoff], states[cutoff:]
    inner = int(len(train) * 0.75)
    tr_in, va_in = train[:inner], train[inner:]
    out = {"grid": grid_size, "compressed": do_compression, "file": os.path.basename(file_path), "comp_len": len(states)}
    val = {b: _ll_orders(va_in, tr_in, n_states, b) for b in betas}
    tst = {b: _ll_orders(test, train, n_states, b) for b in betas}
    for b in betas:
        out[f"ll1_b{b:g}"], out[f"ll2_b{b:g}"], out[f"ll3_b{b:g}"] = tst[b]
        out[f"val1_b{b:g}"], out[f"val2_b{b:g}"], out[f"val3_b{b:g}"] = val[b]       # 内部验证段 LL，供全局选 beta 用
    best = [max(betas, key=lambda b: val[b][k]) for k in range(3)]      # 每个阶各自的最优 beta（逐轨迹选）
    out["beta1"], out["beta2"], out["beta3"] = best
    out["ll1_tuned"], out["ll2_tuned"], out["ll3_tuned"] = tst[best[0]][0], tst[best[1]][1], tst[best[2]][2]
    out["delta_ll_2v1_tuned"] = out["ll2_tuned"] - out["ll1_tuned"]
    out["delta_ll_3v2_tuned"] = out["ll3_tuned"] - out["ll2_tuned"]
    return out


# ---------------------------------------------------------------- 单文件分析 (multiprocessing worker)
def analyze_file(args):
    """一条轨迹 × 一种 (网格, 压缩) 配置。两种平滑的 LL 在同一次遍历里一起算出（CMI 置换是主要开销，LL 几乎免费）。"""
    file_path, grid_size, alpha, do_compression, n_perm, bounds, beta = args
    lat, lon = read_lat_lon(file_path)
    states = map_to_states(lat, lon, grid_size, bounds)
    if len(states) < 30:
        return None
    raw_len = len(states)
    if do_compression:
        states = compress_repeats(states)
    if len(states) < 50:
        return None
    cutoff = int(len(states) * 0.8)
    train, test = states[:cutoff], states[cutoff:]
    n_states = grid_size * grid_size
    M1, M2, M3 = (fit_context_counts(train, k) for k in (1, 2, 3))
    ll1, ll2, ll3, n_test = test_logliks(test, train, M1, M2, M3, n_states, alpha, beta, "additive")
    ll1b, ll2b, ll3b, _ = test_logliks(test, train, M1, M2, M3, n_states, alpha, beta, "backoff")
    seed = stable_seed(file_path)
    c2 = cmi_perm(states, 2, n_perm, seed)
    c3 = cmi_perm(states, 3, n_perm, seed + 1)
    nan = float("nan")
    return {
        "grid": grid_size, "compressed": do_compression, "file": os.path.basename(file_path),
        "raw_len": raw_len, "comp_len": len(states),
        # add-alpha 平滑
        "ll1_per_step": ll1 / n_test, "ll2_per_step": ll2 / n_test, "ll3_per_step": ll3 / n_test,
        "delta_ll_2v1": (ll2 - ll1) / n_test, "delta_ll_3v1": (ll3 - ll1) / n_test, "delta_ll_3v2": (ll3 - ll2) / n_test,
        # back-off 平滑（一阶也用 back-off，三阶模型平滑方式一致）
        "ll1_bo_per_step": ll1b / n_test, "ll2_bo_per_step": ll2b / n_test, "ll3_bo_per_step": ll3b / n_test,
        "delta_ll_2v1_bo": (ll2b - ll1b) / n_test, "delta_ll_3v1_bo": (ll3b - ll1b) / n_test, "delta_ll_3v2_bo": (ll3b - ll2b) / n_test,
        # 条件互信息
        "cmi2_real": c2["real"] if c2 else nan, "cmi2_excess": c2["excess"] if c2 else nan, "cmi2_p": c2["p_value"] if c2 else nan,
        "cmi3_real": c3["real"] if c3 else nan, "cmi3_excess": c3["excess"] if c3 else nan, "cmi3_p": c3["p_value"] if c3 else nan,
    }