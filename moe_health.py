#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
moe-healthcheck —— MoE 模型健康度体检工具

给一个训练好（或训练中）的 MoE 模型做 5 项无监督体检，
不需要任何标注数据，也不需要跑完整评测集。

五项指标
────────
  ① 残差流范数增长    健康 2~10×；>30 报警；>100 判失败
  ② 注意力集中度      单 head 最大权重应 ≤0.9（=1.0 表示 head 完全退化）
  ③ 内容 head 分布    深层（最后 1/4）应至少达到浅层的一半
  ④ 专家权重分布      n 个专家的权重范数应大致均匀；两簇分离说明容量分配不均
  ⑤ 路由卡死层数      连续 N 步中始终选同一专家的层数；≥10 说明隐藏态已退化

为什么是这五项
──────────────
这五项分别对应五个已知失效模式（按发现顺序）：

  · 残差爆炸      —— 信号在层间被放大而非精炼，靠最后的归一化硬救
  · head 退化     —— 某个 head 的注意力退化为常数，1/H 的容量被浪费
  · 深层空转      —— 深层不做语义工作，只做 attention sink
  · 专家容量失衡  —— 部分专家权重幅度只有另一半的 50%
  · 路由卡死      —— 隐藏态退化成近似常量，router 输入不变 ⇒ 固定选同一专家

★ 第 ⑤ 项是本文档的核心贡献：它是「隐藏态是否健康」的零成本代理指标，
  只需一次前向 + 记录每层 top-1 专家，不需要人工判读输出质量。

用法
────
  # 默认三项检查、三个内置提示词
  python moe_health.py path/to/model.tg

  # 自定义提示词 + JSON 输出
  python moe_health.py model.tg --prompts "你好" "写个快排" --json out.json

  # 只看某一项
  python moe_health.py model.tg --only routing

退出码：0=PASS  1=WARN  2=FAIL

依赖
────
  · Python 3.9+
  · numpy
  · tgai_go（TGAI 引擎 Python 绑定，含 libtgai_engine DLL）
    ★ 引擎必须支持 tg_dump_set_enabled / tg_expert_set_disabled
      （2026-10-18 之后的版本）

已知限制
────────
  见 docs/LIMITATIONS.md。最重要的一条：
  · 这些指标衡量的是「内部表示是否健康」，不是「输出质量」
  · 残差 ×600 + 深层全 sink 的模型仍可能把简单问题答对
"""

import os
import sys
import json
import struct
import argparse
import tempfile
import shutil

__version__ = '0.1.0'

# 让同目录的 tg_weights.py 可被导入（开源项目自带，无需 pip install）
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_DUMP_DIR = None


# ══════════════════════════════════════════════════════════════════
# 阈值（可按模型规模调整；下面这套是在 3B / 28 层 / 8 专家上校准的）
# ══════════════════════════════════════════════════════════════════
THRESHOLDS = {
    # ① 残差范数增长倍率（峰值 / 首层）
    'norm_growth_warn': 30.0,
    'norm_growth_fail': 100.0,
    # ② 单 head 最大权重
    'max_weight_warn': 0.95,
    'max_weight_fail': 0.995,
    # ③ 深层内容 head / 浅层内容 head
    'deep_content_ratio_warn': 0.5,
    'deep_content_ratio_fail': 0.2,
    # ④ 专家权重范数变异系数
    'expert_cv_warn': 0.15,
    'expert_cv_fail': 0.30,
    # ⑤ 路由卡死层数
    'stuck_layers_warn': 5,
    'stuck_layers_fail': 10,
    # 判定用的子阈值
    'sharp_entropy': 0.30,      # 归一化熵 ≤ 此值算「尖锐 head」
    'content_entropy': 0.50,    # 算「内容 head」时放宽到 0.50
}

# 模板标记 / 结构 token —— Top-1 落在这些上不算「看内容」
META_TOKENS = {
    '<BOS>', '<EOS>', '<PAD>', '<UNK>',
    '<用户>', '<TGAI>', '<回答>', '<思考>', '<系统>',
    '\n', '\r\n', '\r', ' ', '\t',
}

DEFAULT_PROMPTS = [
    '什么是线性代数？',
    '请用三句话介绍你自己。',
    '中国的首都是哪里？',
]

ARCH_NAMES = {0: 'Dense', 1: 'MoE', 2: 'Mamba', 3: 'Hybrid'}
DTYPE_NAMES = {0: 'FP32', 1: 'FP16', 2: 'INT8', 3: 'INT32',
               4: 'Q8_0', 5: 'Q4_0', 6: 'BF16', 7: 'Q4K'}


# ══════════════════════════════════════════════════════════════════
# 环境与文件头
# ══════════════════════════════════════════════════════════════════
def setup_env(dump_dir):
    """★ 必须在 import 引擎【之前】调用。

    Windows 上 MSVC CRT 在 DLL 加载时复制环境块，
    之后再改 os.environ 传不进 DLL（已实测）。
    """
    global _DUMP_DIR
    _DUMP_DIR = dump_dir
    os.makedirs(dump_dir, exist_ok=True)
    os.environ['TGAI_DUMP_DIR'] = dump_dir
    os.environ['TGAI_DUMP_EXTRA'] = '1'   # 需要 c_partend / c_attnout
    os.environ['TGAI_DUMP_ATTN'] = '1'    # 需要 c_attn


def read_header(path):
    """解析 .tg 128 字节文件头（小端）

      0x00  16×u32 : magic, version, weight_dtype, n_layers, d_model,
                     n_heads, d_ff, vocab_size, max_seq_len, n_experts,
                     n_activated, rope_theta, pad, bos, eos, unk
      0x40   6×u64 : tokenizer_off/len, weight_table_off/count,
                     weight_data_off/len
      0x70   4×u32 : ★ 实测未被写入（0,0,1,0xFFFFFEEA），不要读

    ★ 架构从 n_experts 推导，不读 +112 处的 arch_type —— 那个槽位是空的。
    """
    out = {}
    try:
        with open(path, 'rb') as fh:
            hdr = fh.read(128)
        if len(hdr) < 128:
            return out
        names = ['magic', 'version', 'weight_dtype', 'n_layers', 'd_model',
                 'n_heads', 'd_ff', 'vocab_size', 'max_seq_len', 'n_experts',
                 'n_activated', 'rope_theta', 'pad', 'bos', 'eos', 'unk']
        for k, v in zip(names, struct.unpack_from('<16I', hdr, 0)):
            out[k] = int(v)
    except Exception:
        pass
    return out


# ══════════════════════════════════════════════════════════════════
# 后端：TGAI .tg + tgai_go
# ══════════════════════════════════════════════════════════════════
class TGABackend:
    """TGAI .tg 模型后端

    对外暴露：
      .cfg           dict，含 n_layers / d_model / n_heads / n_experts ...
      .encode(s)     str -> [int]
      .decode(i)     int -> str
      .forward(ids)  跑一次前向，dump 落到 _DUMP_DIR
      .clear_dump()  清空 dump 目录
      .set_dump(on)  开/关 dump（引擎级，低开销）
    """

    name = 'tgai_go / .tg'

    def __init__(self, path):
        try:
            from tgai_go import _engine
        except ImportError:
            raise RuntimeError(
                '需要 tgai_go。安装方式：\n'
                '  pip install -e "D:\\tgai_nlp\\TGAI GO\\python"')
        import ctypes
        self._ct = ctypes
        self.eng = _engine.TGEngine()
        self.path = path
        self.cfg = self.eng.load(path)
        # 引擎支持运行时开关就打开（老版本静默失败）
        try:
            self.eng.lib.tg_dump_set_enabled(1)
        except Exception:
            pass

    def encode(self, s):
        return self.eng.encode(s, add_bos=True)

    def decode(self, i):
        return self.eng.decode(i, detok=True)

    def forward(self, ids):
        ct = self._ct
        n = len(ids)
        cache = self.eng.lib.tg_kvcache_new(self.eng.model, max(256, n + 8))
        if not cache:
            raise RuntimeError('kvcache 创建失败')
        try:
            arr = (ct.c_int32 * n)(*ids)
            lp = self.eng.lib.tg_forward(self.eng.model, arr, n, cache, 0)
            if not lp:
                raise RuntimeError('forward 失败')
            return lp
        finally:
            self.eng.lib.tg_kvcache_free(cache)

    def clear_dump(self):
        if not _DUMP_DIR or not os.path.isdir(_DUMP_DIR):
            return
        for f in os.listdir(_DUMP_DIR):
            if f.endswith('.bin'):
                try:
                    os.remove(os.path.join(_DUMP_DIR, f))
                except OSError:
                    pass

    def set_dump(self, on):
        try:
            self.eng.lib.tg_dump_set_enabled(1 if on else 0)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
# 指标 ①②③：残差范数 + 注意力
# ══════════════════════════════════════════════════════════════════
def _norm_last_tok(path, n_tok, d):
    """读 dump 里最后一个 token 的隐藏态范数（seek，不读整份）"""
    import numpy as np
    try:
        sz = os.path.getsize(path)
    except OSError:
        return None
    if sz != n_tok * d * 4:
        return None
    with open(path, 'rb') as fh:
        fh.seek(sz - d * 4)
        buf = fh.read(d * 4)
    return float(np.linalg.norm(np.frombuffer(buf, dtype=np.float32)))


def collect(backend, prompt_text, dump_dir):
    """跑一次前向，采集残差范数 + 逐 head 注意力"""
    import numpy as np

    nl = int(backend.cfg['n_layers'])
    d = int(backend.cfg['d_model'])
    nh = int(backend.cfg['n_heads'])

    ids = backend.encode(prompt_text)
    n = len(ids)
    toks = [backend.decode(i) for i in ids]

    backend.clear_dump()
    backend.forward(ids)

    norms = []
    for tag in ['c_embed'] + ['c_block%d' % i for i in range(nl)]:
        v = _norm_last_tok(os.path.join(dump_dir, tag + '.bin'), n, d)
        norms.append(v if v is not None else float('nan'))

    heads = []
    log_n = float(np.log(max(2, n)))
    for L in range(nl):
        p = os.path.join(dump_dir, 'c_attn%d.bin' % L)
        if not os.path.exists(p):
            continue
        a = np.fromfile(p, dtype=np.float32)
        if a.size != nh * n:
            continue
        A = a.reshape(nh, n)
        for h in range(nh):
            row = A[h].astype(np.float64)
            s = row.sum()
            if s <= 0:
                continue
            pr = np.clip(row / s, 1e-12, None)
            H = float(-(pr * np.log(pr)).sum()) / log_n
            ti = int(np.argmax(row))
            heads.append([L, h, round(H, 4), ti, round(float(row[ti]), 4)])

    return {'norms': norms, 'heads': heads, 'tokens': toks, 'n_tok': n}


def analyze_norm_attn(sample, nl):
    import numpy as np

    norms = np.array(sample['norms'], dtype=np.float64)
    ok = np.isfinite(norms)
    valid = norms[ok]
    if valid.size == 0:
        return None

    base = float(valid[0]) if valid[0] > 1e-9 else 1.0
    heads = sample['heads']
    toks = sample['tokens']

    Hs = np.array([r[2] for r in heads]) if heads else np.array([])
    sharp = [r for r in heads if r[2] <= THRESHOLDS['sharp_entropy']]

    def is_content(r):
        ti = r[3]
        t = toks[ti] if 0 <= ti < len(toks) else ''
        if t in META_TOKENS or not t.strip():
            return False
        return r[2] <= THRESHOLDS['content_entropy']

    content = [r for r in heads if is_content(r)]
    quarter = max(1, nl // 4)
    shallow = [r for r in content if r[0] < quarter]
    deep = [r for r in content if r[0] >= nl - quarter]

    per_layer = []
    for L in range(nl):
        lh = [r for r in heads if r[0] == L]
        if not lh:
            per_layer.append((L, 0, 0, None))
            continue
        per_layer.append((L,
                          len([r for r in lh if r[2] <= THRESHOLDS['sharp_entropy']]),
                          sum(1 for r in lh if r[3] == 0),
                          min(lh, key=lambda r: r[2])))

    return {
        'norm_base': base,
        'norm_peak': float(valid.max()),
        'norm_peak_layer': int(np.argmax(np.where(ok, norms, -1e30))) - 1,
        'norm_final': float(valid[-1]),
        'norm_growth': float(valid.max()) / base,
        'entropy_mean': float(Hs.mean()) if Hs.size else float('nan'),
        'entropy_p10': float(np.percentile(Hs, 10)) if Hs.size else float('nan'),
        'sharp_count': len(sharp),
        'sharp_ratio': (len(sharp) / len(heads)) if heads else 0.0,
        'content_count': len(content),
        'content_shallow': len(shallow),
        'content_deep': len(deep),
        'max_weight': max((r[4] for r in heads), default=0.0),
        'content_top': sorted(content, key=lambda r: r[2])[:6],
        'per_layer': per_layer,
        'tokens': toks,
        'n_tok': sample['n_tok'],
        'n_head_slots': len(heads),
    }


# ══════════════════════════════════════════════════════════════════
# 指标 ④：专家权重范数
# ══════════════════════════════════════════════════════════════════
EXPERT_KEYS = ('w1', 'w2', 'w3')


def metric_expert_norm(weight_getter, nl, ne):
    """算计每个专家的权重范数，判断是否均匀

    weight_getter(layer, expert, key) -> ndarray 或 None
    返回 dict：per_expert（各专家三矩阵范数之和）、cv、bimodal
    """
    import numpy as np

    sums = {e: [] for e in range(ne)}
    for L in range(nl):
        for e in range(ne):
            tot = 0.0
            got = False
            for k in EXPERT_KEYS:
                a = weight_getter(L, e, k)
                if a is None:
                    continue
                tot += float(np.linalg.norm(a.astype(np.float64)))
                got = True
            if got:
                sums[e].append(tot)

    if not any(sums.values()):
        return None

    per = {e: float(np.mean(v)) if v else float('nan') for e, v in sums.items()}
    vals = np.array([per[e] for e in range(ne)], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    mean = float(vals.mean())
    cv = float(vals.std() / mean) if mean > 1e-9 else float('nan')

    # 两簇检测：排序后找最大间隔
    s = np.sort(vals)
    gaps = np.diff(s)
    gap_ratio = 0.0
    split_at = -1
    if gaps.size:
        i = int(np.argmax(gaps))
        lo, hi = s[:i + 1], s[i + 1:]
        if hi.mean() > 1e-9:
            gap_ratio = 1.0 - float(lo.mean() / hi.mean())
            split_at = i + 1

    return {
        'per_expert': {int(e): (round(per[e], 2) if np.isfinite(per[e]) else None)
                       for e in range(ne)},
        'mean': round(mean, 2),
        'cv': round(cv, 4),
        'gap_ratio': round(gap_ratio, 4),
        'split_at': split_at,
        'n': int(vals.size),
    }


# ══════════════════════════════════════════════════════════════════
# 指标 ⑤：路由卡死层数（核心）
# ══════════════════════════════════════════════════════════════════
def metric_routing(backend, prompt_text, dump_dir, n_steps=8, temp=0.7, top_k=40,
                   seed=42, router_w=None, ln2_w=None, qb=None):
    """逐 token 生成，记录每层 top-1 专家，算「卡死层数」

    定义
    ────
    对每一层 L，收集 n_steps 步里 router 选出的 top-1 专家序列。
    若该序列只有 1 个不同值 ⇒ 这一层「卡死」。

    为什么这个指标有用
    ──────────────────
    router 的输入是 ln2(h)，h = 上一层残差 + attn_out。
    隐藏态健康时，不同 token 的 ln2(h) 差异明显 ⇒ 路由分数差异大 ⇒ 每步换专家。
    隐藏态退化（变成近似常量）⇒ 路由分数几乎不变 ⇒ 固定选同一个 ⇒ 卡死。

    ⇒ 「卡死层数」是隐藏态健康度的零成本代理指标
      不需要人工判读输出，纯数值、可自动化、可直接做训练监控。

    参考实测（TGAI NB V2.7，28 层，8 专家）
    ──────────────────────────────────────
      · 正常：卡死 0 / 28，层熵 1.099，确信度 0.756
      · 禁 E0（必需专家）：卡死 13 / 28，层熵 0.414，确信度 0.692
      · 禁 E7（非必需）：卡死 2 / 28，层熵 0.832，确信度 0.760
        ⇒ 后者路由正常但专家错配 ⇒ 另一种失效模式
    """
    import numpy as np

    nl = int(backend.cfg['n_layers'])
    d = int(backend.cfg['d_model'])
    ne = int(backend.cfg['n_experts'])
    na = int(backend.cfg['n_activated'])
    vocab = int(backend.cfg['vocab_size'])
    eos = int(backend.cfg.get('eos', 3))

    if router_w is None or ln2_w is None:
        return None

    rng = np.random.RandomState(seed)
    ids = backend.encode(prompt_text)
    n0 = len(ids)

    backend.clear_dump()
    lp = backend.forward(ids)
    lg = np.ctypeslib.as_array(
        backend._ct.cast(lp, backend._ct.POINTER(backend._ct.c_float)),
        shape=(vocab,)).astype(np.float64).copy()

    def route_of(kv_len):
        """读当前 dump 算每层 top-1/top-2 专家与权重"""
        prev = np.fromfile(os.path.join(dump_dir, 'c_embed.bin'),
                           dtype=np.float32).reshape(kv_len, d)[-1]
        out = []
        for L in range(nl):
            at = np.fromfile(os.path.join(dump_dir, 'c_attnout%d.bin' % L),
                             dtype=np.float32).reshape(kv_len, d)[-1]
            h = (prev.astype(np.float64) + at.astype(np.float64)).astype(np.float32)
            v = h.astype(np.float64)
            rms = np.sqrt(np.mean(v * v) + 1e-6)
            ln2 = (v / rms) * ln2_w[L].astype(np.float64)
            r = ln2 @ router_w[L].T
            if qb[L] is not None:
                r = r + qb[L].astype(np.float64)
            srt = np.argsort(-r)[:na]
            tv = r[srt]
            ex = np.exp(tv - tv.max())
            tw = ex / ex.sum()
            out.append((int(srt[0]), float(tw[0])))
            prev = np.fromfile(os.path.join(dump_dir, 'c_partend%d.bin' % L),
                               dtype=np.float32).reshape(kv_len, d)[-1]
        return out

    steps = []
    text = []
    pos = n0
    cache = backend.eng.lib.tg_kvcache_new(backend.eng.model, max(256, n0 + n_steps + 8))
    arr = (backend._ct.c_int32 * n0)(*ids)
    try:
        lp = backend.eng.lib.tg_forward(backend.eng.model, arr, n0, cache, 0)
        lg = np.ctypeslib.as_array(
            backend._ct.cast(lp, backend._ct.POINTER(backend._ct.c_float)),
            shape=(vocab,)).astype(np.float64).copy()

        for st in range(n_steps):
            steps.append(route_of(n0 if st == 0 else 1))
            l2 = lg / temp
            idx = np.argsort(-l2)[:top_k]
            c = l2[idx] - l2[idx].max()
            p = np.exp(c); p /= p.sum()
            nx = int(idx[rng.choice(len(idx), p=p)])
            if nx == eos:
                break
            text.append(backend.decode(nx))
            backend.clear_dump()
            b = (backend._ct.c_int32 * 1)(nx)
            lp = backend.eng.lib.tg_forward(backend.eng.model, b, 1, cache, pos)
            if not lp:
                break
            lg = np.ctypeslib.as_array(
                backend._ct.cast(lp, backend._ct.POINTER(backend._ct.c_float)),
                shape=(vocab,)).astype(np.float64).copy()
            pos += 1
    finally:
        backend.eng.lib.tg_kvcache_free(cache)

    if not steps:
        return None

    per_layer = [[s[L][0] for s in steps] for L in range(nl)]
    stuck = sum(1 for seq in per_layer if len(set(seq)) == 1)

    ents = []
    for seq in per_layer:
        cnt = np.bincount(seq, minlength=ne).astype(float)
        cnt /= cnt.sum()
        pr = cnt[cnt > 0]
        ents.append(float(-(pr * np.log(pr)).sum()))

    conf = [s[L][1] for s in steps for L in range(nl)]
    allpick = [p for seq in per_layer for p in seq]
    use = np.bincount(allpick, minlength=ne).astype(float)
    use /= max(1, use.sum())

    return {
        'n_steps': len(steps),
        'stuck_layers': int(stuck),
        'layer_entropy_mean': round(float(np.mean(ents)), 4),
        'conf_mean': round(float(np.mean(conf)), 4),
        'conf_hi_ratio': round(float(np.mean(np.array(conf) >= 0.9)), 3),
        'expert_usage': [round(float(x), 4) for x in use],
        'text': ''.join(text)[:120],
    }


# ══════════════════════════════════════════════════════════════════
# 判定
# ══════════════════════════════════════════════════════════════════
def verdict_norm(a):
    T = THRESHOLDS
    g = a['norm_growth']
    if g >= T['norm_growth_fail']:
        return 'FAIL', '残差增长 x%.0f >= %.0f' % (g, T['norm_growth_fail'])
    if g >= T['norm_growth_warn']:
        return 'WARN', '残差增长 x%.0f >= %.0f' % (g, T['norm_growth_warn'])
    return 'PASS', '残差增长 x%.1f' % g


def verdict_attention(a):
    T = THRESHOLDS
    mw = a['max_weight']
    if mw >= T['max_weight_fail']:
        return 'FAIL', '单 head 最大权重 %.3f >= %.3f（head 退化）' % (mw, T['max_weight_fail'])
    if mw >= T['max_weight_warn']:
        return 'WARN', '单 head 最大权重 %.3f >= %.3f' % (mw, T['max_weight_warn'])
    return 'PASS', '单 head 最大权重 %.3f' % mw


def verdict_content(a):
    """深层 vs 浅层的内容 head 数量对比

    ★ 浅层为 0 时不能算百分比（会得到 99900% 这种无意义数字）。
      这种情况本身说明"整个模型几乎没有内容 head"，应当 WARN 而不是 PASS。
    """
    T = THRESHOLDS
    sh, dp = a['content_shallow'], a['content_deep']
    if sh == 0 and dp == 0:
        return 'WARN', '全模型无内容 head（浅 0 / 深 0）—— 提示词可能过短或模型未学到内容关注'
    if sh == 0:
        return 'WARN', '浅层无内容 head，深层有 %d 个 —— 分布异常' % dp
    ratio = dp / sh
    if ratio < T['deep_content_ratio_fail']:
        return 'FAIL', '深层内容 head 仅浅层的 %.0f%%（浅%d/深%d）' % (ratio * 100, sh, dp)
    if ratio < T['deep_content_ratio_warn']:
        return 'WARN', '深层内容 head 仅浅层的 %.0f%%（浅%d/深%d）' % (ratio * 100, sh, dp)
    return 'PASS', '深层内容 head 为浅层的 %.0f%%（浅%d/深%d）' % (ratio * 100, sh, dp)


def verdict_expert(m):
    T = THRESHOLDS
    cv = m['cv']
    if cv >= T['expert_cv_fail']:
        return 'FAIL', '专家权重 CV=%.3f >= %.3f（容量分配严重不均）' % (cv, T['expert_cv_fail'])
    if cv >= T['expert_cv_warn']:
        return 'WARN', '专家权重 CV=%.3f >= %.3f' % (cv, T['expert_cv_warn'])
    return 'PASS', '专家权重 CV=%.3f' % cv


def verdict_routing(r):
    T = THRESHOLDS
    s = r['stuck_layers']
    if s >= T['stuck_layers_fail']:
        return 'FAIL', '路由卡死 %d 层 >= %d（隐藏态已退化）' % (s, T['stuck_layers_fail'])
    if s >= T['stuck_layers_warn']:
        return 'WARN', '路由卡死 %d 层 >= %d' % (s, T['stuck_layers_warn'])
    return 'PASS', '路由卡死 %d 层' % s


# ══════════════════════════════════════════════════════════════════
# 报告
# ══════════════════════════════════════════════════════════════════
def print_report(results, backend, path, hdr):
    W = 78
    cfg = backend.cfg
    nl = int(cfg['n_layers'])
    nh = int(cfg['n_heads'])
    order = {'PASS': 0, 'WARN': 1, 'FAIL': 2}
    worst = 'PASS'

    print('=' * W)
    print('  MoE 健康度体检报告')
    print('=' * W)
    print('  模型      : %s' % path)
    if hdr:
        print('  格式      : .tg v%d  权重 %s' % (
            hdr.get('version', 0), DTYPE_NAMES.get(hdr.get('weight_dtype', -1), '?')))
    print('  层/维度/头: %d / %d / %d   head_dim=%d' % (
        nl, int(cfg['d_model']), nh, int(cfg['d_model']) // nh))
    print('  专家      : %d 个，激活 %d 个   d_ff=%d' % (
        int(cfg['n_experts']), int(cfg['n_activated']), int(cfg['d_ff'])))
    print('  词表/上下文: %d / %d   rope_theta=%d' % (
        int(cfg['vocab_size']), int(cfg['max_seq']), int(cfg['rope_theta'])))
    try:
        print('  文件大小  : %.2f GB' % (os.path.getsize(path) / (1024 ** 3)))
    except OSError:
        pass
    print()

    for name, sample_tag, payload in results:
        print('-' * W)
        print('%s   [%s]' % (name, sample_tag))
        print('-' * W)
        if payload is None:
            print('  (无数据)')
            print()
            continue
        for label, lv, msg in payload:
            if order[lv] > order[worst]:
                worst = lv
            mark = {'PASS': '[OK] ', 'WARN': '[!]  ', 'FAIL': '[X]  '}[lv]
            print('  %s %-8s %s' % (mark, lv, msg))
        print()

    print('=' * W)
    print('  ★ 总评：%s' % worst)
    print('=' * W)
    print()
    print('  判读提示：')
    print('    · 残差增长：健康 2~10x，>30x 报警，>100x 判失败')
    print('    · head 退化：单 head 最大权重 =1.0 表示该 head 完全退化为常数')
    print('    · 深层空转：深层内容 head 过少 = 后半段层数在语义上空转')
    print('    · 专家失衡：权重范数两簇分离 = 容量分配不均')
    print('    · 路由卡死：≥10 层始终选同一专家 = 隐藏态已退化为近似常量')
    print()
    print('  ★ 重要：这些指标衡量「内部表示是否健康」，不是「输出质量」。')
    print('    残差 x600 + 深层全 sink 的模型仍可能把简单问题答对。')
    print('=' * W)
    return worst


# ══════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description='MoE 模型健康度体检：残差范数 / 注意力 / 专家容量 / 路由卡死',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', help='.tg 模型路径')
    ap.add_argument('--prompts', nargs='+', default=None, help='自定义提示词')
    ap.add_argument('--json', dest='json_out', default=None, help='结果写 JSON')
    ap.add_argument('--dump-dir', default=None, help='dump 临时目录')
    ap.add_argument('--keep-dump', action='store_true', help='保留 dump')
    ap.add_argument('--only', default=None,
                    choices=['norm', 'attention', 'content', 'expert', 'routing'],
                    help='只跑某一项')
    ap.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    args = ap.parse_args()

    model = os.path.abspath(args.model)
    if not os.path.isfile(model):
        print('  [X] 模型不存在: %s' % model, file=sys.stderr)
        return 2

    dump_dir = args.dump_dir or tempfile.mkdtemp(prefix='moe_health_')
    setup_env(dump_dir)          # ★ 必须在 import 引擎前

    print('  加载引擎与模型…', flush=True)
    try:
        backend = TGABackend(model)
    except Exception as e:
        print('  [X] 加载失败: %s' % e, file=sys.stderr)
        return 2

    hdr = read_header(model)
    cfg = backend.cfg
    nl, ne = int(cfg['n_layers']), int(cfg['n_experts'])
    prompts = args.prompts or DEFAULT_PROMPTS

    results = []
    json_out = {'model': model, 'header': hdr,
                'n_layers': nl, 'n_experts': ne,
                'thresholds': THRESHOLDS, 'sections': {}}

    want = args.only

    # ── ①②③ 残差 / 注意力 / 内容 head
    if want in (None, 'norm', 'attention', 'content'):
        for msg in prompts[:1]:     # 这几项只跑第一个提示词，省时间
            full = '<用户>\n' + msg + '<TGAI><回答>\n'
            print('  采集(前向): %s' % msg[:30], flush=True)
            sample = collect(backend, full, dump_dir)
            a = analyze_norm_attn(sample, nl)
            if a is None:
                continue
            payload = []
            if want in (None, 'norm'):
                payload.append(('指标①',) + verdict_norm(a))
            if want in (None, 'attention'):
                payload.append(('指标②',) + verdict_attention(a))
            if want in (None, 'content'):
                payload.append(('指标③',) + verdict_content(a))
            results.append(('残差 / 注意力', msg[:18], payload))
            json_out['sections']['norm_attn'] = a

    # ── ④ 专家权重范数
    if want in (None, 'expert') and ne > 1:
        print('  采集(专家权重)…', flush=True)
        try:
            import numpy as np
            import tg_weights
            Wobj = tg_weights.TGWeights(model)

            def getter(L, e, k):
                nm = 'blocks.%d.moe.experts.%d.%s.weight' % (L, e, k)
                if not Wobj.has(nm):
                    return None
                return np.asarray(Wobj.get(nm))
            m = metric_expert_norm(getter, nl, ne)
        except Exception as e:
            print('  [!] 专家权重读取失败（需要 tg_weights）: %s' % e, file=sys.stderr)
            m = None
        if m:
            lv, msg = verdict_expert(m)
            results.append(('专家容量', 'norm', [('指标④', lv, msg)]))
            json_out['sections']['expert_norm'] = m

    # ── ⑤ 路由卡死
    if want in (None, 'routing') and ne > 1:
        print('  采集(路由卡死)…', flush=True)
        try:
            import numpy as np
            import tg_weights
            Wobj = tg_weights.TGWeights(model)
            RW = [Wobj.get('blocks.%d.moe.router.weight' % L).astype(np.float64)
                  for L in range(nl)]
            LN2 = [Wobj.get('blocks.%d.ln2.weight' % L) for L in range(nl)]
            QB = [Wobj.get('blocks.%d.moe._qb_bias' % L) for L in range(nl)]
            full = '<用户>\n' + prompts[0] + '<TGAI><回答>\n'
            r = metric_routing(backend, full, dump_dir,
                               router_w=RW, ln2_w=LN2, qb=QB)
        except Exception as e:
            print('  [!] 路由指标失败: %s' % e, file=sys.stderr)
            r = None
        if r:
            lv, msg = verdict_routing(r)
            results.append(('路由健康', prompts[0][:18], [('指标⑤', lv, msg)]))
            json_out['sections']['routing'] = r

    worst = print_report(results, backend, model, hdr)

    if args.json_out:
        json_out['worst'] = worst
        od = os.path.dirname(os.path.abspath(args.json_out))
        if od:
            os.makedirs(od, exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump(json_out, fh, ensure_ascii=False, indent=2, default=str)
        print('  [JSON] %s' % args.json_out)

    backend.set_dump(0)
    if not args.keep_dump:
        shutil.rmtree(dump_dir, ignore_errors=True)

    return {'PASS': 0, 'WARN': 1, 'FAIL': 2}[worst]


if __name__ == '__main__':
    sys.exit(main())
