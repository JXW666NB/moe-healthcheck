# -*- coding: utf-8 -*-
"""从 .TG 里读权重（BF16/FP32）+ 精确重算 MoE 路由

为什么能精确重算：
  引擎 dump 出了 c_block{i-1}（该层输入）
  我再用 .TG 里的 ln2_w 做 rmsnorm → 就是引擎内部的 work_ln
  再乘 router_w 加 qb_bias → 就是引擎内部的 router logits
  top-2 选择完全一致

引擎源码依据（tgai_engine.c）：
  L3843: rmsnorm(work_ln, mh, ln2_w, seq, d, 1e-6f)
  L3882: matmul(work_router, work_ln, router_w, seq, d, ne)
  L3905: qb_bias 在 top-k 之前加到 logits
  L3920: top-k 选择排序
  L3931: softmax over top-k
"""
import os
import struct
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

DTYPE_FP32, DTYPE_FP16, DTYPE_INT8, DTYPE_INT32 = 0, 1, 2, 3
DTYPE_Q8_0, DTYPE_Q4_0, DTYPE_BF16, DTYPE_Q4K = 4, 5, 6, 7

TG_MAGIC = 0x49414754
TG_HEADER_SIZE = 128
TG_WEIGHT_ENTRY_SIZE = 52


def fnv1a64(name):
    h = 0xcbf29ce484222325
    for b in name.encode('utf-8'):
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def bf16_to_f32(raw_bytes):
    """BF16 → FP32：左移 16 位重解释"""
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


class TGWeights:
    """只读 .TG 的权重，按名字取（带 LRU 缓存）"""

    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        self._read_header()
        self._read_index()
        self._cache = {}

    def _read_header(self):
        f = self.f
        f.seek(0)
        buf = f.read(TG_HEADER_SIZE)
        (self.magic, self.version, self.weight_dtype, self.n_layers,
         self.d_model, self.n_heads, self.d_ff, self.vocab_size,
         self.max_seq_len, self.n_experts, self.n_activated,
         self.rope_theta, self.pad_id, self.bos_id, self.eos_id,
         self.unk_id) = struct.unpack_from('<16I', buf, 0)
        (self.tokenizer_off, self.tokenizer_len,
         self.weight_table_off, self.weight_table_count,
         self.weight_data_off, self.weight_data_len) = struct.unpack_from('<6Q', buf, 64)
        (self.arch_type, self.num_kv_heads,
         self.modality_flags, self.extension_off) = struct.unpack_from('<4I', buf, 112)
        if self.magic != TG_MAGIC:
            raise ValueError('bad magic')
        if self.num_kv_heads == 0:
            self.num_kv_heads = self.n_heads

    def _read_index(self):
        self.weights = {}
        f = self.f
        f.seek(self.weight_table_off)
        raw = f.read(self.weight_table_count * TG_WEIGHT_ENTRY_SIZE)
        for i in range(self.weight_table_count):
            off = i * TG_WEIGHT_ENTRY_SIZE
            h, woff, ln, dt, ndim = struct.unpack_from('<QQQII', raw, off)
            shape = struct.unpack_from('<4I', raw, off + 32)
            self.weights[h] = {
                'offset': woff, 'length': ln, 'dtype': dt,
                'shape': tuple(shape[:ndim]),
            }

    def has(self, name):
        return fnv1a64(name) in self.weights

    def get(self, name, cache=True):
        """取权重 → np.float32 数组（按 shape 还原）"""
        if cache and name in self._cache:
            return self._cache[name]
        e = self.weights.get(fnv1a64(name))
        if e is None:
            return None
        self.f.seek(self.weight_data_off + e['offset'])
        raw = self.f.read(e['length'])
        dt = e['dtype']
        if dt == DTYPE_BF16:
            a = bf16_to_f32(raw)
        elif dt == DTYPE_FP32:
            a = np.frombuffer(raw, dtype=np.float32)
        elif dt == DTYPE_FP16:
            a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        else:
            raise NotImplementedError('dtype %d 暂不支持' % dt)
        a = a.reshape(e['shape']).astype(np.float32)
        if cache:
            # 只缓存小权重（norm/bias/router），大矩阵不缓存
            if a.size <= 1048576:
                self._cache[name] = a
        return a


def rmsnorm(x, w, eps=1e-6):
    """与引擎 L2633 一致：x / sqrt(mean(x^2) + eps) * weight"""
    x = x.astype(np.float64)
    ss = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(ss + eps) * w).astype(np.float32)


def moe_route(W, layer, h_in):
    """精确重算该层的 MoE 路由

    h_in : [d_model] 该层输入（引擎 dump 的 c_block{i-1}，或 c_embed 当 i=0）
    返回 : dict(logits[8], top_ids, top_weights, gate_ln)
    """
    ln2 = W.get('blocks.%d.ln2.weight' % layer)
    rw = W.get('blocks.%d.moe.router.weight' % layer)
    qb = W.get('blocks.%d.moe._qb_bias' % layer)
    if ln2 is None or rw is None:
        return None

    ln = rmsnorm(h_in, ln2)                  # [d]
    logits = (rw @ ln).astype(np.float64)     # [8]   rw 是 [8, d]
    if qb is not None:
        logits = logits + qb.astype(np.float64)   # ★ top-k 之前加

    ne = logits.shape[0]
    na = min(int(W.n_activated), ne)
    # top-k（与引擎选择排序等价：取最大的 na 个）
    order = np.argsort(-logits)
    top_ids = order[:na]
    top_vals = logits[top_ids]
    # softmax over top-k
    mx = top_vals.max()
    ex = np.exp(top_vals - mx)
    top_w = ex / ex.sum()

    return {
        'logits': logits,
        'top_ids': top_ids,
        'top_weights': top_w,
        'gate_ln': ln,
    }


def moe_route_batch(W, layer, H_in):
    """批量版本：[seq, d] → 每行一个 dict"""
    return [moe_route(W, layer, H_in[t]) for t in range(H_in.shape[0])]
