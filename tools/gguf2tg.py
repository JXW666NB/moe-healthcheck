#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GGUF → .TG 一键转换 + 验证

把 llama.cpp 生态的 GGUF 模型转成 TGAI 引擎能直接加载的 .TG，
并在转换后自动做三层验证，确保转换没有静默出错。

为什么需要验证
──────────────
转换器的失败模式往往不是「报错」，而是「静默出错」：

  · 权重名没匹配上 ⇒ 那个张量被跳过 ⇒ 推理时读到 NULL 或全零
  · tokenizer 的 merges 没对上 ⇒ 编码出的 token id 全错 ⇒ 输出乱码
  · 特殊标记 id 超出 vocab ⇒ embedding 越界
  · rope_theta / num_kv_heads 读错 ⇒ 位置编码或 GQA 布局错位

这些问题在转换器的日志里可能只表现为「Skipped: 1」。
所以转换完必须实测一遍。

三层验证
────────
  ① 加载验证    引擎能不能打开并读到正确的超参
  ② 分词往返    encode(s) → decode 必须等于 s
  ③ 前向 + 生成 logits 不是 NaN，能生成非空文本

用法
────
  # 转换 + 验证
  python tools/gguf2tg.py model.gguf -o model.TG --dtype fp16 --verify

  # 只转换
  python tools/gguf2tg.py model.gguf -o model.TG

  # 只验证已有的 .TG
  python tools/gguf2tg.py --verify-only model.TG

退出码：0=成功  1=验证失败  2=转换失败
"""

import os
import sys
import argparse
import subprocess
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


PROBE_TEXTS = [
    '你好',
    'Hello',
    '什么是线性代数？',
    'The quick brown fox jumps over the lazy dog.',
    '1 + 1 = 2',
]


def verify(tg_path, prompts=None, n_gen=24):
    """三层验证，返回 (ok, report_lines)"""
    L = []
    ok = True

    # ── ① 加载
    try:
        import ctypes
        import numpy as np
        from tgai_go import _engine
    except ImportError as e:
        return False, ['  [X] 缺少依赖: %s' % e]

    L.append('① 加载验证')
    try:
        eng = _engine.TGEngine()
        cfg = eng.load(os.path.abspath(tg_path))
    except Exception as e:
        return False, L + ['  [X] 引擎加载失败: %s' % e]

    nl = int(cfg['n_layers'])
    vocab = int(cfg['vocab_size'])
    L.append('  [OK] n_layers=%d  d_model=%d  n_heads=%d  d_ff=%d'
             % (nl, int(cfg['d_model']), int(cfg['n_heads']), int(cfg['d_ff'])))
    L.append('       vocab=%d  max_seq=%d  n_experts=%d  rope_theta=%d'
             % (vocab, int(cfg['max_seq']), int(cfg['n_experts']),
                int(cfg['rope_theta'])))

    # 超参合理性检查
    problems = []
    if nl <= 0 or nl > 200:
        problems.append('n_layers=%d 不合理' % nl)
    if vocab <= 0 or vocab > 1_000_000:
        problems.append('vocab_size=%d 不合理' % vocab)
    if int(cfg['d_model']) % int(cfg['n_heads']) != 0:
        problems.append('d_model 不能被 n_heads 整除')
    if problems:
        ok = False
        for p in problems:
            L.append('  [X] %s' % p)
    else:
        L.append('  [OK] 超参合理性检查通过')

    # ── ② 分词往返
    L.append('')
    L.append('② 分词器往返')
    bad = 0
    for s in PROBE_TEXTS:
        try:
            ids = eng.encode(s, add_bos=True)
            back_parts = []
            for i in ids:
                back_parts.append(eng.decode(i, detok=True))
            back = ''.join(back_parts)
            # 去掉 BOS 文本
            if back.startswith('<BOS>'):
                back = back[5:]
            same = (back == s)
            if not same:
                bad += 1
            L.append('  %s %-40r -> %2d ids  roundtrip=%r'
                     % ('[OK]' if same else '[!] ', s, len(ids), back[:40]))
        except Exception as e:
            bad += 1
            L.append('  [X] %r 编码异常: %s' % (s, e))
    if bad:
        ok = False
        L.append('  [X] %d / %d 个探针往返不一致' % (bad, len(PROBE_TEXTS)))
    else:
        L.append('  [OK] 全部往返一致')

    # ── ③ 前向 + 生成
    L.append('')
    L.append('③ 前向与生成')
    eng.lib.tg_dump_set_enabled(0)
    try:
        import numpy as np
        rng = np.random.RandomState(42)
        for prompt in (prompts or ['<用户>\n你好<TGAI><回答>\n'])[:1]:
            ids = eng.encode(prompt, add_bos=True)
            n0 = len(ids)
            cache = eng.lib.tg_kvcache_new(eng.model, max(256, n0 + n_gen + 16))
            if not cache:
                ok = False
                L.append('  [X] kvcache 创建失败')
                break
            try:
                arr = (ctypes.c_int32 * n0)(*ids)
                lp = eng.lib.tg_forward(eng.model, arr, n0, cache, 0)
                if not lp:
                    ok = False
                    L.append('  [X] prefill 失败')
                    break
                lg = np.ctypeslib.as_array(
                    ctypes.cast(lp, ctypes.POINTER(ctypes.c_float)),
                    shape=(vocab,)).astype(np.float64).copy()

                if not np.all(np.isfinite(lg)):
                    ok = False
                    L.append('  [X] logits 含 NaN/Inf —— 权重可能有空洞')
                    break
                L.append('  [OK] prefill 成功，top-5 token: %s'
                         % np.argsort(-lg)[:5].tolist())

                eos = int(cfg.get('eos', 3))
                out = []
                pos = n0
                for _ in range(n_gen):
                    l2 = lg / 0.7
                    idx = np.argsort(-l2)[:40]
                    c = l2[idx] - l2[idx].max()
                    p = np.exp(c); p /= p.sum()
                    nx = int(idx[rng.choice(len(idx), p=p)])
                    if nx == eos:
                        break
                    out.append(nx)
                    b = (ctypes.c_int32 * 1)(nx)
                    lp = eng.lib.tg_forward(eng.model, b, 1, cache, pos)
                    if not lp:
                        break
                    lg = np.ctypeslib.as_array(
                        ctypes.cast(lp, ctypes.POINTER(ctypes.c_float)),
                        shape=(vocab,)).astype(np.float64).copy()
                    pos += 1
            finally:
                eng.lib.tg_kvcache_free(cache)

            txt = ''.join(eng.decode(i, detok=True) for i in out)
            if len(out) == 0:
                ok = False
                L.append('  [X] 生成 0 个 token（立刻 EOS）')
            elif '<UNK>' in txt:
                ok = False
                L.append('  [!] 生成 %d token，但含 <UNK>（词表映射可能有问题）' % len(out))
                L.append('      %r' % txt[:120])
            else:
                L.append('  [OK] 生成 %d token: %r' % (len(out), txt[:120]))
    except Exception as e:
        ok = False
        L.append('  [X] 前向异常: %s' % e)

    return ok, L


def main():
    ap = argparse.ArgumentParser(
        description='GGUF → .TG 一键转换 + 三层验证',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', nargs='?', help='输入 .gguf 路径')
    ap.add_argument('-o', '--output', default=None, help='输出 .TG 路径')
    ap.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32'],
                    help='目标精度（默认 fp16）')
    ap.add_argument('--verify', action='store_true', help='转换后验证')
    ap.add_argument('--verify-only', metavar='TG', default=None,
                    help='只验证已有的 .TG 文件，不转换')
    args = ap.parse_args()

    # ── 只验证
    if args.verify_only:
        print('=' * 70)
        print('  验证 %s' % args.verify_only)
        print('=' * 70)
        ok, lines = verify(args.verify_only)
        for ln in lines:
            print(ln)
        print('=' * 70)
        print('  ★ 结果：%s' % ('PASS' if ok else 'FAIL'))
        print('=' * 70)
        return 0 if ok else 1

    if not args.input:
        ap.error('需要 input（.gguf 路径）')

    src = os.path.abspath(args.input)
    if not os.path.isfile(src):
        print('  [X] 输入不存在: %s' % src, file=sys.stderr)
        return 2
    if not src.lower().endswith('.gguf'):
        print('  [!] 输入扩展名不是 .gguf，仍尝试转换')

    dst = args.output or (os.path.splitext(src)[0] + '.TG')
    dst = os.path.abspath(dst)

    conv = os.path.join(_HERE, 'gguf_to_tg.py')
    if not os.path.isfile(conv):
        print('  [X] 找不到转换器: %s' % conv, file=sys.stderr)
        return 2

    print('=' * 70)
    print('  GGUF → .TG')
    print('=' * 70)
    print('  输入: %s  (%.1f MB)' % (src, os.path.getsize(src) / 1024 / 1024))
    print('  输出: %s' % dst)
    print('  精度: %s' % args.dtype)
    print()

    import time
    t0 = time.time()
    r = subprocess.run([sys.executable, conv, '-i', src, '-o', dst,
                        '--dtype', args.dtype],
                       capture_output=False)
    el = time.time() - t0
    if r.returncode != 0:
        print('\n  [X] 转换失败（退出码 %d）' % r.returncode, file=sys.stderr)
        return 2
    if not os.path.isfile(dst):
        print('\n  [X] 转换器报告成功但输出文件不存在', file=sys.stderr)
        return 2

    print('\n  转换完成：%.1f MB，耗时 %.1f 秒'
          % (os.path.getsize(dst) / 1024 / 1024, el))

    if not args.verify:
        print('\n  提示：加 --verify 可在转换后自动做三层验证')
        return 0

    print()
    print('=' * 70)
    print('  验证')
    print('=' * 70)
    ok, lines = verify(dst)
    for ln in lines:
        print(ln)
    print('=' * 70)
    print('  ★ 结果：%s' % ('PASS' if ok else 'FAIL'))
    print('=' * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
