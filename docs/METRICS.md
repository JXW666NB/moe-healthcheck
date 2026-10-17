# 五项指标的定义与推导

本文档给出每一项指标的**精确定义**、**为什么它有意义**、以及**在其他框架里怎么复现**。

---

## 指标 ① 残差流范数增长

### 定义

```
对一次前向，取【最后一个 token】的隐藏态 h，计算 ‖h‖₂。
得到 n_layers + 1 个值：embed 之后 + 每层之后。

增长倍率 = max(‖h‖) / ‖h_embed‖
```

### 实测参考

| 模型状态 | 增长倍率 |
|---|---|
| 健康 | **2 ~ 10×** |
| 可疑 | 10 ~ 30× |
| 报警 | 30 ~ 100× |
| 明显异常 | **> 100×** |

### 为什么有意义

健康模型的残差流应该「逐步精炼信息」——每层往残差里加一点修正量，
范数缓慢增长。如果增长 600 倍，说明：

1. 每层的修正量远大于残差本身 ⇒ 信号被反复放大
2. 最终靠 `final_norm` 一次性压回去 ⇒ 中间的隐表示处于「溢出」边缘
3. 后续层的梯度会被这个量级污染

### 其他框架怎么测

```python
# PyTorch / HF
h_norms = []
def hook(m, i, o):
    h = o[0] if isinstance(o, tuple) else o
    h_norms.append(h[:, -1, :].norm(dim=-1).item())

# embed + 每层
model.model.embed_tokens.register_forward_hook(hook)
for layer in model.model.layers:
    layer.register_forward_hook(hook)
```

**注意**：如果模型有 `attn_res_block` / `Block AttnRes` 之类的变体，
「残差流」可能不是你预期的那条，需要看实现。

---

## 指标 ② 注意力集中度

### 定义

```
对每层的每个 head：
    取最后一个 query 位置的注意力行 row[h][:]（长度 = kv_len）
    归一化：p = row / sum(row)
    归一化熵：H = -Σ p·log(p) / log(kv_len)        ∈ [0, 1]
    最大权重：w_max = max(row)

全模型统计：
    max_weight = 所有 (层, head) 里最大的 w_max
    sharp_ratio = H ≤ 0.30 的 head 占比
```

### 判定

| 指标 | 健康 | 报警 |
|---|---|---|
| `max_weight` | ≤ 0.9 | ≥ 0.995（=1.0 表示完全退化） |
| `sharp_ratio` | ≥ 8% | < 3% |

### 为什么有意义

`w_max = 1.000` 且 `H = 0.000` 意味着：**这个 head 对任何输入都输出同一个分布**
（把 100% 注意力给同一个位置）。它已经不是一个注意力机制，而是一个常数。

这浪费了 `1/n_heads` 的容量。在 16 head 的模型里就是 6.25%。

**★ 注意：`<BOS>` 上出现高权重是正常的（Attention Sink 现象），
所有 decoder-only Transformer 都有。真正异常的是「接近 1.0」。**

---

## 指标 ③ 内容 head 分布

### 定义

```
「内容 head」需同时满足：
   1. 归一化熵 H ≤ 0.50（不算太平坦）
   2. Top-1 位置对应的 token 不属于 META_TOKENS：
      <BOS> <EOS> <PAD> <UNK> <用户> <TGAI> <回答> <思考> <系统>
      \n \r\n \r 空格 \t
   3. 该 token 不是纯空白

分深度统计：
   浅层 = 前 1/4 层
   深层 = 后 1/4 层
```

### 判定

```
深层 / 浅层 < 0.2  → FAIL
深层 / 浅层 < 0.5  → WARN
浅层 = 0           → WARN（不能算百分比，本身说明模型几乎没有内容 head）
```

### 为什么有意义

如果深层的注意力全都落在 `<BOS>` 或换行符上，说明**深层没有承担语义工作**。
模型的有效深度小于名义深度。

**★ 这一项对提示词长度非常敏感**：14 个 token 里可能有 6 个是模板标记，
内容词只有 8 个，内容 head 天然稀少。
**建议用 100+ token 的真实文本复测。**

---

## 指标 ④ 专家权重分布

### 定义

```
对每个专家 e，取所有层所有矩阵（w1/w2/w3）的 Frobenius 范数之和，
再对层取平均：

    S[e] = mean_L ( ‖w1_L,e‖_F + ‖w2_L,e‖_F + ‖w3_L,e‖_F )

统计：
    CV = std(S) / mean(S)
    两簇间隔 = 1 - (小簇均值 / 大簇均值)   （对排序后的 S 找最大间隔）
```

### 判定

| 指标 | 健康 | 报警 |
|---|---|---|
| `CV` | ≤ 0.15 | ≥ 0.30 |

### 为什么有意义

如果一半专家的权重范数只有另一半的 55%，说明**容量分配不均**：
小权重专家只能做增量修正，不能独立承担任务。

**★ 实测反例：范数最小的专家不一定最不重要。**
在 TGAI V2.7 里，E6 的范数最小（175.5）但它是中文主力（7/28 层 top-1）。
**⇒ 不能只用范数判断重要性，必须配合指标 ⑤ 或消融实验。**

---

## 指标 ⑤ 路由卡死层数 ★

### 定义

```
生成 N 个 token（默认 8，采样 t=0.7 / top_k=40 / 固定 seed）

对每一层 L，记录每步的 top-1 专家，得到长度 N 的序列 seq_L

卡死层数 = |{ L : seq_L 里只有 1 个不同的值 }|

辅助指标：
    层熵 = mean_L ( seq_L 的归一化专家分布熵 )
    确信度 = mean( softmax(top-2 logits)[0] )   （top-1 的权重）
```

### 判定

```
卡死层数 ≤ 3   → PASS
卡死层数 5..9  → WARN
卡死层数 ≥ 10  → FAIL
```

### 为什么有意义

router 的输入是 `ln2(h)`，即**当前 token 在这一层的隐藏态**：

```
r = ln2(h) · W_router^T + qb_bias
top2 = argmax_2(r)
```

所以：

```
隐藏态健康  → 不同 token 的 ln2(h) 差异明显 → 路由分数差异大 → 每步换专家
隐藏态退化  → ln2(h) 近似常量            → 路由分数几乎不变 → 固定选同一个
```

**★★ 「卡死层数」因此是隐藏态健康度的直接代理指标。
它不需要人工判读输出质量，纯数值、可自动化、可直接接进训练监控。**

### 两种失效模式（这是本工具的核心观察）

**实测数据（TGAI NB V2.7，28 层，8 专家，prompt="你好"）：**

| 配置 | 输出 | 卡死层数 | 层熵 | 确信度 |
|---|---|---|---|---|
| 基线 | `你好！很高兴认识你！😊` | **0 / 28** | 1.099 | 0.756 |
| 禁 E6 | `您好！我是一个人工智能模型` | 1 / 28 | 1.002 | 0.768 |
| 禁 E7 | `你好！<UNK>...` | **2 / 28** | **0.832** | 0.760 |
| 禁 E0 | `我红我！精` | **13 / 28** | **0.414** | 0.692 |

```
模式 A：路由卡死（禁 E0）
   卡死 13/28，层熵 0.414（腰斩），确信度 0.692（略降）
   机制：隐藏态退化 → router 输入近似常量 → 固定选同一专家
   特征：输出极短、无意义、立刻 EOS

模式 B：路由正常但专家错配（禁 E7）
   卡死 2/28，层熵 0.832（接近正常），确信度 0.760（和基线一样）
   机制：路由照常工作，但选中的专家不会干这活
   特征：前半句正常、后半句崩、出现 <UNK>
```

**★★★ 判别价值：两种模式的输出都是「坏」，但原因完全不同。
只看输出无法区分，看卡死层数就能立刻判断是「隐藏态问题」还是「专家能力问题」。**

### 其他框架怎么测

```python
# HF MoE 模型：hook router，记录 top-k
picks = []   # picks[layer][step] = list of expert ids
def make_hook(L):
    def hook(m, i, o):
        # o 通常是 router_logits: [batch, seq, n_experts]
        logits = o[0] if isinstance(o, tuple) else o
        top = logits[:, -1, :].topk(2, dim=-1).indices[0].tolist()
        picks[L].append(top)
    return hook

for L, layer in enumerate(model.model.layers):
    layer.mlp.gate.register_forward_hook(make_hook(L))
```

然后对每层的 `picks[L]` 序列算「不同值的个数」。

---

## 阈值速查表

```python
THRESHOLDS = {
    'norm_growth_warn': 30.0,
    'norm_growth_fail': 100.0,
    'max_weight_warn': 0.95,
    'max_weight_fail': 0.995,
    'deep_content_ratio_warn': 0.5,
    'deep_content_ratio_fail': 0.2,
    'expert_cv_warn': 0.15,
    'expert_cv_fail': 0.30,
    'stuck_layers_warn': 5,
    'stuck_layers_fail': 10,
    'sharp_entropy': 0.30,
    'content_entropy': 0.50,
}
```

**★ 这套阈值是在 3B / 28 层 / 16 head / 8 专家上校准的。
换规模时，`norm_growth` 和 `expert_cv` 需要重新标定；
`max_weight` 和 `stuck_layers` 相对稳定（它们衡量的是「退化」而非「规模」）。**
