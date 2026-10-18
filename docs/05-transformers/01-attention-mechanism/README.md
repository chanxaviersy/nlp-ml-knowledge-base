# 🧩 Attention Mechanism | 注意力機制

> 摘要：注意力機制是現代 NLP 的核心突破，讓模型能動態聚焦於輸入序列的不同部分。本章詳細介紹自注意力（Self-Attention）、多頭注意力（Multi-Head Attention）的原理與代碼實現。  
> Abstract: The attention mechanism is the core breakthrough in modern NLP, enabling models to dynamically focus on different parts of the input. This chapter covers Self-Attention, Multi-Head Attention, and their implementations.

---

## 1. 注意力機制的起源 | Origin of Attention

### 1.1 Seq2Seq 的瓶頸 | Bottleneck of Seq2Seq

傳統的 Encoder-Decoder 架構中，Encoder 將整個序列壓縮為一個固定維度的向量 $\mathbf{c}$（「編碼器最後一個隱狀態」），成為瓶頸：

$$\mathbf{c} = \text{Encoder}(\mathbf{x}_1, ..., \mathbf{x}_n)$$

$$\hat{y}_t = \text{Decoder}(\mathbf{c}, \hat{y}_1, ..., \hat{y}_{t-1})$$

**問題**：無論序列多長，都只有一個上下文向量 $\mathbf{c}$，造成：
- **信息瓶頸**：長序列時前面詞的信息被遺忘
- **梯度消失**：反向傳播路徑過長
- **無差異對待**：解碼時對所有輸入位置一視同仁

### 1.2 Bahdanau Attention (2014)

Bahdanau 等人提出「加性注意力」：

$$\alpha_t = \text{softmax}(e_t), \quad e_t = v^\top \tanh(\mathbf{W}_h \mathbf{h}_t + \mathbf{W}_s \mathbf{s}_{t-1})$$

$$\mathbf{c}_t = \sum_{i=1}^{n} \alpha_{ti} \mathbf{h}_i$$

其中 $\mathbf{h}_i$ 是 Encoder 的隱狀態，$\mathbf{s}_{t-1}$ 是 Decoder 的上一個隱狀態。

**核心思想**：解碼時，每個時間步都可以「看向」所有編碼器隱狀態，並動態分配不同的權重 $\alpha_{ti}$。

---

## 2. 自注意力機制 | Self-Attention

### 2.1 Scaled Dot-Product Attention (Vaswani et al., 2017)

Transformer 的核心模塊，也是現代 LLM 的基礎：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

**三個步驟**：

1. **Query-Key 匹配**：計算 Query 和 Key 的相似度 $\mathbf{Q}\mathbf{K}^\top$
2. **縮放 (Scale)**：除以 $\sqrt{d_k}$ 防止點積過大導致 softmax 梯度消失
3. **加權求和**：用 softmax 權重對 Value 加權求和

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention
    
    參數:
        Q: (batch, seq_q, d_k)
        K: (batch, seq_k, d_k)
        V: (batch, seq_v, d_v)  通常 seq_k == seq_v
        mask: 可選的注意力遮罩
    
    返回:
        output: (batch, seq_q, d_v)
        attention_weights: (batch, seq_q, seq_k)
    """
    d_k = Q.shape[-1]
    
    # Step 1: 計算相似度分數
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Step 2: 應用遮罩（可選）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 3: Softmax 標準化
    attention_weights = F.softmax(scores, dim=-1)
    
    # Step 4: 加權求和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


class SelfAttention(nn.Module):
    """自注意力層實現"""
    
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 投影矩陣
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
    
    def split_heads(self, x, batch_size):
        """將 d_model 維度拆分為 n_heads 個頭"""
        x = x.view(batch_size, -1, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq, d_k)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # 線性投影
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)
        
        # 分頭
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # Scaled Dot-Product Attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 合併多頭：(batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # 最終線性投影
        output = self.W_O(attn_output)
        
        return output, attn_weights
```

### 2.2 為什麼要「縮放」(Scaling)？

點積的方差與 $d_k$ 成正比：
$$\text{Var}(q \cdot k) = d_k \cdot \text{Var}(q_k) \cdot \text{Var}(k_k)$$

當 $d_k$ 很大時，點積的值會很大，使 softmax 進入飽和區（梯度趨近於 0）。

除以 $\sqrt{d_k}$ 可以讓方差歸一化到 1。

```python
import matplotlib.pyplot as plt
import numpy as np

def softmax_gradient_demo():
    """展示大數值輸入時 softmax 的梯度消失問題"""
    x = np.linspace(-10, 10, 1000)
    
    # 不同溫度的 softmax
    def softmax_temp(x, T):
        e_x = np.exp(x / T - np.max(x / T))
        return e_x / e_x.sum()
    
    # 標準 softmax (T=1)
    y_std = softmax_temp(x, T=1)
    y_approx_uniform = softmax_temp(x, T=100)  # 高溫近似均勻
    
    # 梯度 = p_i * (1 - p_i)
    grad_std = y_std * (1 - y_std)
    grad_approx = y_approx_uniform * (1 - y_approx_uniform)
    
    print(f"Sum of gradients (T=1):   {grad_std.sum():.6f}")
    print(f"Sum of gradients (T=100): {grad_approx.sum():.6f}")
    print("→ 高溫時梯度更均勻，收斂更穩定")

softmax_gradient_demo()
```

---

## 3. 多頭注意力 | Multi-Head Attention

### 3.1 原理 | Principle

將 Query、Key、Value 分別投影到 $h$ 個子空間，每個頭獨立計算注意力，最後拼接：

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, ..., \text{head}_h) \mathbf{W}^O$$

$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

**每一個頭可以學習不同的「注意力模式」**：
- 頭1：專注於語法結構（主語-謂語關係）
- 頭2：專注於語義相似性（同義詞）
- 頭3：專注於位置鄰近性（局部上下文）
- 頭4：專注於長距離依賴（指代消解）

```python
class MultiHeadAttention(nn.Module):
    """完整的多頭注意力實現"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # 線性投影 + 分頭
        Q = self.split_heads(self.W_Q(Q), batch_size)  # (B, h, L, d_k)
        K = self.split_heads(self.W_K(K), batch_size)
        V = self.split_heads(self.W_V(V), batch_size)
        
        # 注意力計算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)  # (B, h, L, d_k)
        
        # 合併多頭
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        output = self.W_O(context)
        
        return output, attention_weights
    
    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
```

### 3.2 可視化：注意力權重 | Attention Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention_weights(attn_weights, tokens, title="Attention Weights"):
    """可視化注意力權重熱力圖"""
    # attn_weights: (n_heads, seq_len, seq_len) 或 (seq_len, seq_len)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(attn_weights, 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='viridis', 
                annot=False,
                fmt='.2f',
                ax=ax,
                cbar_kws={'label': 'Attention Weight'})
    
    ax.set_xlabel('Key (Attended To)', fontsize=12)
    ax.set_ylabel('Query (Attending From)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 示例
tokens = ["[CLS]", "The", "cat", "sat", "on", "the", "mat", "[SEP]"]
# 假設第3個頭的注意力：'cat' 主要看向 'sat'（主語-謂語關係）
weights_example = np.random.rand(8, 8)
weights_example = weights_example / weights_example.sum(axis=-1, keepdims=True)
visualize_attention_weights(weights_example, tokens)
```

---

## 4. 注意力機制的變體 | Attention Variants

### 4.1 種類總覽

| 類型 | 描述 | 代表模型 |
|------|------|----------|
| **Bahdanau (Additive)** | 加性注意力，拼接 Query 和 Key | Neural MT (2014) |
| **Luong (Multiplicative)** | 乘性注意力（Dot-Product） | Neural MT (2015) |
| **Scaled Dot-Product** | 縮放點積注意力 | Transformer |
| **Multi-Query Attention** | 多個 Query 共享 Key/Value | FlashAttention |
| **Grouped-Query Attention (GQA)** | Query 分組共享 KV | LLaMA 2+ |
| **Flash Attention** | IO 優化的精確注意力 | GPT-4, LLaMA |

### 4.2 Flash Attention：高效注意力

Flash Attention 通過分塊計算和重新計算，避免了 $O(N^2)$ 的顯存佔用，將複雜度降低到 $O(N)$：

```python
# 使用 HuggingFace + Flash Attention
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",  # 啟用 Flash Attention
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### 4.3 旋轉位置編碼 (RoPE) 中的注意力

RoPE 在 Query 和 Key 旋轉位置信息，而非像標準 Transformer 那樣使用加性位置編碼：

$$\mathbf{q}_m = \mathbf{R}_m \mathbf{W}_q \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_n \mathbf{W}_k \mathbf{x}_n$$

$$\text{Attention}(\mathbf{q}_m, \mathbf{k}_n, \mathbf{v}_n) = \mathbf{s}^T \mathbf{v}_n, \quad \mathbf{s} = \frac{\mathbf{q}_m \mathbf{k}_n^T}{d}$$

其中 $\mathbf{R}_m = \text{diag}(R_m^{(1)}, R_m^{(2)}, ...)$ 是旋轉矩陣。

---

## 5. 注意力機制在 NLP 中的應用 | Applications

### 5.1 Transformer Encoder（雙向注意力）

所有位置可以關注所有其他位置，適合理解任務：

$$\text{Output} = \text{MultiHeadAttention}(\mathbf{X}, \mathbf{X}, \mathbf{X})$$

### 5.2 Transformer Decoder（帶遮罩的注意力）

用於生成任務，遮罩未來位置：

```python
def create_causal_mask(seq_len):
    """創建因果遮罩，防止看到未來位置"""
    mask = torch.ones(seq_len, seq_len).triu(diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))

# 或用更高效的 2D 位置遮罩
def create_causal_mask_2d(seq_len):
    """2D 翻譯不變的因果遮罩"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask
```

### 5.3 Cross-Attention（跨注意力）

Decoder 層關注 Encoder 的輸出：

$$\text{Output} = \text{MultiHeadAttention}(\mathbf{Q}_{dec}, \mathbf{K}_{enc}, \mathbf{V}_{enc})$$

---

## 6. 總結 | Summary

```
┌─────────────────────────────────────────────────────────────┐
│  Attention 機制的三個 Q / K / V 的直覺理解                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query (我想要什麼信息？)                                    │
│    "I am looking for a cat that is sleeping on the mat"    │
│                              ↑                              │
│         Query: "sleeping" → 需要找到 sleeping 的主語        │
│                                                             │
│  Key (我有什麼信息？)                                        │
│    "cat" → Key = "cat is sleeping"                          │
│    "mat" → Key = "mat is being sat on"                     │
│                                                             │
│  Value (實際的內容)                                          │
│    當 Query 匹配 Key 時，對應的 Value 會被取出             │
│                                                             │
│  Attention Score = Q · K^T / √d  →  Softmax  →  加權 Sum   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

| 組件 | 作用 |
|------|------|
| $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ | 查詢、鍵、值的線性投影 |
| $\sqrt{d_k}$ 縮放 | 防止大維度時 softmax 梯度消失 |
| 多頭注意力 | 多子空間學習不同類型的依賴關係 |
| 遮罩 (Mask) | 實現因果性、填充忽略 |
| 殘差連接 + LayerNorm | 穩定訓練、緩解梯度問題 |

---

## 參考文獻 | References

1. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
2. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*.
3. Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS*.
4. Su, J., et al. (2022). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv*.

---

*© 2026 chanxaviersy*
