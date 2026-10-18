# 🌍 Transformer 架構 | Transformer Architecture

> 摘要：本章介紹 Transformer 的完整架構，從 Encoder-Decoder 框架到 BERT、GPT 系列模型的原理與差異。涵蓋位置編碼、預訓練、微調策略等核心概念。  
> Abstract: This chapter covers the complete Transformer architecture, from the Encoder-Decoder framework to BERT, GPT, and beyond. Key concepts include positional encoding, pretraining, and fine-tuning strategies.

---

## 1. Transformer 總覽 | Overview

Transformer (Vaswani et al., 2017) 完全基於注意力機制，摒棄了 RNN 的順序計算，支持並行訓練。

```
┌──────────────────────────────────────────────────────────────────┐
│                        Transformer                                │
├────────────────────────────┬─────────────────────────────────────┤
│         Encoder             │              Decoder                 │
│  ┌──────────────────┐      │  ┌─────────────────────────────┐    │
│  │  Input Embedding │      │  │  Output Embedding (shifted) │    │
│  │  + Positional    │      │  │  + Positional Enc.         │    │
│  │  Encoding        │      │  │                             │    │
│  └────────┬─────────┘      │  └──────────┬──────────────────┘    │
│           ↓                │             ↓                        │
│  ┌─────────────────────┐  │  ┌─────────────────────────────┐    │
│  │  Encoder Layer × N   │──┼─→│  Masked Multi-Head Attn.   │    │
│  │  ┌─────────────────┐ │  │  │  (causal, cannot see future)│    │
│  │  │ Multi-Head     │ │  │  └──────────┬──────────────────┘    │
│  │  │ Self-Attention │ │  │             ↓                        │
│  │  └────────┬────────┘ │  │  ┌─────────────────────────────┐    │
│  │           ↓          │  │  │  Cross-Attention            │    │
│  │  ┌─────────────────┐ │  │  │  (attend to Encoder output) │    │
│  │  │ Feed-Forward    │ │  │  └──────────┬──────────────────┘    │
│  │  │ Network (FFN)   │ │  │             ↓                        │
│  │  └────────┬────────┘ │  │  ┌─────────────────────────────┐    │
│  └───────────┼──────────┘  │  │  Feed-Forward Network      │    │
│              ↓              │  └──────────┬──────────────────┘    │
│  ┌─────────────────────┐  │             ↓                        │
│  │  Encoder Layer × N  │  │  ┌─────────────────────────────┐    │
│  └───────────┬─────────┘  │  │  Decoder Layer × N (×N)    │    │
│              ↓             │  └──────────┬──────────────────┘    │
│  ┌─────────────────────┐  │             ↓                        │
│  │  Encoder Output     │─────────────→│  Linear + Softmax      │    │
│  └─────────────────────┘              │  → Output Probabilities │    │
│                                       └─────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. 位置編碼 | Positional Encoding

注意力機制本身是「置換等變」的（Permutation Equivariant）：改變輸入順序不會改變輸出順序。**位置編碼**為序列注入位置信息。

### 2.1 正弦/餘弦位置編碼 (Sinusoidal PE)

$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    """Transformer 原創的正弦/餘弦位置編碼"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 創建位置編碼矩陣 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def visualize_positional_encoding(d_model=64, max_len=100):
    """可視化位置編碼"""
    pe = np.zeros((max_len, d_model))
    position = np.arange(0, max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    plt.figure(figsize=(14, 5))
    plt.imshow(pe.T, aspect='auto', cmap='RdBu')
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Dimension', fontsize=12)
    plt.title('Positional Encoding Heatmap', fontsize=14)
    plt.colorbar(label='Encoding Value')
    plt.tight_layout()
    plt.show()

visualize_positional_encoding()
```

### 2.2 相對位置編碼 | Relative Positional Encoding

Transformer-XL 引入相對位置編碼，直接建模 Token 之間的相對距離：

$$\text{Attention}_{\text{rel}} = \frac{(\mathbf{Q} + \mathbf{u})(\mathbf{K} + \mathbf{v})^\top}{\sqrt{d}}$$

新增的可學習參數 $\mathbf{u}, \mathbf{v}$ 分別代表 Query 和 Key 的偏置。

### 2.3 RoPE（旋轉位置編碼）| Rotary Position Embedding

RoPE 不使用加性位置編碼，而是在 Query 和 Key 上**旋轉**：

$$\mathbf{q}_m^T \mathbf{k}_n = (\mathbf{R}_m \mathbf{q}_m)^T (\mathbf{R}_n \mathbf{k}_n) = \mathbf{q}_m^T \mathbf{R}_{m-n} \mathbf{k}_n$$

這種設計使注意力分數只依賴於**相對位置** $m-n$，且具有長上下文外推能力。

---

## 3. BERT：雙向 Encoder | Bidirectional Encoder

### 3.1 架構特點

- **僅 Encoder**：雙向建模，無因果遮罩
- **雙向語言模型目標 (MLM)**：遮蓋 15% 的 Token，預測被遮蓋的詞
- **下一句預測 (NSP)**：預測句對 B 是否是句對 A 的下一句

### 3.2 代碼實現

```python
from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')

# MLM 演示
text = "今天的[MASK]很好，我要去公園散步。"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
top_tokens = torch.topk(predictions[0, mask_token_index], 5)

for i, (token_id, score) in enumerate(zip(top_tokens.indices, top_logits.scores)):
    word = tokenizer.decode([token_id])
    print(f"  Top {i+1}: '{word}' (score: {score.item():.4f})")
```

---

## 4. GPT：單向 Decoder | Unidirectional Decoder

### 4.1 架構特點

- **僅 Decoder**：因果遮罩，只能看到前面的 Token
- **下一個 Token 預測 (Next Token Prediction)**：標準語言模型目標
- **生成能力強**：適合文本生成任務

### 4.2 GPT 系列演進

| 版本 | 參數量 | 關鍵創新 |
|------|--------|----------|
| GPT-1 | 117M | 首次提出預訓練+微調範式 |
| GPT-2 | 1.5B | Zero-shot Task Transfer |
| GPT-3 | 175B | In-context Learning (Few-shot) |
| GPT-4 | ~1.8T (估) | 多模態、RLHF、 長上下文 |
| GPT-4o | - | 原生多模態、實時推理 |
| o1 / o3 | - | 推理時擴展 (Test-time Compute) |

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 文本生成演示
input_text = "Artificial intelligence will"
inputs = tokenizer(input_text, return_tensors="pt")

output = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id,
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

---

## 5. Encoder vs Decoder vs Encoder-Decoder

| 特性 | Encoder-Only | Decoder-Only | Encoder-Decoder |
|------|-------------|-------------|----------------|
| **代表模型** | BERT, RoBERTa | GPT, LLaMA, PaLM | T5, BART, ChatGLM |
| **注意力** | 雙向 (Bidirectional) | 單向因果 (Causal) | Encoder 雙向 + Decoder 交叉 |
| **訓練目標** | MLM + NSP | Next Token Prediction | Seq2seq (encoder: MLM, decoder: LM) |
| **擅長任務** | 理解：分類、NER、QA | 生成：寫作、代碼、對話 | 轉換：翻譯、摘要、問答 |
| **推理效率** | 高 | 高 | 中等 |
| **上下文** | 雙向，適合閱讀理解 | 單向，適合自回歸生成 | 混合，encoder 看全文，decoder 生成 |
| **典型應用** | NLP 分類任務 | ChatGPT, Copilot | 機器翻譯、Text-to-SQL |

---

## 6. 預訓練與微調 | Pretraining & Fine-tuning

### 6.1 預訓練策略

| 策略 | 描述 | 代表方法 |
|------|------|----------|
| **MLM** | 遮蓋語言模型，預測 [MASK] Token | BERT |
| **CLM** | 因果語言模型，下一個 Token 預測 | GPT |
| **PLM** | 置換語言模型，任意順序預測 | XLNet |
| **Span Corruption** | 遮蓋連續片段 [X] | T5, BART |
| **Denoising** | 添加噪聲，學習重建 | BART |

### 6.2 微調策略

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 準備數據
texts = ["這部電影很好看！", "一般般，沒什麼亮點", "浪費時間，很失望"]
labels = [2, 1, 0]  # positive, neutral, negative

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
inputs["labels"] = torch.tensor(labels)

# 微調訓練
outputs = model(**inputs)
loss = outputs.loss
loss.backward()  # 標準 PyTorch 訓練循環
```

### 6.3 高效微調方法

| 方法 | 全稱 | 原理 | 適用場景 |
|------|------|------|----------|
| **LoRA** | Low-Rank Adaptation | 對 W = W₀ + BA (低秩分解) | 資源有限、多任務 |
| **QLoRA** | Quantized LoRA | 4-bit 量化 + LoRA | 超大模型微調 |
| **Adapter** | - | 插入小型適配層 | 快速任務切換 |
| **Prefix Tuning** | - | 在輸入前加可學習前綴 | 生成任務 |
| **Prompt Tuning** | - | 只微調虛擬 Token embeddings | 極低成本遷移 |

```python
# LoRA 示例
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # 通常只對注意力投影層應用
)

# 將 LoRA 應用到模型
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Trainable: 0.19%, 1.8M params (假設 base model 有 1B 參數)
```

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│                   Transformer 生態                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   預訓練基礎模型 (Pretrained Foundation Models)              │
│   ├── Encoder Only (雙向理解)                                │
│   │   ├── BERT, RoBERTa, DeBERTa                            │
│   │   └── 適合：分類、NER、情感分析、QA                       │
│   │                                                            │
│   ├── Decoder Only (自回歸生成)                              │
│   │   ├── GPT 系列, LLaMA, PaLM, ChatGLM                    │
│   │   └── 適合：對話、寫作、代碼生成、推理                    │
│   │                                                            │
│   └── Encoder-Decoder (序列到序列)                          │
│       ├── T5, BART, FLAN-T5, NLLB                           │
│       └── 適合：翻譯、摘要、問答、Text-to-Text               │
│                                                              │
│   高效微調：LoRA, QLoRA, Adapter, Prefix Tuning              │
│   部署優化：量化 (INT4/INT8), 蒸餾, 剪枝, KV Cache          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 參考文獻 | References

1. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
2. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL*.
3. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Technical Report*.
4. Brown, T., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.
5. Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.

---

*© 2026 chanxaviersy*
