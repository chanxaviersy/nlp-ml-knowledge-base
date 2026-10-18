# 📖 語言模型 | Language Models

> 摘要：語言模型是 NLP 的核心，學習詞序列的機率分佈 $P(w_1, w_2, ..., w_n)$。本章從 N-gram 統計語言模型開始，介紹 RNN/LSTM/GRU 神經語言模型，再到現代 Transformer 語言模型。  
> Abstract: Language models learn the probability distribution over word sequences. This chapter progresses from N-gram statistical LMs to RNN/LSTM/GRU neural LMs, and finally to modern Transformer LMs.

---

## 1. 語言模型定義 | Definition

**目標**：學習一個詞序列的機率分佈：

$$P(w_1, w_2, ..., w_n) = P(w_1) \cdot P(w_2 | w_1) \cdot P(w_3 | w_1, w_2) \cdots P(w_n | w_1, ..., w_{n-1})$$

**應用**：
- 文本生成（聊天、寫作、補全）
- 語音識別（語音 → 多個文本候選，選最可能的）
- 機器翻譯（候選翻譯評分）
- 拼寫檢查
- 困惑度 (Perplexity) 衡量模型質量

---

## 2. N-Gram 語言模型 | N-Gram LM

### 2.1 馬可夫假設 | Markov Assumption

假設第 $n$ 個詞只依賴前 $n-1$ 個詞（n-gram）：

$$P(w_i | w_1, ..., w_{i-1}) \approx P(w_i | w_{i-n+1}, ..., w_{i-1})$$

**常見 n-gram**：
- **Unigram (n=1)**：$P(w_i)$
- **Bigram (n=2)**：$P(w_i | w_{i-1})$
- **Trigram (n=3)**：$P(w_i | w_{i-1}, w_{i-2})$

### 2.2 最大似然估計

$$P(w_i | w_{i-n+1}^{i-1}) = \frac{C(w_{i-n+1}^{i})}{C(w_{i-n+1}^{i-1})}$$

```python
from collections import defaultdict, Counter
import nltk
from nltk.util import ngrams

nltk.download('punkt_tab')

def train_ngram_model(corpus, n=2):
    """訓練 n-gram 語言模型"""
    tokens = nltk.word_tokenize(corpus.lower())
    ngram_counts = Counter(ngrams(tokens, n))
    context_counts = Counter(ngrams(tokens, n-1))
    
    model = defaultdict(lambda: defaultdict(float))
    
    for ngram, count in ngram_counts.items():
        context = ngram[:-1]
        word = ngram[-1]
        model[context][word] = count / context_counts[context]
    
    return model

# 訓練 bigram
corpus = "the cat sat on the mat the cat ate the mouse"
bigram_model = train_ngram_model(corpus, n=2)

# 查詢機率
print("P(cat | the):", bigram_model[('the',)]['cat'])
print("P(sat | cat):", bigram_model[('cat',)]['sat'])
```

### 2.3 數據稀疏問題與平滑 | Sparsity & Smoothing

| 問題 | 描述 |
|------|------|
| **未見 n-gram** | $C(w_{i-n+1}^{i}) = 0$ → 機率為 0 |
| **低頻 n-gram** | 統計不穩定 |

**平滑方法**：

| 方法 | 公式 | 思想 |
|------|------|------|
| **Laplace (Add-1)** | $P(w_i \| w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + V}$ | 加 1 平滑 |
| **Add-k** | 加 $k$（$0 < k < 1$） | 較溫和 |
| **Kneser-Ney** | 基於延續機率 | 當前 SOTA 平滑 |
| **Interpolation** | 線性混合 unigram + bigram + trigram | 多階數據共享 |

```python
class NGramLM:
    """N-gram 語言模型含 Kneser-Ney 平滑"""
    
    def __init__(self, n=3):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()
    
    def train(self, corpus):
        tokens = corpus.split()
        self.vocab.update(tokens)
        
        # 計算 n-gram 頻率
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            context = ngram[:-1]
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1
    
    def probability(self, ngram, discount=0.75):
        """Kneser-Ney 簡化版"""
        context = ngram[:-1]
        word = ngram[-1]
        
        if self.context_counts[context] == 0:
            return 1 / len(self.vocab)
        
        # 折扣後的機率
        prob = max(self.ngram_counts[ngram] - discount, 0) / self.context_counts[context]
        
        return prob
```

### 2.4 評估指標：困惑度 | Perplexity (PPL)

$$PPL(W) = 2^{H(W)} = P(w_1 w_2 ... w_N)^{-1/N}$$

困惑度越低，模型越好。完美模型的 PPL = 1。

```python
import math

def perplexity(model, test_corpus):
    """計算困惑度"""
    tokens = test_corpus.split()
    n = model.n
    log_prob = 0
    N = 0
    
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        p = model.probability(ngram)
        log_prob += math.log2(p) if p > 0 else -100
        N += 1
    
    return 2 ** (-log_prob / N)
```

---

## 3. 神經語言模型 | Neural Language Models

### 3.1 從 N-gram 到神經網路

**問題**：N-gram 無法泛化到語義相似的詞（"cat" 和 "dog" 共享特徵）。

**突破**：用神經網路的**分布式表示**代替離散詞索引。

### 3.2 前饋神經語言模型 (Bengio et al., 2003)

```
輸入: w_{t-1}, w_{t-2}, ..., w_{t-n+1}
   ↓
查表 (Embedding)
   ↓
拼接成 [v(w_{t-1}); v(w_{t-2}); ...; v(w_{t-n+1})]
   ↓
隱藏層: tanh(W · [v1, v2, ...] + b)
   ↓
輸出層: Softmax(V · h + c)
   ↓
P(w_t | w_{t-1}, ..., w_{t-n+1})
```

### 3.3 RNN 語言模型

RNN 通過循環連接維持歷史信息：

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t)$$

$$P(w_t | w_{<t}) = \text{softmax}(\mathbf{W}_{hy} \mathbf{h}_t)$$

```python
import torch
import torch.nn as nn

class RNNLanguageModel(nn.Module):
    """RNN 語言模型"""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers,
            dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len) - token indices
        """
        emb = self.dropout(self.embed(x))           # (batch, seq, embed)
        output, hidden = self.rnn(emb, hidden)      # (batch, seq, hidden)
        output = self.dropout(output)
        logits = self.fc(output)                    # (batch, seq, vocab)
        return logits, hidden
    
    def generate(self, start_token, max_len=50, temperature=1.0):
        """自回歸生成"""
        self.eval()
        generated = [start_token]
        hidden = None
        x = torch.tensor([[start_token]])
        
        with torch.no_grad():
            for _ in range(max_len):
                logits, hidden = self.forward(x, hidden)
                logits = logits[0, -1] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                x = torch.tensor([[next_token]])
        
        return generated
```

### 3.4 LSTM：解決長期依賴

LSTM 通過**門控機制**控制信息流：

$$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(Forget gate)}$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(Input gate)}$$
$$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C) \quad \text{(Candidate)}$$
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t \quad \text{(Cell state)}$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(Output gate)}$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$

```python
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, 
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        emb = self.dropout(self.embed(x))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout(output)
        return self.fc(output), hidden
```

### 3.5 GRU：簡化版 LSTM

GRU 只有兩個門：

$$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(Update gate)}$$
$$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(Reset gate)}$$
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

**LSTM vs GRU**：
- LSTM：3 個門（input, forget, output）+ Cell State
- GRU：2 個門（update, reset），無單獨 Cell State
- GRU 參數更少，訓練更快；LSTM 表達能力更強

---

## 4. RNN 訓練技巧 | Training Tips

### 4.1 BPTT (Backpropagation Through Time)

RNN 的反向傳播沿時間展開，計算量大且容易梯度消失/爆炸。

**Teacher Forcing**：訓練時用真實序列作為輸入而非模型預測：

```python
def train_step(model, optimizer, criterion, x, y, hidden=None):
    """
    x: (batch, seq_len) - 輸入序列
    y: (batch, seq_len) - 目標序列（x 向左偏移 1）
    """
    optimizer.zero_grad()
    logits, hidden = model(x, hidden)
    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    optimizer.step()
    return loss.item(), hidden
```

### 4.2 梯度裁剪

防止梯度爆炸：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

---

## 5. 現代語言模型 | Modern Language Models

### 5.1 Transformer 語言模型

現代 LM 基於 Transformer 架構：

| 模型 | 架構 | 規模 | 發表 |
|------|------|------|------|
| GPT-2 | Decoder | 1.5B | 2019 |
| GPT-3 | Decoder | 175B | 2020 |
| LLaMA | Decoder | 7B-65B | 2023 |
| LLaMA 2/3 | Decoder | 7B-405B | 2023-2024 |
| Qwen | Decoder | 0.5B-72B | 2023-2024 |
| Mistral | Decoder | 7B-22B | 2023-2024 |
| ChatGLM | Encoder-Decoder | 6B-130B | 2023-2024 |

### 5.2 使用 HuggingFace 訓練小 LM

```python
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from datasets import load_dataset

# 1. 載入資料
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

# 2. 載入 tokenizer 和模型
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. 預處理
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 4. Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # 因果語言模型
)

# 5. 訓練配置
training_args = TrainingArguments(
    output_dir="./lm_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    warmup_steps=500,
    logging_steps=100,
)

# 6. 訓練
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

trainer.train()

# 7. 文本生成
from transformers import pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
text = generator("Once upon a time", max_length=50, num_return_sequences=1)
print(text[0]["generated_text"])
```

---

## 6. 評估語言模型 | Evaluation

### 6.1 困惑度 (Perplexity)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    
# Loss 就是交叉熵
loss = outputs.loss
perplexity = torch.exp(loss)
print(f"Perplexity: {perplexity.item():.2f}")
```

### 6.2 下游任務評估 (GLUE, SuperGLUE)

| 任務 | 描述 |
|------|------|
| **GLUE** | 9 個 NLU 任務的基準（SST-2, MNLI, QQP, ...） |
| **SuperGLUE** | 更難的 NLU 任務 |
| **MMLU** | 57 個學科的多選題 |
| **HumanEval** | 代碼生成 |
| **GSM8K** | 數學推理 |
| **HellaSwag** | 常識推理 |

---

## 7. 條件語言模型 | Conditional Language Models

語言模型不僅能生成文本，還能基於條件生成：

$$P(\mathbf{y} | \mathbf{x}) = \prod_{t=1}^{T} P(y_t | y_{<t}, \mathbf{x})$$

| 任務 | 條件 | 生成 |
|------|------|------|
| **機器翻譯** | 源語言句子 | 目標語言句子 |
| **摘要** | 長文檔 | 短摘要 |
| **QA** | 問題 + 上下文 | 答案 |
| **對話** | 對話歷史 | 回覆 |

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│  語言模型發展路線圖                                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  統計方法                                                    │
│  ├── N-gram (1990s)                                         │
│  ├── 平滑：Kneser-Ney                                       │
│  └── 局限：稀疏、無法泛化                                   │
│                                                              │
│  神經方法                                                    │
│  ├── 前饋 NN LM (Bengio 2003)                               │
│  ├── RNN LM (Mikolov 2010)                                  │
│  ├── LSTM/GRU (2014-2017)                                   │
│  └── 局限：順序訓練、無法並行、長距離依賴                   │
│                                                              │
│  Transformer 時代                                            │
│  ├── GPT-2/3 (2019-2020) — Decoder                          │
│  ├── BERT (2018) — Encoder (MLM)                            │
│  ├── T5/BART (2019-2020) — Encoder-Decoder                  │
│  ├── LLaMA/Qwen/Mistral (2023+) — 開源大模型                 │
│  └── GPT-4/Claude/Gemini (2023+) — 閉源頂級                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

| 評估 | 內容 |
|------|------|
| 困惑度 PPL | 衡量預測能力，越低越好 |
| GLUE/SuperGLUE | NLU 綜合基準 |
| MMLU | 多任務語言理解 |
| HumanEval | 代碼能力 |
| 人工評估 | 開放式生成質量 |

---

## 參考文獻 | References

1. Bengio, Y., et al. (2003). A Neural Probabilistic Language Model. *JMLR*.
2. Mikolov, T., et al. (2010). Recurrent neural network based language model. *INTERSPEECH*.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
4. Chung, J., et al. (2014). Empirical Evaluation of Gated Recurrent Neural Networks. *arXiv*.
5. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
6. Brown, T., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.

---

*© 2026 chanxaviersy*
