# 📝 文本摘要 | Text Summarization

> 摘要：自動摘要將長文本壓縮為保留核心信息的短文本。本章介紹抽取式（Extractive）和生成式（Abstractive）兩大範式，以及 Pre-training + Fine-tuning 範式（BART、T5、Pegasus）。  
> Abstract: Automatic summarization compresses long documents into short texts preserving key information. This chapter covers extractive and abstractive approaches, and pre-training + fine-tuning paradigms.

---

## 1. 任務定義 | Task Definition

**輸入**：源文檔 $D = (w_1, w_2, ..., w_n)$，$n$ 較大  
**輸出**：摘要 $S = (w_1, w_2, ..., w_m)$，$m \ll n$

**兩大範式**：
- **抽取式 (Extractive)**：從原文中選擇重要句子
- **生成式 (Abstractive)**：生成新詞，可能包含原文中沒有的詞

---

## 2. 抽取式摘要 | Extractive Summarization

### 2.1 TextRank

基於 PageRank 的圖算法：

1. 將每個句子作為圖的一個節點
2. 句子之間的相似度作為邊權重
3. 迭代計算 PageRank 分數
4. 選擇 top-k 個句子

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def textrank(sentences, top_k=3, damping=0.85, max_iter=100):
    """TextRank 抽取式摘要"""
    # 1. 計算句子相似度矩陣
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(sentence_vectors)
    
    # 2. 構建圖
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph, alpha=damping, max_iter=max_iter)
    
    # 3. 排序選 top-k
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), 
                    reverse=True)
    summary = [s for _, s in ranked[:top_k]]
    return summary
```

### 2.2 傳統特徵方法

| 特徵 | 描述 |
|------|------|
| 詞頻 | TF 或 TF-IDF |
| 句子位置 | 開頭/結尾的句子更重要 |
| 標題相似度 | 與標題相似的句子重要 |
| 關鍵詞 | 包含文檔關鍵詞的句子 |
| 句子長度 | 太短/太長的句子不選 |

### 2.3 神經抽取式摘要

```python
import torch
import torch.nn as nn

class ExtractiveSummarizer(nn.Module):
    """句子級抽取式摘要（BERTSum 風格）"""
    
    def __init__(self, model_name="bert-base-chinese", num_sentences=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        # input_ids: (batch, num_sentences, max_seq_len)
        batch_size, num_sentences, seq_len = input_ids.shape
        
        # 重塑為 (batch * num_sentences, seq_len)
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        
        # BERT 編碼
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] 向量
        cls_hidden = outputs.last_hidden_state[:, 0]  # (batch * num_sentences, hidden)
        cls_hidden = cls_hidden.view(batch_size, num_sentences, -1)
        
        # 句子級分類
        logits = self.classifier(cls_hidden).squeeze(-1)  # (batch, num_sentences)
        return logits  # 每個句子的重要性分數
```

---

## 3. 生成式摘要 | Abstractive Summarization

### 3.1 Seq2Seq 基線

```
[編碼器]  →  [上下文向量]  →  [解碼器]
   源文檔                          生成摘要
```

### 3.2 BART

BART 是 denoising autoencoder，預訓練 + 微調：

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

model_name = "fnlp/bart-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 數據準備
def preprocess(examples):
    inputs = tokenizer(examples["document"], max_length=1024, 
                       truncation=True, padding="max_length")
    labels = tokenizer(examples["summary"], max_length=128, 
                       truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

# 訓練
training_args = TrainingArguments(
    output_dir="./bart_summarization",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=3e-5,
    warmup_steps=500,
    weight_decay=0.01,
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

# 推理
def summarize(text, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summarize(long_article))
```

### 3.3 T5 / mT5

T5 是 Text-to-Text Transfer Transformer：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# T5 用前綴指定任務
input_text = "summarize: " + article
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```

### 3.4 PEGASUS

PEGASUS 的預訓練任務專門針對摘要（Gap Sentence Generation）：

```python
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=8)
```

---

## 4. 長文檔摘要 | Long Document Summarization

### 4.1 挑戰

- LLM 上下文窗口有限（4K-32K tokens）
- 長文檔中關鍵信息分散
- 計算開銷大

### 4.2 解決方案

| 方法 | 描述 |
|------|------|
| **Hierarchical Attention** | 先編碼段落，再編碼文檔 |
| **Sparse Attention** | Longformer、BigBird |
| **Sliding Window** | 滑動窗口分段摘要 |
| **Retrieval-based** | 先檢索相關段落，再摘要 |
| **Chunk + Aggregate** | 分塊摘要後合併 |

```python
# 使用 Longformer
from transformers import LongformerTokenizer, LongformerForSeq2SeqLM

tokenizer = LongformerTokenizer.from_pretrained("allenai/led-base-16384")
model = LongformerForSeq2SeqLM.from_pretrained("allenai/led-base-16384")

# 處理 16K 上下文
inputs = tokenizer(article, return_tensors="pt", max_length=16384, truncation=True)

# 設置全局 attention（聚焦於 [CLS]）
global_attention_mask = torch.zeros_like(inputs["input_ids"])
global_attention_mask[:, 0] = 1  # [CLS] 全局關注

summary_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    global_attention_mask=global_attention_mask,
    max_length=512,
    num_beams=4,
)
```

---

## 5. 評估指標 | Evaluation Metrics

### 5.1 ROUGE

**ROUGE-N**：n-gram 重疊

$$\text{ROUGE-N} = \frac{\sum_{S \in Ref} \sum_{gram_n \in S} Count_{match}(gram_n)}{\sum_{S \in Ref} \sum_{gram_n \in S} Count(gram_n)}$$

**ROUGE-L**：最長公共子序列

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print(scores)
# {'rouge1': Score(precision=0.5, recall=0.6, fmeasure=0.545), ...}
```

### 5.2 其他指標

| 指標 | 描述 |
|------|------|
| **BERTScore** | 基於 BERT 嵌入的相似度 |
| **METEOR** | 考慮同義詞和詞幹 |
| **BLEU** | 機器翻譯經典指標 |
| **FactCC / QAGS** | 事實一致性 |
| **BARTScore** | 基於 BART 的生成評分 |

```python
# BERTScore
from bert_score import score
P, R, F1 = score([generated_summary], [reference_summary], lang="zh")
print(f"BERTScore F1: {F1.item():.4f}")
```

---

## 6. 提示工程摘要 | Prompting for Summarization

```python
def prompt_summarize(text, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "system",
            "content": "你是一個專業的文本摘要助手。請生成簡潔、準確的中文摘要，保留原文關鍵信息。"
        }, {
            "role": "user",
            "content": f"請為以下文章生成 100 字以內的摘要：\n\n{text}\n\n摘要："
        }],
        temperature=0.3,
    )
    return response.choices[0].message.content
```

**Few-shot 範例**：
```python
FEW_SHOT_PROMPT = """請為以下文章生成 100 字以內的摘要。

範例 1：
文章：[關於氣候變遷的長文]
摘要：氣候變遷加劇，極端天氣頻發，全球需加強減排合作。

範例 2：
文章：[關於新科技的報導]
摘要：新型量子計算機突破，運算速度提升 1000 倍。

現在請摘要：
文章：{text}
摘要："""
```

---

## 7. 應用場景 | Applications

| 領域 | 應用 |
|------|------|
| **新聞** | 新聞摘要、每日頭條 |
| **會議** | 自動生成會議紀要 |
| **法律** | 合同摘要、判決書摘要 |
| **醫療** | 病歷摘要、文獻綜述 |
| **學術** | 論文摘要、文獻綜述 |
| **客服** | 對話摘要 |
| **電商** | 商品描述、用戶評論摘要 |

---

## 總結 | Summary

| 方法 | 優點 | 缺點 | 代表模型 |
|------|------|------|----------|
| **TextRank** | 無監督、快速 | 句子級、不流暢 | LexRank |
| **BERTSum** | 抽取效果好 | 仍受原文限制 | BERTSum |
| **BART** | 生成流暢、可控 | 上下文有限 | BART, mBART |
| **T5** | 統一框架、多任務 | 計算成本高 | T5, mT5, FLAN-T5 |
| **PEGASUS** | 摘要專用預訓練 | 訓練成本高 | PEGASUS |
| **Longformer** | 長文檔 | 模型較大 | LED, LongT5 |
| **LLM Prompting** | 零樣本能力強 | 成本高、不穩定 | GPT-4, Claude |

---

## 參考文獻 | References

1. Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing Order into Texts. *EMNLP*.
2. Liu, Y., et al. (2018). Generating Wikipedia by Summarizing Long Sequences. *ICLR*.
3. Lewis, M., et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training. *ACL*.
4. Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5). *JMLR*.
5. Zhang, J., et al. (2020). PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization. *ICML*.

---

*© 2026 chanxaviersy*
