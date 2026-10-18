# 🏷️ 命名實體識別 | Named Entity Recognition (NER)

> 摘要：NER 從文本中識別並分類命名實體（人名、地名、機構名等）。本章介紹 BIO 標註、CRF、BiLSTM-CRF、BERT-NER 等方法。  
> Abstract: NER identifies and classifies named entities. This chapter covers BIO tagging, CRF, BiLSTM-CRF, and BERT-NER.

---

## 1. 任務定義 | Task Definition

**輸入**：文本序列  
**輸出**：實體的**邊界**和**類型**

**常見實體類型**：
- **PER** (Person)：人名
- **LOC** (Location)：地名
- **ORG** (Organization)：機構名
- **MISC** (Miscellaneous)：其他
- **TIME、DATE、MONEY、PERCENT** 等

**範例**：
```
原文：[CLS] 馬斯克 是 Tesla 的 CEO [SEP]
標註：O    B-PER  O  B-ORG O O
```

---

## 2. 標註方案 | Tagging Schemes

### 2.1 BIO 標註

- **B-X**：實體 X 的**開始**詞
- **I-X**：實體 X 的**內部**詞
- **O**：非實體

### 2.2 BIOES 標註（更精細）

- **B-X**：開始
- **I-X**：內部
- **E-X**：結束
- **S-X**：單個詞實體
- **O**：非實體

### 2.3 示例對比

| 詞 | BIO | BIOES |
|----|-----|-------|
| 馬 | B-PER | B-PER |
| 斯 | I-PER | I-PER |
| 克 | I-PER | E-PER |
| 在 | O | O |
| 台 | B-LOC | B-LOC |
| 北 | I-LOC | E-LOC |

```python
# 構建標籤映射
tag2id = {
    "O": 0,
    "B-PER": 1, "I-PER": 2,
    "B-LOC": 3, "I-LOC": 4,
    "B-ORG": 5, "I-ORG": 6,
    "B-MISC": 7, "I-MISC": 8,
    "[PAD]": 9,  # 用於填充
}

id2tag = {v: k for k, v in tag2id.items()}
```

---

## 3. 特徵工程方法 | Feature-based Methods

### 3.1 傳統特徵

- **詞性 (POS)**：人名通常是名詞
- **詞形**：首字母大寫、地名關鍵詞
- **詞典**：gazetteer 匹配
- **詞綴**：-tion, -ment（機構）
- **前後文窗口**

### 3.2 CRF (Conditional Random Field)

CRF 通過考慮**相鄰標籤轉移**對序列標註建模：

$$P(y | x) = \frac{1}{Z(x)} \exp\left(\sum_{t} \sum_k \lambda_k f_k(y_{t-1}, y_t, x_t)\right)$$

```python
# pip install sklearn-crfsuite
import sklearn_crfsuite
from sklearn_crfsuite import metrics

def word2features(sent, i):
    """提取特徵"""
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
        })
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
        })
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# 訓練 CRF
X_train = [sent2features(s) for s in train_sents]
y_train = [s[1] for s in train_sents]  # 標籤列表

crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1)
crf.fit(X_train, y_train)
```

---

## 4. 深度學習方法 | Deep Learning

### 4.1 BiLSTM-CRF

經典架構：BiLSTM 編碼 + CRF 解碼

```python
import torch
import torch.nn as nn

class BiLSTM_CRF(nn.Module):
    """BiLSTM + CRF 命名實體識別"""
    
    def __init__(self, vocab_size, tag2id, embed_dim=100, hidden_dim=200):
        super().__init__()
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, 
                            bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # CRF 轉移矩陣：transitions[i][j] = 從標籤 j 到標籤 i 的分數
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag2id["[START]"]:, :] = -10000
        self.transitions.data[:, tag2id["[END]"]] = -10000
        
        self.start_tag = tag2id["[START]"]
        self.end_tag = tag2id["[END]"]
    
    def forward(self, x, mask=None):
        embeds = self.embedding(x)  # (batch, seq, embed)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)  # (batch, seq, tag_size)
        return emissions
    
    def crf_forward(self, emissions, tags, mask):
        """CRF 前向算法，計算正確序列的分數"""
        batch_size, seq_len, _ = emissions.shape
        
        score = torch.zeros(batch_size)
        for i in range(batch_size):
            valid_len = mask[i].sum() if mask is not None else seq_len
            for t in range(valid_len):
                emit_score = emissions[i, t, tags[i, t]]
                if t == 0:
                    trans_score = self.transitions[tags[i, t], self.start_tag]
                else:
                    trans_score = self.transitions[tags[i, t], tags[i, t-1]]
                score[i] += emit_score + trans_score
            # 加上到 END 的轉移
            score[i] += self.transitions[self.end_tag, tags[i, valid_len-1]]
        return score
    
    def crf_decode(self, emissions, mask=None):
        """Viterbi 解碼"""
        # 簡化實現：實際應用 Viterbi 算法
        return torch.argmax(emissions, dim=-1)
```

### 4.2 BERT-NER

```python
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    Trainer, TrainingArguments, DataCollatorForTokenClassification
)

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    num_labels=len(tag2id),
    id2label=id2tag,
    label2id=tag2id,
)

# 對齊標籤：BERT tokenizer 會將詞分成多個 subword
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        padding=True,
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # subword 設為 -100（忽略）
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="./bert_ner",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
```

### 4.3 推理：從 token 標籤恢復實體

```python
def extract_entities(tokens, predictions, id2tag):
    """從 BIO 標籤恢復實體"""
    entities = []
    current_entity = None
    
    for token, pred_id in zip(tokens, predictions):
        tag = id2tag[pred_id]
        
        if tag.startswith("B-"):
            # 開始新實體
            if current_entity:
                entities.append(current_entity)
            current_entity = {"text": token, "type": tag[2:]}
        elif tag.startswith("I-"):
            # 繼續實體
            if current_entity and current_entity["type"] == tag[2:]:
                current_entity["text"] += token
            else:
                # 標籤不連續，重置
                if current_entity:
                    entities.append(current_entity)
                current_entity = None
        else:  # "O"
            if current_entity:
                entities.append(current_entity)
            current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

# 使用
text = "馬斯克是Tesla的CEO"
tokens = list(text)
predictions = model_predict(tokens)  # 假設返回 tag ids
entities = extract_entities(tokens, predictions, id2tag)
print(entities)
# [{'text': '馬斯克', 'type': 'PER'}, {'text': 'Tesla', 'type': 'ORG'}]
```

---

## 5. 中文 NER 特殊處理 | Chinese-specific

### 5.1 中文分詞對 NER 的影響

中文沒有顯式詞邊界，分詞錯誤會影響 NER。

**解決方案**：
1. 使用字符級 NER（中文主流）
2. 聯合分詞 + NER
3. 使用預訓練的中文模型（自動處理）

### 5.2 預訓練中文 NER 模型

```python
# 推薦的開源中文 NER 模型
MODELS = {
    "ckiplab/bert-base-chinese-ner": "繁體中文 SOTA",
    "bert-base-chinese": "簡體中文通用",
    "hfl/chinese-roberta-wwm-ext": "用全詞遮罩預訓練",
    "xlm-roberta-base": "多語言，跨語言遷移",
}

# 使用
from transformers import pipeline

ner = pipeline("ner", 
               model="ckiplab/bert-base-chinese-ner",
               aggregation_strategy="average")

text = "張三在台北的台積電工作，負責先進製程研發。"
results = ner(text)

for r in results:
    print(f"{r['word']:6s} | {r['entity_group']:6s} | score={r['score']:.4f}")
```

---

## 6. 評估 | Evaluation

### 6.1 嚴格評估 (Strict)

實體的邊界和類型都必須完全匹配：

```python
from seqeval.metrics import classification_report, f1_score

# y_true 和 y_pred 是標籤字符串列表
y_true = [["B-PER", "I-PER", "O", "B-LOC"]]
y_pred = [["B-PER", "I-PER", "O", "B-ORG"]]

print(f"F1 (strict): {f1_score(y_true, y_pred):.4f}")
print(classification_report(y_true, y_pred))
```

### 6.2 寬鬆評估 (Lenient)

只評估實體類型是否正確，邊界允許部分重疊。

### 6.3 基準性能 (CoNLL-2003)

| 模型 | F1 |
|------|-----|
| BiLSTM-CRF | 91.0 |
| BERT-base | 92.8 |
| BERT-large | 93.5 |
| RoBERTa-large | 95.6 |
| DeBERTa-v3-large | 96.1 |

---

## 7. 數據增強 | Data Augmentation

```python
# 1. 實體替換
def entity_replacement(tokens, labels, entity_dict):
    """用同類型實體替換"""
    new_tokens, new_labels = [], []
    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            entity_type = label[2:]
            if entity_type in entity_dict and entity_dict[entity_type]:
                new_token = random.choice(entity_dict[entity_type])
                new_tokens.append(new_token)
            else:
                new_tokens.append(token)
        else:
            new_tokens.append(token)
        new_labels.append(label)
    return new_tokens, new_labels

# 2. 標籤保持的詞替換（使用同義詞詞典）
# 3. 回譯增強
```

---

## 8. 常見問題 | Common Issues

### 8.1 嵌套實體 | Nested Entities

"北京大學" 既包含 "北京" (LOC) 又包含 "北京大學" (ORG)。

**解決方法**：
- **層次標註**：使用多層標籤
- **Span-based 方法**：直接預測所有可能的 span
- **Pi-RCNN、ARN**：候選 span 分類

### 8.2 非連續實體 | Discontinuous Entities

"中國和美國" → 應為兩個獨立的 LOC 實體，但容易被識別為一個。

**解決方案**：
- **Transition-based NER**
- **Hypergraph 結構**

### 8.3 實體鏈接 | Entity Linking

識別出實體後，鏈接到知識庫（Wikipedia、YAGO）。

```python
# 流程：
# 1. NER 識別實體
# 2. 候選實體生成（從 KB 找候選）
# 3. 實體消歧（基於上下文）
# 4. 鏈接到 KB ID
```

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│  NER 方法發展路徑                                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  傳統特徵工程                                                │
│  └── 規則 + 詞典 + 特徵 + CRF (F1 ~ 80%)                    │
│                                                              │
│  深度學習                                                    │
│  ├── BiLSTM-CRF (F1 ~ 90%)                                  │
│  ├── BiLSTM + Attention                                     │
│  └── IDCNN (Iterated Dilated CNN)                           │
│                                                              │
│  預訓練模型                                                  │
│  ├── BERT-NER (F1 ~ 93%)                                    │
│  ├── RoBERTa-NER (F1 ~ 95%)                                 │
│  └── DeBERTa-NER (F1 ~ 96%)                                 │
│                                                              │
│  統一框架                                                    │
│  ├── Universal NER (UniNER)                                 │
│  ├── Universal Information Extraction (UIE)                 │
│  └── GPT-4 Few-shot (F1 ~ 90% for many languages)           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 參考文獻 | References

1. Huang, Z., et al. (2015). Bidirectional LSTM-CRF Models for Sequence Tagging. *arXiv*.
2. Devlin, J., et al. (2018). BERT. *NAACL*.
3. Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 Shared Task. *CoNLL*.
4. Li, J., et al. (2020). Dice Loss for Data-imbalanced NLP Tasks. *ACL*.
5. Tjiputra, A., et al. (2022). Nested Named Entity Recognition. *ACM Computing Surveys*.

---

*© 2026 chanxaviersy*
