# 🏷️ 文本分類 | Text Classification

> 摘要：文本分類是 NLP 最基礎的任務，將文本分入預定義類別。本章介紹傳統方法（Naive Bayes、SVM）、深度學習方法（TextCNN、BERT）以及多標籤、不平衡等進階問題。  
> Abstract: Text classification is the most fundamental NLP task. This chapter covers traditional methods (Naive Bayes, SVM), deep learning (TextCNN, BERT), and advanced issues like multi-label and imbalance.

---

## 1. 任務定義 | Task Definition

**輸入**：文本序列 $T = (w_1, w_2, ..., w_n)$  
**輸出**：類別 $y \in \{1, 2, ..., K\}$

**常見任務**：
- 情感分析（正面/負面/中性）
- 主題分類（體育/科技/政治）
- 垃圾郵件檢測
- 意圖識別（客服/智能助手）
- 假新聞檢測
- 仇恨言論檢測

---

## 2. 傳統方法 | Traditional Methods

### 2.1 文本表示

**詞袋模型 (Bag-of-Words)**：

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "我喜歡這部電影",
    "這部電影很精彩",
    "我討厭這部電影",
    "電影很無聊",
]

# 詞袋
bow = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
X = bow.fit_transform(corpus)
print("詞彙表:", bow.get_feature_names_out())
print("BoW:\n", X.toarray())
```

**TF-IDF**：

```python
tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
X_tfidf = tfidf.fit_transform(corpus)
print("TF-IDF:\n", X_tfidf.toarray().round(3))
```

### 2.2 樸素貝葉斯

```python
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred = nb.predict(X_test_tfidf)
```

**優點**：極快、適合高維稀疏特徵  
**缺點**：條件獨立假設太強

### 2.3 SVM

```python
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
    ('svm', LinearSVC(C=1.0)),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### 2.4 邏輯回歸

```python
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr', LogisticRegression(max_iter=1000)),
])
```

**傳統方法基線性能**（IMDB 情感分類）：

| 方法 | 準確率 |
|------|--------|
| Majority Class | 50% |
| TF-IDF + LR | 88-89% |
| TF-IDF + SVM | 89-90% |
| TF-IDF + NB | 85-87% |

---

## 3. 深度學習方法 | Deep Learning

### 3.1 TextCNN (Kim, 2014)

對文本做 1D 卷積，捕捉不同長度的 n-gram 特徵：

```python
import torch
import torch.nn as nn

class TextCNN(nn.Module):
    """TextCNN 文本分類器"""
    
    def __init__(self, vocab_size, embed_dim, num_classes, 
                 filter_sizes=[3, 4, 5], num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, 
                     out_channels=num_filters, 
                     kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)              # (batch, seq_len, embed_dim)
        embedded = embedded.permute(0, 2, 1)      # (batch, embed_dim, seq_len)
        
        # 對每個 filter size 進行卷積
        conv_outputs = []
        for conv in self.convs:
            conv_out = nn.functional.relu(conv(embedded))  # (batch, num_filters, L)
            pooled = nn.functional.max_pool1d(conv_out, conv_out.size(2))  # (batch, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))
        
        # 拼接所有 filter 的輸出
        concat = torch.cat(conv_outputs, dim=1)  # (batch, len(filter_sizes) * num_filters)
        concat = self.dropout(concat)
        logits = self.fc(concat)
        return logits
```

### 3.2 BiLSTM + Attention

```python
class BiLSTMAttention(nn.Module):
    """BiLSTM 配合注意力機制"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, 
                            bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))  # (batch, seq, embed)
        lstm_out, _ = self.lstm(embedded)           # (batch, seq, 2*hidden)
        
        # Attention
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        return self.fc(context)
```

### 3.3 BERT 微調

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 數據準備
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)

# 訓練
training_args = TrainingArguments(
    output_dir="./text_clf",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()
```

**BERT 性能**（IMDB）：
- BERT-base：92-93%
- BERT-large：94-95%
- RoBERTa-large：96%

---

## 4. 進階主題 | Advanced Topics

### 4.1 多標籤分類 | Multi-Label Classification

每個樣本可以有多個標籤：

```python
# 方法 1：Sigmoid + BCE Loss（每個類別獨立）
class MultiLabelClassifier(nn.Module):
    def __init__(self, model, num_labels):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(model.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # 對每個類別用 Sigmoid（不是 Softmax）
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        return torch.sigmoid(logits)

# 損失：BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()
```

### 4.2 層次分類 | Hierarchical Classification

類別有層次結構（如：體育 > 足球 > 英超）：

```python
class HierarchicalClassifier(nn.Module):
    """層次分類：先分大類，再分小類"""
    
    def __init__(self, model, num_coarse, num_fine):
        super().__init__()
        self.model = model
        self.coarse_classifier = nn.Linear(model.config.hidden_size, num_coarse)
        self.fine_classifier = nn.Linear(model.config.hidden_size, num_fine)
    
    def forward(self, input_ids, attention_mask):
        h = self.model(input_ids, attention_mask).last_hidden_state[:, 0]
        return {
            'coarse': self.coarse_classifier(h),
            'fine': self.fine_classifier(h),
        }
```

### 4.3 零樣本分類 | Zero-Shot Classification

無需訓練，直接用 NLI 模型分類：

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", 
                      model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")

text = "這部電影的視覺效果令人驚嘆，但劇情薄弱"
candidate_labels = ["正面", "負面", "中性"]
result = classifier(text, candidate_labels)

for label, score in zip(result["labels"], result["scores"]):
    print(f"{label}: {score:.4f}")
```

### 4.4 少樣本分類 | Few-Shot

```python
from openai import OpenAI

client = OpenAI()

prompt = """你是文本分類器。將以下評論分類為：positive 或 negative。

示範：
評論：「很棒的產品！」 → positive
評論：「完全不推薦」 → negative

現在分類：
評論：「{text}」 →"""

def classify_few_shot(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt.format(text=text)}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

print(classify_few_shot("東西還行，但物流太慢"))
```

---

## 5. 處理類別不平衡 | Handling Imbalance

### 5.1 數據層面

| 方法 | 描述 |
|------|------|
| **隨機過採樣** | 複製少數類樣本 |
| **隨機欠採樣** | 隨機刪除多數類 |
| **SMOTE** | 合成少數類樣本 |
| **ADASYN** | 自適應合成 |

### 5.2 損失函數層面

```python
# Focal Loss：關注難分類樣本
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# 類別權重
weights = torch.tensor([1.0, 5.0, 1.0])  # 少數類權重高
criterion = nn.CrossEntropyLoss(weight=weights)
```

---

## 6. 解釋性 | Interpretability

### 6.1 LIME

```python
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['negative', 'positive'])
exp = explainer.explain_instance(
    text_sample,
    classifier.predict_proba,
    num_features=10,
)
exp.show_in_notebook()
```

### 6.2 SHAP

```python
import shap

explainer = shap.Explainer(model)
shap_values = explainer(X_test_texts)

# 可視化
shap.plots.text(shap_values[0])
```

### 6.3 Attention 可視化

```python
# 提取 BERT attention 權重
outputs = model(**inputs, output_attentions=True)
attention = outputs.attentions[-1]  # 最後一層 attention

# 可視化特定頭
from bertviz import head_view
head_view(attention, tokens)
```

---

## 7. 評估 | Evaluation

### 7.1 指標

| 任務 | 推薦指標 |
|------|----------|
| 二分類（平衡） | Accuracy |
| 二分類（不平衡） | F1, PR-AUC |
| 多分類 | Macro/Weighted F1 |
| 多標籤 | Hamming Loss, Subset Accuracy, F1 (samples) |

### 7.2 混淆矩陣分析

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_true, y_pred, target_names=class_names))

# 找出最常被誤分類的對
cm = confusion_matrix(y_true, y_pred)
errors = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] > 0:
            errors.append((class_names[i], class_names[j], cm[i, j]))
errors.sort(key=lambda x: -x[2])
for true, pred, count in errors[:5]:
    print(f"{true} → {pred}: {count} 次")
```

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│  文本分類方法選擇                                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  快速基線 (1-2 小時)                                         │
│  └── TF-IDF + LR / SVM / NB                                │
│                                                              │
│  深度學習 (幾小時)                                            │
│  └── TextCNN / BiLSTM                                       │
│                                                              │
│  最佳性能 (1-2 天)                                          │
│  └── BERT / RoBERTa / DeBERTa 微調                          │
│                                                              │
│  極小樣本 (<100)                                             │
│  └── Few-shot prompting with GPT-4 / Claude                 │
│                                                              │
│  零樣本                                                        │
│  └── NLI-based zero-shot                                    │
│                                                              │
│  特定場景                                                     │
│  ├── 不平衡 → Focal Loss / Class Weight                     │
│  ├── 多標籤 → Sigmoid + BCE                                 │
│  ├── 層次 → 分層分類                                        │
│  └── 大類別 (10K+) → 層次 softmax                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 參考文獻 | References

1. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *EMNLP*.
2. Devlin, J., et al. (2018). BERT. *NAACL*.
3. Liu, Y., et al. (2019). RoBERTa. *arXiv*.
4. Sun, C., et al. (2019). How to Fine-Tune BERT for Text Classification. *arXiv*.
5. Lin, T. Y., et al. (2017). Focal Loss. *ICCV*.

---

*© 2026 chanxaviersy*
