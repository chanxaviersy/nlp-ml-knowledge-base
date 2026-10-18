# 🧠 Lab 02: CNN 文本分類 | Text Classification with CNN

> 摘要：本 Lab 使用 PyTorch 實現 TextCNN 模型進行中文文本分類，對比不同 filter size、embedding 預訓練的影響。  
> Abstract: This lab implements TextCNN with PyTorch for Chinese text classification and compares different filter sizes and pretrained embeddings.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chanxaviersy/nlp-ml-knowledge-base/blob/main/docs/08-labs/02-text-classification-cnn/lab.ipynb)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)
![GPU Recommended](https://img.shields.io/badge/GPU-recommended-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)

---

## 🎯 目標 | Objectives

- 從零實現 TextCNN 模型
- 學習詞嵌入的使用與訓練
- 理解卷積核在文本上的作用
- 掌握 PyTorch 訓練流程

---

## 📋 環境準備 | Environment

```bash
pip install torch torchvision jieba scikit-learn pandas numpy matplotlib
```

---

## 📊 步驟 1: 數據準備 | Data Preparation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import jieba
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 設定隨機種子
torch.manual_seed(42)
np.random.seed(42)

# 載入數據（這裡使用小型示例數據集，實際可換成大數據集）
texts = [
    "這部電影真的很棒，劇情精彩，演員演技出色",
    "非常推薦這部電影，感人至深",
    "很喜歡這個產品，質量好",
    "服務態度很好，下次還會再來",
    "糟糕透了，完全是浪費時間",
    "質量太差，與描述不符",
    "服務態度惡劣，差評",
    "不推薦購買，體驗非常糟糕",
] * 50  # 放大數據集
labels = [1, 1, 1, 1, 0, 0, 0, 0] * 50

df = pd.DataFrame({"text": texts, "label": labels})
print(f"數據集大小: {len(df)}")
```

---

## 🔧 步驟 2: 分詞與詞表構建 | Tokenization & Vocabulary

```python
# 1. 分詞
df['tokens'] = df['text'].apply(lambda x: list(jieba.cut(x)))

# 2. 構建詞表
from collections import Counter

def build_vocab(tokens_list, min_freq=2):
    counter = Counter()
    for tokens in tokens_list:
        counter.update(tokens)
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(df['tokens'].tolist(), min_freq=2)
print(f"詞表大小: {len(vocab)}")

# 3. 將 token 轉為索引
def encode(tokens, vocab, max_len=20):
    indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if len(indices) < max_len:
        indices += [vocab["<PAD>"]] * (max_len - len(indices))
    return indices[:max_len]

MAX_LEN = 20
df['ids'] = df['tokens'].apply(lambda x: encode(x, vocab, MAX_LEN))

print(f"示例編碼: {df['ids'].iloc[0]}")
print(f"解碼: {[list(vocab.keys())[i] for i in df['ids'].iloc[0] if i != 0]}")
```

---

## 🏗️ 步驟 3: TextCNN 模型 | TextCNN Model

```python
class TextCNN(nn.Module):
    """
    TextCNN 模型
    
    架構:
    Embedding → Multi-size Conv1D → Max Pooling → Concat → FC → Output
    """
    
    def __init__(self, vocab_size, embed_dim, num_classes, 
                 filter_sizes=[2, 3, 4, 5], num_filters=100, 
                 dropout=0.5, pretrained_embeddings=None):
        super().__init__()
        
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # 凍結
        
        # 多個不同大小的卷積核
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=fs,
            )
            for fs in filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 全連接層
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        """
        x: (batch, seq_len)
        """
        # Embedding: (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # 轉換維度: (batch, embed_dim, seq_len) - Conv1D 期望的格式
        embedded = embedded.permute(0, 2, 1)
        
        # 卷積 + ReLU + Max Pooling
        conv_outputs = []
        for conv in self.convs:
            # conv: (batch, num_filters, seq_len - kernel_size + 1)
            conv_out = torch.relu(conv(embedded))
            # max pool: (batch, num_filters, 1)
            pooled = torch.max_pool1d(conv_out, conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))  # (batch, num_filters)
        
        # 拼接所有 filter 的輸出
        concat = torch.cat(conv_outputs, dim=1)  # (batch, len(filter_sizes) * num_filters)
        concat = self.dropout(concat)
        
        # 全連接
        logits = self.fc(concat)  # (batch, num_classes)
        return logits

# 創建模型
VOCAB_SIZE = len(vocab)
EMBED_DIM = 128
NUM_CLASSES = 2

model = TextCNN(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_classes=NUM_CLASSES,
    filter_sizes=[2, 3, 4, 5],
    num_filters=100,
    dropout=0.5,
)

print(model)
print(f"參數量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 🔧 步驟 4: 自定義數據集 | Custom Dataset

```python
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# 切分數據
X_train, X_test, y_train, y_test = train_test_split(
    df['ids'].tolist(), df['label'].tolist(), 
    test_size=0.2, random_state=42, stratify=df['label']
)

train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"訓練集: {len(train_dataset)}, 測試集: {len(test_dataset)}")
```

---

## 🔧 步驟 5: 訓練循環 | Training Loop

```python
# 訓練配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 訓練
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += len(labels)
    
    return total_loss / total, total_correct / total


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            logits = model(texts)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(all_labels), accuracy, all_preds, all_labels


# 訓練循環
NUM_EPOCHS = 20
train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc, _, _ = evaluate(model, test_loader, device)
    
    scheduler.step()
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
```

---

## 🔧 步驟 6: 結果可視化 | Visualize Results

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 損失曲線
axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(test_losses, label='Test Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Test Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 準確率曲線
axes[1].plot(train_accs, label='Train Acc')
axes[1].plot(test_accs, label='Test Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Test Accuracy')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 🔧 步驟 7: 評估與分析 | Evaluation

```python
# 最終評估
test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, device)
print(f"\n最終測試準確率: {test_acc:.4f}")
print(f"\n分類報告:\n{classification_report(y_true, y_pred, target_names=['負面', '正面'])}")

# 混淆矩陣
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['負面', '正面'], yticklabels=['負面', '正面'])
plt.xlabel('預測')
plt.ylabel('實際')
plt.title('TextCNN 混淆矩陣')
plt.show()
```

---

## 🔧 步驟 8: 實驗：不同 Filter Size 的影響

```python
filter_configs = [
    ([2, 3], "small"),
    ([3, 4, 5], "medium"),
    ([2, 3, 4, 5], "all"),
    ([3, 4, 5, 6, 7], "large"),
]

results = {}

for filters, name in filter_configs:
    model_exp = TextCNN(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES,
        filter_sizes=filters, num_filters=100, dropout=0.5
    ).to(device)
    
    optimizer_exp = optim.Adam(model_exp.parameters(), lr=1e-3)
    
    # 簡化訓練
    for _ in range(10):
        model_exp.train()
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer_exp.zero_grad()
            loss = criterion(model_exp(texts), labels)
            loss.backward()
            optimizer_exp.step()
    
    _, acc, _, _ = evaluate(model_exp, test_loader, device)
    results[name] = acc
    print(f"Filter sizes {filters} ({name}): Accuracy = {acc:.4f}")

# 可視化
plt.bar(results.keys(), results.values(), color='teal')
plt.ylabel('Accuracy')
plt.title('Effect of Filter Sizes')
plt.ylim([0, 1.1])
plt.grid(axis='y', alpha=0.3)
plt.show()
```

---

## 🔧 步驟 9: 預測新樣本 | Predict on New Samples

```python
def predict_text(text, model, vocab, max_len=20):
    """預測單個文本"""
    model.eval()
    tokens = list(jieba.cut(text))
    ids = encode(tokens, vocab, max_len)
    input_tensor = torch.tensor([ids]).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = logits.argmax(dim=1).item()
    
    label = "正面 😊" if pred == 1 else "負面 😞"
    return label, probs.cpu().numpy()

# 測試
test_texts = [
    "這個產品太棒了，強烈推薦",
    "質量很差，物流也很慢",
    "還可以，沒有特別的感覺",
    "服務一流，體驗很棒",
    "非常糟糕的一次購物",
]

for text in test_texts:
    label, probs = predict_text(text, model, vocab)
    print(f"文本: {text}")
    print(f"預測: {label} (負面: {probs[0]:.2%}, 正面: {probs[1]:.2%})\n")
```

---

## 🎓 思考題 | Exercises

1. **預訓練詞嵌入**：使用預訓練的中文詞向量（如 Tencent AI Lab Embedding）初始化 embedding
2. **多通道**：實現 static + non-static 雙通道 embedding
3. **數據增強**：嘗試 EDA（Easy Data Augmentation）等文本增強
4. **更深的網路**：在 TextCNN 基礎上加入多層卷積
5. **與 LSTM 對比**：實現 BiLSTM 模型，對比 CNN 的優缺點

---

## 📊 預期結果 | Expected Results

| 配置 | 準確率 | 備註 |
|------|--------|------|
| Random Init, Filter=[2,3,4,5] | 0.85+ | 基線 |
| Pretrained Embedding | 0.92+ | 大幅提升 |
| Filter=[3,4,5,6,7] | 0.88+ | 略好 |
| Dropout 0.3 vs 0.5 | 0.83 vs 0.85 | 視數據集 |

---

## 📚 參考 | References

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP.](https://arxiv.org/abs/1408.5882)
- [PyTorch 官方教程](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
- [TextCNN 論文解讀](https://zhuanlan.zhihu.com/p/77633433)

---

## ⚡ 快速開始 | Quick Start

```bash
cd docs/08-labs/02-text-classification-cnn
python lab.py
# 訓練 10 個 epoch，視覺化 loss/acc 曲線與混淆矩陣
```

需要 `torch >= 2.0`、`jieba`、`datasets`。

---

## 🧪 練習題 | Exercises

| # | 難度 | 題目 |
|---|------|------|
| 1 | ⭐ | 改變 `kernel_sizes=(3, 4, 5)` 為 `(2,3,4,5,6)`，觀察效果 |
| 2 | ⭐⭐ | 用預訓練的 word2vec / Tencent Embedding 替換隨機初始化 |
| 3 | ⭐⭐ | 實現 **DPCNN**（深層金字塔 CNN）對比 TextCNN |
| 4 | ⭐⭐⭐ | 雙通道 embedding（static + non-static） |
| 5 | ⭐⭐⭐ | 加入 Self-Attention 機制（即 TextRCNN） |
| 6 | ⭐⭐⭐⭐ | 在 4 萬條 THUCNews 上做多分類（10+ 類） |

---

## 📊 預期結果 | Expected Results

| 配置 | 準確率 | 備註 |
|------|--------|------|
| Random Init, Filter=[2,3,4,5] | 0.85+ | 基線 |
| Pretrained Embedding | 0.92+ | 大幅提升 |
| Filter=[3,4,5,6,7] | 0.88+ | 略好 |
| Dropout 0.3 vs 0.5 | 0.83 vs 0.85 | 視數據集 |

---

## 🔧 關鍵技巧 | Key Tips

- **Padding 必須設為 0**，這樣 max-pooling 不會選到 padding
- **kernel size 不要太大**（3-5 最佳），太大感受野過大會損失信息
- **多尺度融合**：用不同 kernel 提取不同 n-gram 特徵
- **Dropout 0.3-0.5** 在卷積後、FC 前最有效
- **批次大小** 16-64，視顯存而定

---

*© 2026 chanxaviersy*
