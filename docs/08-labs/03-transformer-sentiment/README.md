# Lab 03: Transformer 情感分析 | Transformer-based Sentiment Analysis

> 摘要：本 Lab 使用 HuggingFace Transformers 微調 BERT 進行中文情感分類。  
> Abstract: This lab fine-tunes BERT for Chinese sentiment classification using HuggingFace Transformers.

---

## 🎯 目標 | Objectives

- 學習使用 HuggingFace Transformers
- 微調預訓練 BERT 模型
- 理解訓練配置（學習率、warmup、評估策略）
- 評估和部署模型

---

## 📋 環境準備 | Environment

```bash
pip install transformers datasets torch evaluate scikit-learn pandas numpy matplotlib accelerate
```

---

## 📊 步驟 1: 載入數據集 | Load Dataset

```python
from datasets import load_dataset
import pandas as pd

# 方法 1：使用 HuggingFace Hub 數據集
# 這裡用 ChnSentiCorp 作為示例
dataset = load_dataset("lansinuote/ChnSentiCorp", split="train[:5000]+test[:1000]")
print(f"數據集大小: {len(dataset)}")
print(f"示例: {dataset[0]}")

# 方法 2：使用自定義數據
custom_data = {
    "text": [
        "這部電影真的太棒了",
        "服務態度極差，差評",
        # ...
    ],
    "label": [1, 0],
}
df = pd.DataFrame(custom_data)
```

---

## 🔧 步驟 2: 數據預處理 | Data Preprocessing

```python
from transformers import AutoTokenizer

# 載入中文 BERT tokenizer
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    """對文本進行分詞"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

# 應用到整個數據集
tokenized_dataset = dataset.map(preprocess_function, batched=True)
print(f"Tokenized 示例: {tokenized_dataset[0]}")

# 切分訓練/測試集
train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"訓練集: {len(train_dataset)}, 測試集: {len(eval_dataset)}")
```

---

## 🔧 步驟 3: 載入模型 | Load Model

```python
from transformers import AutoModelForSequenceClassification

# 載入預訓練 BERT，添加分類頭
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # 二分類：正面/負面
    id2label={0: "負面", 1: "正面"},
    label2id={"負面": 0, "正面": 1},
)

print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")
print(f"模型結構:\n{model.classifier}")
```

---

## 🔧 步驟 4: 配置訓練參數 | Training Configuration

```python
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

# 1. 評估指標
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    """計算評估指標"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    
    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"],
    }

# 2. 訓練參數
training_args = TrainingArguments(
    output_dir="./bert_sentiment",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",  # 不上報到 W&B
    fp16=True if torch.cuda.is_available() else False,  # 混合精度
)

# 3. Data Collator
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```

---

## 🔧 步驟 5: 訓練 | Training

```python
# 開始訓練
trainer.train()

# 保存最終模型
trainer.save_model("./bert_sentiment_final")
tokenizer.save_pretrained("./bert_sentiment_final")
```

**預期訓練輸出**：
```
Epoch 1: {'eval_loss': 0.32, 'eval_accuracy': 0.92, 'eval_f1': 0.92}
Epoch 2: {'eval_loss': 0.25, 'eval_accuracy': 0.94, 'eval_f1': 0.94}
Epoch 3: {'eval_loss': 0.23, 'eval_accuracy': 0.95, 'eval_f1': 0.95}
```

---

## 🔧 步驟 6: 評估 | Evaluation

```python
# 完整評估
eval_results = trainer.evaluate()
print(f"最終評估結果: {eval_results}")

# 預測測試集
predictions = trainer.predict(eval_dataset)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

# 分類報告
from sklearn.metrics import classification_report
print("\n分類報告:")
print(classification_report(y_true, y_pred, target_names=["負面", "正面"]))

# 混淆矩陣
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['負面', '正面'], yticklabels=['負面', '正面'])
plt.xlabel('預測')
plt.ylabel('實際')
plt.title('BERT 混淆矩陣')
plt.show()
```

---

## 🔧 步驟 7: 推理 | Inference

```python
from transformers import pipeline

# 方法 1：使用 pipeline
classifier = pipeline(
    "text-classification",
    model="./bert_sentiment_final",
    tokenizer="./bert_sentiment_final",
    device=0 if torch.cuda.is_available() else -1,
)

# 預測
texts = [
    "這部電影真的太精彩了，演員演技出色",
    "質量太差，完全不推薦",
    "服務態度很好，下次還會再來",
    "非常糟糕的一次購物體驗",
    "還行，沒有什麼特別的",
]

results = classifier(texts)
for text, result in zip(texts, results):
    label = "正面 😊" if result["label"] == "正面" else "負面 😞"
    print(f"[{label} (score: {result['score']:.4f})] {text}")

# 方法 2：手動推理
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("./bert_sentiment_final")
tokenizer = AutoTokenizer.from_pretrained("./bert_sentiment_final")

def predict_manual(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred = torch.argmax(probs).item()
    
    label = "正面" if pred == 1 else "負面"
    confidence = probs[pred].item()
    return label, confidence, probs.numpy()

label, conf, probs = predict_manual("這個產品很棒")
print(f"預測: {label} (信心: {conf:.2%}, 負面/正面概率: {probs})")
```

---

## 🔧 步驟 8: 處理長文本 | Long Text Handling

```python
# 滑動窗口處理長文本
def predict_long_text(text, model, tokenizer, max_length=512, stride=128, threshold=0.5):
    """對長文本分段預測並聚合"""
    inputs = tokenizer(
        text, 
        return_tensors="pt",
        truncation=False,  # 不截斷
    )
    
    input_ids = inputs["input_ids"][0]
    
    # 切分成多個窗口
    chunks = []
    for i in range(0, len(input_ids), max_length - stride):
        chunk = input_ids[i:i + max_length]
        chunks.append(chunk)
        if i + max_length >= len(input_ids):
            break
    
    # 預測每個 chunk
    all_probs = []
    for chunk in chunks:
        chunk_tensor = chunk.unsqueeze(0)
        with torch.no_grad():
            logits = model(chunk_tensor).logits
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs[0].numpy())
    
    # 平均概率
    avg_prob = np.mean(all_probs, axis=0)
    pred = np.argmax(avg_prob)
    
    label = "正面" if pred == 1 else "負面"
    return label, avg_prob

# 測試
long_text = """這部電影的開頭有些平淡，讓我一度想放棄。但隨著劇情深入，
演員的表演越來越精彩，劇情的反轉也讓人驚喜連連。特別是最後一幕，
情感的爆發讓我感動得流淚。總體來說，這是一部值得推薦的好電影。"""

label, probs = predict_long_text(long_text, model, tokenizer)
print(f"長文本預測: {label}")
print(f"負面概率: {probs[0]:.4f}, 正面概率: {probs[1]:.4f}")
```

---

## 🔧 步驟 9: 模型解釋 | Model Interpretation

```python
# 使用 SHAP 解釋預測
import shap

# 創建預測函數
def f(x):
    inputs = tokenizer(list(x), padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=-1).numpy()

# 解釋
text = ["這部電影真的很精彩，演員演技出色"]
explainer = shap.Explainer(f, shap.maskers.Text(tokenizer))
shap_values = explainer(text)

# 可視化
shap.plots.text(shap_values[0])
```

---

## 🎓 思考題 | Exercises

1. **嘗試不同預訓練模型**：`bert-base-chinese`、`hfl/chinese-roberta-wwm-ext`、`hfl/chinese-macbert-base`
2. **多分類**：改為 5 星評分預測
3. **數據增強**：使用回譯、EDA 等方法擴充訓練集
4. **超參數搜索**：使用 Optuna 搜索最佳學習率、batch size
5. **蒸餾**：將 BERT-large 蒸餾到 BERT-tiny
6. **部署**：用 ONNX 或 TensorRT 加速推理

---

## 📊 預期結果 | Expected Results

| 模型 | 準確率 | F1 | 訓練時間 |
|------|--------|-----|----------|
| **TF-IDF + LR** (Lab 01) | ~0.88 | ~0.88 | < 1 min |
| **TextCNN** (Lab 02) | ~0.90 | ~0.90 | ~2 min |
| **BERT-base** (本 Lab) | ~0.95 | ~0.95 | ~10 min |
| **BERT-large** | ~0.96 | ~0.96 | ~30 min |
| **RoBERTa-large** | ~0.97 | ~0.97 | ~30 min |

---

## 🔧 常見問題 | FAQ

**Q: 訓練時 GPU 記憶體不足怎麼辦？**  
A: 減小 `per_device_train_batch_size`，或啟用 `gradient_accumulation_steps`、`gradient_checkpointing`。

**Q: 學習率該如何設置？**  
A: BERT 微調通常 1e-5 到 5e-5。RoBERTa 略高（2e-5 到 1e-4）。

**Q: 為什麼訓練效果不如預期？**  
A: 檢查數據品質、類別分佈、學習率。中文任務優先選擇 `chinese-roberta-wwm-ext`。

---

## 📚 參考 | References

- [BERT 論文](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers 文檔](https://huggingface.co/docs/transformers)
- [Chinese-BERT 預訓練模型](https://github.com/ymcui/Chinese-BERT-wwm)

---

## ⚡ 快速開始 | Quick Start

```bash
cd docs/08-labs/03-transformer-sentiment
pip install transformers datasets accelerate
python lab.py
# 預設使用 3000 條訓練 + 1000 條測試（可在程式碼中調整）
```

如需在 Apple Silicon 上加速，PyTorch 會自動使用 MPS。

---

## 🧪 練習題 | Exercises

| # | 難度 | 題目 |
|---|------|------|
| 1 | ⭐ | 改用 `hfl/chinese-roberta-wwm-ext` 對比效果 |
| 2 | ⭐⭐ | 嘗試 `bert-base-chinese` vs `hfl/chinese-macbert-base` |
| 3 | ⭐⭐ | 加上 **混合精度訓練** (fp16) 加速 |
| 4 | ⭐⭐ | 用 **LoRA** 微調（見 `05-transformers/03-llm-finetuning`）|
| 5 | ⭐⭐⭐ | 嘗試 **層凍結**：只訓練最後 4 層 |
| 6 | ⭐⭐⭐ | 加入 **對抗訓練** (FGM/PGD) |
| 7 | ⭐⭐⭐⭐ | 部署為 FastAPI / Gradio 服務 |

---

## 🔧 微調最佳實踐 | Best Practices

```python
# 推薦超參
EPOCHS = 3-5           # 太多會過擬合
LR = 2e-5              # BERT 黃金學習率
BATCH = 16             # 視顯存調整
MAX_LEN = 128          # 電商評論足夠
WARMUP = 0.1           # 10% warmup
```

- **資料增強**：回譯（zh→en→zh）、同義詞替換
- **對抗訓練**：FGM/PGD
- **多任務學習**：加入語言模型頭
- **集成**：5 個不同 seed 模型取平均

---

## 📈 預期表現 | Expected Performance

| 模型 | F1 | 訓練時間（GPU）|
|------|------|------|
| bert-base-chinese | 0.93+ | ~3 分鐘 |
| chinese-roberta-wwm-ext | 0.94+ | ~3 分鐘 |
| chinese-macbert-base | 0.94+ | ~4 分鐘 |
| xlm-roberta-base | 0.92+ | ~5 分鐘 |

---

*© 2026 chanxaviersy*
