# Lab 04: BERT 命名實體識別 | NER with BERT

> 摘要：本 Lab 使用 BERT 微調實現中文命名實體識別（NER），識別 PER、LOC、ORG 等實體類型。  
> Abstract: This lab fine-tunes BERT for Chinese Named Entity Recognition.

---

## 🎯 目標 | Objectives

- 理解 BIO 標註方案
- 處理 subword 對齊問題
- 微調 BERT 用於序列標註
- 提取實體並評估

---

## 📋 環境準備 | Environment

```bash
pip install transformers datasets seqeval torch accelerate
```

---

## 📊 步驟 1: 數據集準備 | Dataset

```python
from datasets import load_dataset

# 載入中文 NER 數據集（MSRA 或自定義）
# 方法 1：使用 HuggingFace 數據集
dataset = load_dataset("msra_ner", split="train[:5000]")
# 注意：需要登錄 HuggingFace 或使用其他鏡像

# 方法 2：使用自定義數據
# 數據格式：tokens 列表 + ner_tags 列表
example_data = {
    "tokens": ["張", "三", "在", "台", "北", "的", "台", "積", "電", "工", "作", "。"],
    "ner_tags": [1, 2, 0, 3, 4, 0, 5, 6, 6, 0, 0, 0],
    #         B-PER I-PER  O  B-LOC I-LOC O  B-ORG I-ORG I-ORG O    O  O
}

# 我們創建示例數據集
import pandas as pd

train_data = []
texts_with_entities = [
    ("張三在台積電工作。", [("張三", "PER"), ("台積電", "ORG")]),
    ("李四是台灣大學的學生。", [("李四", "PER"), ("台灣大學", "ORG")]),
    ("王五在台北的咖啡廳喝咖啡。", [("王五", "PER"), ("台北", "LOC")]),
    # ... 更多樣本
]

# 構建 BIO 標註
def create_bio_tags(text, entities):
    """創建 BIO 標註"""
    tags = ["O"] * len(text)
    
    for entity, entity_type in entities:
        start = text.find(entity)
        if start != -1:
            tags[start] = f"B-{entity_type}"
            for i in range(1, len(entity)):
                tags[start + i] = f"I-{entity_type}"
    
    return tags

# 收集所有標籤
all_tags = set()
for text, entities in texts_with_entities:
    tags = create_bio_tags(text, entities)
    all_tags.update(tags)

tag2id = {tag: i for i, tag in enumerate(sorted(all_tags))}
id2tag = {v: k for k, v in tag2id.items()}

print(f"標籤集: {tag2id}")
```

---

## 🔧 步驟 2: 數據預處理 | Preprocessing

```python
from transformers import AutoTokenizer

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    """
    對齊 subword 和標籤
    
    重要：BERT 使用 WordPiece 分詞，中文每個字符通常是一個 token。
    但對於罕見字或英文詞，可能會被拆成多個 subword。
    對齊策略：subword 設為 -100，PyTorch 會自動忽略。
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128,
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # 特殊 token（[CLS], [SEP], [PAD]）
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # 這是新詞的第一個 subword，使用對應標籤
                label_ids.append(label[word_idx])
            else:
                # 同一詞的後續 subword，標記為 -100（忽略）
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized["labels"] = labels
    return tokenized

# 應用
# 假設 dataset 是 HuggingFace Dataset
# tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
```

---

## 🔧 步驟 3: 載入模型 | Load Model

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(tag2id),
    id2label=id2tag,
    label2id=tag2id,
)

print(f"參數量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 🔧 步驟 4: 訓練配置 | Training Setup

```python
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import evaluate

# 1. 評估指標（使用 seqeval）
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    """使用 seqeval 評估實體級 F1"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    
    # 將 -100 過濾掉
    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 2. 訓練參數
training_args = TrainingArguments(
    output_dir="./bert_ner",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    fp16=True if torch.cuda.is_available() else False,
)

# 3. Data Collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

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
trainer.train()
trainer.save_model("./bert_ner_final")
```

---

## 🔧 步驟 6: 實體提取 | Entity Extraction

```python
import torch

def extract_entities(text, model, tokenizer, id2tag):
    """從文本中提取實體"""
    # 分詞
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"][0]
    
    # 預測
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    
    # 將 token 索引轉換回 token
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # 提取實體
    entities = []
    current_entity = None
    
    for token, pred_id in zip(tokens, predictions):
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        
        tag = id2tag[pred_id]
        
        if tag.startswith("B-"):
            # 開始新實體
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "text": token.replace("##", ""),
                "type": tag[2:],
                "start": len("".join([e["text"] for e in entities])) + len(current_entity["text"] if current_entity else ""),
            }
        elif tag.startswith("I-") and current_entity and tag[2:] == current_entity["type"]:
            # 繼續實體
            current_entity["text"] += token.replace("##", "")
        else:
            # O 或標籤不連續
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

# 使用
text = "張三在台積電工作，他的同事李四住在新竹。"
entities = extract_entities(text, model, tokenizer, id2tag)

for entity in entities:
    print(f"實體: {entity['text']:10s} 類型: {entity['type']}")

# 應到原文
def align_entities_to_text(text, entities):
    """將實體對齊到原文中的位置"""
    aligned = []
    for entity in entities:
        e_text = entity["text"]
        start = text.find(e_text)
        if start != -1:
            aligned.append({
                "text": e_text,
                "type": entity["type"],
                "start": start,
                "end": start + len(e_text),
            })
    return aligned

aligned = align_entities_to_text(text, entities)
for e in aligned:
    print(f"[{e['start']:2d}:{e['end']:2d}] {e['type']:5s} → {e['text']}")
```

---

## 🔧 步驟 7: 批量預測 | Batch Prediction

```python
def predict_batch(texts, model, tokenizer, id2tag, batch_size=16):
    """批量預測多個文本"""
    model.eval()
    all_entities = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            logits = model(**encodings).logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        
        for j, (text, pred) in enumerate(zip(batch, predictions)):
            tokens = tokenizer.convert_ids_to_tokens(encodings["input_ids"][j])
            
            # 提取實體（簡化版）
            entities = []
            current = None
            for token, p in zip(tokens, pred):
                if token in ("[CLS]", "[SEP]", "[PAD]"):
                    continue
                tag = id2tag[p]
                if tag.startswith("B-"):
                    if current:
                        entities.append(current)
                    current = {"text": token.replace("##", ""), "type": tag[2:]}
                elif tag.startswith("I-") and current and tag[2:] == current["type"]:
                    current["text"] += token.replace("##", "")
                else:
                    if current:
                        entities.append(current)
                    current = None
            if current:
                entities.append(current)
            
            all_entities.append({"text": text, "entities": entities})
    
    return all_entities

# 測試
texts = [
    "馬斯克是Tesla的CEO。",
    "蘋果公司在加州庫比蒂諾。",
    "張三畢業於台大，現在在Google工作。",
]

results = predict_batch(texts, model, tokenizer, id2tag)
for r in results:
    print(f"\n文本: {r['text']}")
    for e in r['entities']:
        print(f"  {e['type']:5s}: {e['text']}")
```

---

## 🔧 步驟 8: 評估 | Evaluation

```python
# 完整評估
eval_results = trainer.evaluate()
print(f"評估結果: {eval_results}")

# 實體級評估
predictions = trainer.predict(eval_dataset)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

# 轉換為標籤字符串
true_predictions = [
    [id2tag[p] for (p, l) in zip(pred, label) if l != -100]
    for pred, label in zip(y_pred, y_true)
]
true_labels = [
    [id2tag[l] for (p, l) in zip(pred, label) if l != -100]
    for pred, label in zip(y_pred, y_true)
]

# 詳細分類報告
from seqeval.metrics import classification_report
print(classification_report(true_labels, true_predictions))
```

---

## 🔧 步驟 9: 構建 NER Pipeline

```python
class NERPipeline:
    """NER 推理 Pipeline"""
    
    def __init__(self, model, tokenizer, id2tag, device='cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.id2tag = id2tag
        self.device = device
        self.model.eval()
    
    def __call__(self, text):
        return self.extract(text)
    
    def extract(self, text):
        """提取實體"""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # 提取實體
        entities = []
        current = None
        char_pos = 0
        
        for token, pred_id in zip(tokens, predictions):
            if token in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            
            tag = self.id2tag[pred_id]
            clean_token = token.replace("##", "")
            
            if tag.startswith("B-"):
                if current:
                    entities.append(current)
                current = {
                    "text": clean_token,
                    "type": tag[2:],
                    "start": char_pos,
                }
                char_pos += len(clean_token)
            elif tag.startswith("I-") and current and tag[2:] == current["type"]:
                current["text"] += clean_token
                char_pos += len(clean_token)
            else:
                if current:
                    entities.append(current)
                current = None
                char_pos += len(clean_token)
        
        if current:
            entities.append(current)
        
        # 對齊到原文
        for entity in entities:
            start = text.find(entity["text"])
            if start != -1:
                entity["start"] = start
                entity["end"] = start + len(entity["text"])
        
        return entities

# 使用
ner = NERPipeline(model, tokenizer, id2tag, device='cpu')
text = "馬斯克是Tesla的CEO，公司總部位於加州。"
entities = ner(text)
for e in entities:
    print(f"{e['text']:10s} ({e['type']:5s}) @ [{e['start']}:{e['end']}]")
```

---

## 🎓 思考題 | Exercises

1. **嵌套實體**：嘗試識別嵌套實體（如 "北京大學" 包含 "北京"）
2. **更多實體類型**：添加 MISC、TIME、DATE 等
3. **CRF 層**：在 BERT 之上加 CRF 層
4. **領域適配**：用金融、醫療領域數據繼續微調
5. **實體鏈接**：將識別出的實體鏈接到知識庫

---

## 📊 預期結果 | Expected Results

| 模型 | F1 | 備註 |
|------|-----|------|
| BiLSTM-CRF | ~0.88 | 經典基線 |
| BERT-base | ~0.93 | 大幅提升 |
| BERT-large | ~0.95 | 進一步提升 |
| RoBERTa | ~0.96 | 中文 SOTA |

---

## 📚 參考 | References

- [BERT for NER](https://huggingface.co/docs/transformers/tasks/token_classification)
- [中文 NER 數據集](https://github.com/SophonPlus/ChineseNlpCorpus)
- [CKIP Lab 中文 NER](https://github.com/ckiplab/ckip-transformers)

---

*© 2026 chanxaviersy*
