# Lab 02 數據集：文本分類數據 | Text Classification Dataset

> 用於 TextCNN 文本分類的範例數據集

---

## 📊 自動下載大型數據集

```python
import os
import pandas as pd
from datasets import load_dataset

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 方法 1：使用 ChnSentiCorp（情感分類）
dataset = load_dataset("lansinuote/ChnSentiCorp", split="train[:5000]")
df = pd.DataFrame({
    "text": dataset["text"],
    "label": dataset["label"],
})
df.to_csv(f"{DATA_DIR}/sentiment_train.csv", index=False)

# 方法 2：使用 THUCNews（新聞分類）
# 中文長文本新聞分類
thucnews = load_dataset("thucnews", split="train[:5000]", trust_remote_code=True)
df_thuc = pd.DataFrame({
    "text": thucnews["text"],
    "label": thucnews["label"],
})
df_thuc.to_csv(f"{DATA_DIR}/thucnews_sample.csv", index=False)

print(f"已下載 {len(df)} 條情感數據")
print(f"已下載 {len(df_thuc)} 條新聞數據")
```

---

## 📋 範例數據（sample.csv）

本倉庫提供 200 條 2 類別情感數據作為離線示例：

```python
import pandas as pd
df = pd.read_csv("data/sample.csv")
print(f"樣本數: {len(df)}")
print(df['label'].value_counts())
```

---

## 🔧 多分類數據集

```python
# 使用 TNEWS 短文本分類
from datasets import load_dataset
tnews = load_dataset("clue", "tnews", split="train[:5000]", trust_remote_code=True)
df_tnews = pd.DataFrame({
    "text": tnews["text"],
    "label": tnews["label"],
})
# 15+ 個類別：財經、體育、科技、娛樂...
```

---

## 📚 推薦中文文本分類數據集

| 數據集 | 類別 | 規模 | 場景 |
|--------|------|------|------|
| **ChnSentiCorp** | 2 (正/負) | 9.6K | 情感分析 |
| **THUCNews** | 14 | 74K | 新聞分類 |
| **TNEWS** | 17 | 73K | 短新聞 |
| **IFLYTEK** | 119 | 17K | APP 描述 |
| **Waimai** | 2 | 12K | 外賣評論 |
| **Online_shopping** | 2 | 60K | 電商評論 |

---

*© 2026 chanxaviersy*
