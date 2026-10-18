# Lab 03 數據集：ChnSentiCorp 情感分析 | Sentiment Analysis Dataset

> 完整資料集 + 自動下載腳本  
> 規模：~9,600 條 ChnSentiCorp 全量

---

## 📊 自動下載完整數據集

### 方法 1：HuggingFace（推薦）

```python
import os
import pandas as pd
from datasets import load_dataset

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 載入完整 ChnSentiCorp
print("下載 ChnSentiCorp 數據集...")
train_data = load_dataset("lansinuote/ChnSentiCorp", split="train")
test_data = load_dataset("lansinuote/ChnSentiCorp", split="test")

# 保存
df_train = pd.DataFrame({
    "text": train_data["text"],
    "label": train_data["label"],
})
df_test = pd.DataFrame({
    "text": test_data["text"],
    "label": test_data["label"],
})

df_train.to_csv(f"{DATA_DIR}/train_full.csv", index=False)
df_test.to_csv(f"{DATA_DIR}/test_full.csv", index=False)

print(f"訓練集: {len(df_train)} 條")
print(f"測試集: {len(df_test)} 條")
print(f"類別比例: {df_train['label'].value_counts().to_dict()}")
```

### 方法 2：直接下載 TSV 文件

```python
import urllib.request

urls = {
    "train.tsv": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp/ChnSentiCorp_htl_ba_2000/train.tsv",
    "test.tsv": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp/ChnSentiCorp_htl_ba_2000/test.tsv",
    "dev.tsv": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp/ChnSentiCorp_htl_ba_2000/dev.tsv",
}

for filename, url in urls.items():
    print(f"下載 {filename}...")
    urllib.request.urlretrieve(url, f"data/{filename}")
```

---

## 📊 數據集概覽

| 屬性 | 值 |
|------|-----|
| **來源** | 攜程網酒店評論 |
| **規模** | 訓練 ~6,000 / 測試 ~1,200 / 驗證 ~1,200 |
| **類別** | 1=好評, 0=差評 |
| **平均長度** | ~80 字 |
| **領域** | 酒店、電商、旅遊 |

---

## 🔍 數據格式

```csv
text,label
"房間乾淨整潔，服務態度很好，下次還會選擇這裏。",1
"酒店位置不便，房間狹小，與圖片相差甚遠。",0
```

**標籤說明**：
- `1`: 正面評價（好評）
- `0`: 負面評價（差評）

---

## 🚀 快速開始

```python
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 載入數據
df = pd.read_csv("data/train_full.csv")
print(f"訓練集: {len(df)} 條樣本")
print(df.head())

# 載入模型
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 開始微調（見主 Lab 03）
```

---

## 📚 相關資源

- **HuggingFace**: https://huggingface.co/datasets/lansinuote/ChnSentiCorp
- **GitHub**: https://github.com/SophonPlus/ChineseNlpCorpus
- **論文**: 有關中文情感分類的多項研究都使用此數據集

---

*© 2026 chanxaviersy*
