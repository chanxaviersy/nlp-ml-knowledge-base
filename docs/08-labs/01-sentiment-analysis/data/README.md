# Lab 01 數據集：中文情感分析 | Sentiment Analysis Dataset

> 資料來源：混合資料集（ChnSentiCorp 子集 + 合成資料）  
> 規模：5,000 條訓練 / 1,000 條測試  
> 語言：簡體中文

---

## 📊 資料集說明

| 屬性 | 值 |
|------|-----|
| **任務** | 二分類情感分析（正面 / 負面） |
| **樣本數** | 6,000 條 |
| **來源** | 開源 ChnSentiCorp 數據集 + 自動生成 |
| **格式** | CSV (`text`, `label`) |
| **label** | 1=正面, 0=負面 |
| **平均長度** | ~50 字 |

---

## 📥 下載數據集

```python
import os
import urllib.request
import zipfile
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 方法 1：直接從 HuggingFace 下載
from datasets import load_dataset

dataset = load_dataset("lansinuote/ChnSentiCorp", split="train")
df_train = pd.DataFrame({
    "text": dataset["text"],
    "label": dataset["label"],
})
df_train.to_csv(f"{DATA_DIR}/train.csv", index=False)
print(f"訓練集: {len(df_train)} 條")

# 載入測試集
test_dataset = load_dataset("lansinuote/ChnSentiCorp", split="test")
df_test = pd.DataFrame({
    "text": test_dataset["text"],
    "label": test_dataset["label"],
})
df_test.to_csv(f"{DATA_DIR}/test.csv", index=False)
print(f"測試集: {len(df_test)} 條")

# 方法 2：使用 sample 數據（包含在倉庫中）
# data/sample_train.csv - 200 條小樣本
# data/sample_test.csv - 100 條小樣本
```

---

## 🔍 數據預覽

```python
import pandas as pd

df = pd.read_csv("data/train.csv")
print(f"數據集形狀: {df.shape}")
print(f"\n前 5 條:")
print(df.head())

print(f"\n類別分佈:")
print(df['label'].value_counts())

print(f"\n文本長度統計:")
df['length'] = df['text'].str.len()
print(df['length'].describe())
```

**範例數據**：

| text | label |
|------|-------|
| 這家餐廳的菜很好吃，服務也很周到，下次還會再來 | 1 |
| 完全不符合描述，質量差，物流也慢 | 0 |
| 非常滿意的一次購物體驗，五星好評 | 1 |
| 退貨流程複雜，客服態度差 | 0 |

---

## 📈 數據分析

```python
import matplotlib.pyplot as plt
from collections import Counter
import jieba

# 1. 類別分佈
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df['label'].value_counts().plot(kind='bar', ax=axes[0], color=['#ff7f7f', '#7fbf7f'])
axes[0].set_title('Class Distribution')
axes[0].set_xlabel('Label (0=Negative, 1=Positive)')
axes[0].set_ylabel('Count')

# 2. 文本長度分佈
df['length'].hist(bins=50, ax=axes[1])
axes[1].set_title('Text Length Distribution')
axes[1].set_xlabel('Characters')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# 3. 高頻詞分析
all_tokens = []
for text in df['text']:
    all_tokens.extend(jieba.lcut(text))

counter = Counter(all_tokens)
print("\nTop 20 高頻詞:")
for word, count in counter.most_common(20):
    print(f"  {word:8s}: {count}")
```

---

## 🔗 相關資源

- **ChnSentiCorp**: https://github.com/SophonPlus/ChineseNlpCorpus
- **HuggingFace**: https://huggingface.co/datasets/lansinuote/ChnSentiCorp
- **其他中文情感數據集**:
  - 網易雲音樂評論
  - 豆瓣影評
  - 微博情感

---

## 📝 引用

```bibtex
@dataset{chnsenti,
  title={ChnSentiCorp: A Chinese Sentiment Classification Dataset},
  author={SophonPlus},
  year={2019},
  url={https://github.com/SophonPlus/ChineseNlpCorpus}
}
```

---

*© 2026 chanxaviersy*
