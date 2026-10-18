# Lab 04 數據集：中文命名實體識別 | Chinese NER Dataset

> 中文 NER 數據集 - 包含 MSRA / People Daily / 自定義格式

---

## 📊 自動下載開源 NER 數據集

```python
import os
import urllib.request
import json

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_msra_ner():
    """下載 MSRA 簡體中文 NER 數據集"""
    base = "https://raw.githubusercontent.com/lemonhu/NER-data/master/MSRA"
    files = ["train.txt", "test.txt"]
    for f in files:
        url = f"{base}/{f}"
        target = f"{DATA_DIR}/msra_{f}"
        print(f"下載 {url} ...")
        urllib.request.urlretrieve(url, target)
    print("✅ MSRA 數據集下載完成")

def download_renmin_ner():
    """下載人民日報 NER 數據集"""
    url = "https://raw.githubusercontent.com/InsaneLife/ChineseNER/master/data/example.train"
    target = f"{DATA_DIR}/renmin.txt"
    print(f"下載 {url} ...")
    urllib.request.urlretrieve(url, target)
    print("✅ 人民日報數據集下載完成")

# 執行下載
download_msra_ner()
# download_renmin_ner()
```

---

## 📋 數據格式

### BIO 標註格式

每行一個字符，後跟其標籤；句子之間用空行分隔。

**範例**：
```
張	B-PER
三	I-PER
在	O
台	B-LOC
北	I-LOC
的	O
台	B-ORG
積	I-ORG
電	I-ORG
工	O
作	O
。	O

李	B-PER
四	I-PER
是	O
台	B-LOC
灣	I-LOC
大	B-ORG
學	I-ORG
的	O
學	O
生	O
。	O
```

---

## 🏷️ 標籤集

| 類別 | 縮寫 | 含義 | 範例 |
|------|------|------|------|
| **PER** | Person | 人名 | 張三、馬斯克 |
| **LOC** | Location | 地名 | 台灣、北京 |
| **ORG** | Organization | 機構名 | 台積電、台大 |
| **TIME** | Time | 時間 | 今天、2023年 |
| **MISC** | Miscellaneous | 其他 | 事件、產品名 |

### 完整標籤集（9 類）

```python
tag2id = {
    "O": 0,
    "B-PER": 1, "I-PER": 2,
    "B-LOC": 3, "I-LOC": 4,
    "B-ORG": 5, "I-ORG": 6,
    "B-MISC": 7, "I-MISC": 8,
}
```

---

## 📊 內建示例數據集

本倉庫 `data/sample_train.txt` 提供 200 條標註好的繁體中文 NER 數據（自建），可離線使用：

```python
def load_ner_data(filepath):
    """加載 BIO 格式 NER 數據"""
    sentences = []
    labels = []
    tokens = []
    tags = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens = []
                    tags = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                tags.append(parts[-1])
    
    if tokens:
        sentences.append(tokens)
        labels.append(tags)
    
    return sentences, labels

sentences, labels = load_ner_data("data/sample_train.txt")
print(f"句數: {len(sentences)}")
print(f"示例: {list(zip(sentences[0], labels[0]))}")
```

---

## 🚀 開源中文 NER 數據集對比

| 數據集 | 規模 | 語言 | 實體類別 | 來源 |
|--------|------|------|----------|------|
| **MSRA** | 50K | 簡體 | 3 (PER/LOC/ORG) | 微軟亞洲研究院 |
| **People Daily** | 28K | 簡體 | 4 (PER/LOC/ORG/TIME) | 人民日報 |
| **OntoNotes** | 15K | 多語言 | 18 | 開放數據集 |
| **Weibo** | 1.5K | 簡體 | 4 (社交媒體) | 新浪微博 |
| **CLUENER** | 12K | 簡體 | 10 | 細粒度 |

---

## 🔗 相關資源

- **MSRA**: https://github.com/lemonhu/NER-data
- **CLUENER**: https://github.com/CLUEbenchmark/CLUENER
- **CKIP Lab**: https://github.com/ckiplab/ckiptagger
- **HuggingFace**: https://huggingface.co/datasets?search=ner+chinese

---

*© 2026 chanxaviersy*
