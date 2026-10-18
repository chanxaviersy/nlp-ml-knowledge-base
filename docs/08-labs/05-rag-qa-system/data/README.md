# Lab 05 數據集：RAG 文檔集合 | RAG Document Collection

> 包含多個領域的範例文檔，適合建立 RAG 問答系統測試

---

## 📊 文檔集合

本 Lab 使用以下範例文檔（自動生成，可在 `data/documents/` 找到）：

| 領域 | 文檔數 | 主題 |
|------|--------|------|
| **科技公司** | 4 | 台積電、NVIDIA、Google、Microsoft |
| **教育** | 2 | 台灣大學、史丹佛大學 |
| **地理** | 3 | 台灣、日本東京、美國加州 |
| **人物** | 3 | 張忠謀、黃仁勳、Elon Musk |
| **技術概念** | 3 | 人工智慧、區塊鏈、量子計算 |

---

## 📄 範例文檔

### 台積電 (TSMC)

```
台積電（Taiwan Semiconductor Manufacturing Company，TSMC）是全球最大的半導體代工廠，
總部位於台灣新竹科學園區，由張忠謀於1987年創立。

台積電的主要產品包括各種先進製程的晶圓代工服務，目前最先進的製程為3奈米。
公司的主要客戶包括蘋果（Apple）、NVIDIA、AMD、Qualcomm等全球領先的科技公司。

台積電在全球設有多座晶圓廠，包括台灣的新竹、台中、台南，以及美國亞利桑那州、日本熊本等地。
2023年台積電市值超過5000億美元，是亞洲市值最大的公司之一。
```

### NVIDIA

```
NVIDIA 是一家以 GPU（圖形處理器）和 AI 晶片聞名的美國公司，由黃仁勳於 1993 年在美國加州創立。
NVIDIA 的 GeForce 系列 GPU 在遊戲市場佔據主導地位，而 Tesla、H100 等數據中心 GPU 廣泛用於人工智慧訓練。

NVIDIA 在 AI 晶片市場佔據主導地位。2024年NVIDIA 的市值曾突破 3 萬億美元，超越蘋果成為全球市值最大的公司之一。

NVIDIA 的 CUDA 編程平台徹底改變了通用 GPU 計算，是深度學習革命的關鍵基礎設施之一。
```

### 人工智慧 (AI)

```
人工智慧（Artificial Intelligence，AI）是計算機科學的一個分支，致力於創建能夠執行通常需要人類智慧的任務的系統。

現代 AI 的核心技術包括機器學習（Machine Learning）、深度學習（Deep Learning）和自然語言處理（NLP）。
深度學習基於神經網路，特別是 Transformer 架構，已經徹底改變了語言理解、圖像識別、語音合成等領域。

近年來，大語言模型（LLM）如 GPT-4、Claude、Gemini 等展現出強大的通用智慧能力，能夠進行對話、寫作、編程、推理等多種任務。
```

### 區塊鏈 (Blockchain)

```
區塊鏈（Blockchain）是一種去中心化的分布式帳本技術，最著名的應用是比特幣（Bitcoin）。

區塊鏈的核心特性包括：
1. 去中心化：無需中央機構驗證交易
2. 不可篡改：一旦記錄，難以修改
3. 透明公開：所有交易可查詢
4. 智能合約：自動執行的合約（如以太坊）

區塊鏈技術的應用已擴展到金融、供應鏈、數字身份、版權保護等多個領域。
```

### 量子計算 (Quantum Computing)

```
量子計算（Quantum Computing）是一種利用量子力學原理進行計算的新型計算模式。

與傳統計算機使用位元（bit，0 或 1）不同，量子計算機使用量子位元（qubit），
可以同時表示 0 和 1 的疊加態（superposition），這使得量子計算機在某些問題上具有指數級加速。

量子計算的潛在應用包括：
- 密碼學：破解現有加密算法，開發量子安全加密
- 藥物發現：模擬分子結構
- 優化問題：物流、金融建模
- 機器學習：加速 AI 訓練

目前 IBM、Google、Microsoft 等公司都在積極研發量子計算技術。
```

### 張忠謀

```
張忠謀（台灣，1931年7月10日出生），被譽為「台灣半導體教父」，是台積電的創辦人。

張忠謀曾在美國德州儀器（Texas Instruments）擔任資深副總裁，1987年在台灣創立台積電，
開創了晶圓代工（Foundry）的商業模式，徹底改變了全球半導體產業格局。

張忠謀於 2018 年退休，但他對台灣半導體產業的影響深遠至今。
```

### Elon Musk

```
Elon Musk（1971年6月28日出生）是南非裔美國企業家，目前擔任特斯拉（Tesla）和 SpaceX 的 CEO。

Musk 的主要成就包括：
1. PayPal：聯合創辦網路支付公司，後被 eBay 收購
2. SpaceX：創立太空探索技術公司，目標殖民火星
3. Tesla：引領電動車革命
4. Neuralink：腦機接口技術
5. X（原 Twitter）：2022年收購社群媒體平台

Musk 也是 OpenAI 的聯合創辦人之一，但後來離開了董事會。
```

---

## 📥 自動下載更多文檔

```python
import os
import requests

DATA_DIR = "data/documents"
os.makedirs(DATA_DIR, exist_ok=True)

# 從 Wikipedia 下載文檔（範例）
def download_wikipedia_articles(articles):
    """下載 Wikipedia 條目"""
    for title in articles:
        url = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{title}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            content = f"{data['title']}\n\n{data.get('extract', '')}"
            with open(f"{DATA_DIR}/{title}.txt", 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 已下載: {title}")

articles = ["台積電", "NVIDIA", "人工智慧", "區塊鏈", "量子計算", "黃仁勳", "张忠谋"]
download_wikipedia_articles(articles)
```

---

## 🏗️ 構建 RAG 系統

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 加載文檔
loader = DirectoryLoader("data/documents", glob="**/*.txt", show_progress=True)
docs = loader.load()

# 2. 分塊
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 構建向量庫
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("data/faiss_index")

print(f"✅ 已加載 {len(docs)} 個文檔，分成 {len(chunks)} 個塊")
```

---

## 🧪 測試問題

以下是測試 RAG 系統的範例問題：

| 問題 | 期望答案涵蓋 |
|------|--------------|
| 台積電的創辦人是誰？ | 張忠謀 |
| NVIDIA 的創辦人是誰？ | 黃仁勳 |
| 台積電的總部在哪裡？ | 新竹 |
| 什麼是區塊鏈？ | 去中心化、分布式帳本 |
| 量子計算使用什麼單位？ | 量子位元 (qubit) |
| Elon Musk 創辦了哪些公司？ | Tesla, SpaceX, PayPal |

---

*© 2026 chanxaviersy*
