# 📚 文檔索引 | Documentation Index

> 完整的 NLP & ML 知識庫導航，按學習路徑排序。  
> Complete navigation of the NLP & ML knowledge base, organized by learning path.

---

## 🎯 學習路徑推薦 | Recommended Learning Path

```
入門 (Beginner)
    ↓
01-基礎 → 02-ML → 03-DL → 04-NLP基礎 → 05-Transformer
    ↓
進階 (Intermediate)
    ↓
06-應用 → 08-Labs (做中學)
    ↓
高級 (Advanced)
    ↓
07-前沿 → 09-擴展 → 進階論文閱讀
```

---

## 📖 完整索引 | Complete Index

### 一、基礎知識 Foundations (Part 1)
- [📐 數學基礎 Mathematics](docs/01-foundations/01-mathematics/) — 微積分、機率論、資訊理論
- [📊 統計學 Statistics](docs/01-foundations/02-statistics/) — 推論統計、貝氏方法、A/B 測試
- [🔢 線性代數 Linear Algebra](docs/01-foundations/03-linear-algebra/) — 向量、矩陣、SVD、特徵分解

### 二、機器學習 Machine Learning (Part 2)
- [📈 監督學習 Supervised](docs/02-machine-learning/01-supervised-learning/) — 線性/邏輯回歸、SVM、樹模型、集成
- [🔍 無監督學習 Unsupervised](docs/02-machine-learning/02-unsupervised-learning/) — 聚類、降維、關聯規則
- [📏 模型評估 Evaluation](docs/02-machine-learning/03-model-evaluation/) — 指標、CV、Optuna 調參

### 三、深度學習 Deep Learning (Part 3)
- [🧠 神經網路 Neural Networks](docs/03-deep-learning/01-neural-networks/) — MLP、Backprop、激活函數
- [⚙️ 優化 Optimization](docs/03-deep-learning/02-optimization/) — SGD、Adam、學習率調度
- [🛡️ 正則化 Regularization](docs/03-deep-learning/03-regularization/) — Dropout、BN、知識蒸餾

### 四、NLP 基礎 NLP Fundamentals (Part 4)
- [✂️ 文本預處理 Preprocessing](docs/04-nlp-fundamentals/01-text-preprocessing/) — 分詞、停用詞、n-gram
- [🔤 詞嵌入 Embeddings](docs/04-nlp-fundamentals/02-word-embeddings/) — Word2Vec、GloVe、FastText
- [📝 語言模型 Language Models](docs/04-nlp-fundamentals/03-language-models/) — N-gram、RNN、LSTM、GRU

### 五、Transformer 與 LLM (Part 5)
- [🎯 注意力機制 Attention](docs/05-transformers/01-attention-mechanism/) — Self/Cross-Attention、多頭
- [🤖 BERT & GPT](docs/05-transformers/02-bert-gpt/) — 預訓練、微調、編碼器/解碼器
- [🎛️ LLM 微調 Fine-tuning](docs/05-transformers/03-llm-finetuning/) — LoRA、QLoRA、RLHF、DPO

### 六、應用場景 Applications (Part 6)
- [🏷️ 文本分類 Classification](docs/06-applications/01-text-classification/) — 情感、主題、垃圾
- [🔎 命名實體識別 NER](docs/06-applications/02-named-entity-recognition/) — BIO、BiLSTM-CRF、BERT-NER
- [📰 摘要 Summarization](docs/06-applications/03-summarization/) — TextRank、BART、T5、PEGASUS
- [❓ 問答系統 QA](docs/06-applications/04-question-answering/) — SQuAD、Multi-hop、Conversational
- [🌐 機器翻譯 MT](docs/06-applications/05-machine-translation/) — Seq2Seq、MarianMT、NLLB
- [✨ 文本生成 Text Generation](docs/06-applications/06-text-generation/) — Decoding、CoT、ToT

### 七、前沿專題 Advanced Topics (Part 7)
- [📚 RAG 檢索增強生成](docs/07-advanced-topics/01-rag/) — LangChain、FAISS、Re-ranking
- [🤖 AI Agents](docs/07-advanced-topics/02-agents/) — ReAct、Function Calling、LangGraph
- [🎨 多模態 Multimodal](docs/07-advanced-topics/03-multimodal/) — CLIP、BLIP-2、LLaVA、SD

### 八、實戰 Lab (Part 8)
| Lab | 主題 | 難度 | 數據集 | 技術棧 |
|-----|------|------|--------|--------|
| [Lab 01](docs/08-labs/01-sentiment-analysis/) | 情感分析 | ⭐ | ChnSentiCorp | sklearn, TF-IDF |
| [Lab 02](docs/08-labs/02-text-classification-cnn/) | CNN 分類 | ⭐⭐ | ChnSentiCorp | PyTorch |
| [Lab 03](docs/08-labs/03-transformer-sentiment/) | BERT 情感 | ⭐⭐ | ChnSentiCorp | HuggingFace |
| [Lab 04](docs/08-labs/04-ner-with-bert/) | BERT NER | ⭐⭐⭐ | MSRA | HuggingFace |
| [Lab 05](docs/08-labs/05-rag-qa-system/) | RAG QA | ⭐⭐ | 8 範本文檔 | LangChain + FAISS |

### 九、前沿擴展 Cutting-Edge (Part 9)
- [🎮 強化學習 RL](docs/09-advanced/01-reinforcement-learning/) — DQN、PPO、SAC、RLHF、DPO
- [🎤 語音識別 Speech](docs/09-advanced/02-speech-recognition/) — MFCC、CTC、Whisper、TTS
- [🎯 推薦系統 Recommender](docs/09-advanced/03-recommender-systems/) — CF、DeepFM、SASRec、LLM
- [🕸️ 圖神經網路 GNN](docs/09-advanced/04-graph-neural-networks/) — GCN、GAT、GraphSAGE

---

## 🛠 工具與腳本 | Tools & Scripts

| 工具 | 路徑 | 描述 |
|------|------|------|
| 通用工具 | [scripts/utils.py](scripts/utils.py) | 隨機種子、可視化、評估 |
| 預處理 | [scripts/text_preprocessing.py](scripts/text_preprocessing.py) | 中文/英文預處理管道 |
| 訓練器 | [scripts/trainer.py](scripts/trainer.py) | PyTorch Trainer 類 |
| 評估指標 | [scripts/metrics.py](scripts/metrics.py) | 分類/NER/生成/RAG 指標 |
| 數據載入 | [datasets/loader.py](datasets/loader.py) | 統一數據集接口 |

---

## 📊 數據集總覽 | Dataset Overview

| 數據集 | 任務 | 規模 | 語言 |
|--------|------|------|------|
| ChnSentiCorp | 情感分類 | ~9.6K | 中文 |
| MSRA | NER | ~50K | 中文 |
| THUCNews | 新聞分類 | ~74K | 中文 |
| TNEWS | 短文本分類 | ~73K | 中文 |
| SQuAD | 閱讀理解 | ~100K | 英文 |
| CoNLL-2003 | NER | ~22K | 英文 |
| IMDB | 情感分類 | ~50K | 英文 |
| AG News | 主題分類 | ~127K | 英文 |

執行 `python datasets/loader.py --info` 查看完整列表。

---

## 📌 重點章節 | Highlights

### 剛開始學？| Just Starting?
→ [數學基礎](docs/01-foundations/01-mathematics/) → [監督學習](docs/02-machine-learning/01-supervised-learning/) → [Lab 01](docs/08-labs/01-sentiment-analysis/)

### 想轉 LLM？| Want to do LLMs?
→ [Attention](docs/05-transformers/01-attention-mechanism/) → [BERT & GPT](docs/05-transformers/02-bert-gpt/) → [LLM 微調](docs/05-transformers/03-llm-finetuning/) → [RAG](docs/07-advanced-topics/01-rag/)

### 想做 RAG / Agent？| RAG / Agents?
→ [RAG](docs/07-advanced-topics/01-rag/) → [Lab 05](docs/08-labs/05-rag-qa-system/) → [Agents](docs/07-advanced-topics/02-agents/)

### 想做推薦/語音/RL？| Rec/Speech/RL?
→ [強化學習](docs/09-advanced/01-reinforcement-learning/) / [語音](docs/09-advanced/02-speech-recognition/) / [推薦](docs/09-advanced/03-recommender-systems/)

---

## 📈 更新日誌 | Changelog

### v1.3 (2026-10-18) - 完善版
- ✅ 5 個 Lab 加完整可執行 `lab.py` 腳本
- ✅ Lab 加「快速開始」「練習題」「進階方向」章節
- ✅ 新增 `datasets/loader.py` 統一數據集接口
- ✅ 新增 `scripts/text_preprocessing.py` / `trainer.py` / `metrics.py`
- ✅ 完善根 README 和 DOCS_INDEX

### v1.2 (2026-10-18)
- ✅ 新增 Part 9：RL / Speech / 推薦 / GNN
- ✅ 完善 Lab 數據集

### v1.1 (2026-10)
- ✅ 完成 38 個教學模組
- ✅ 5 個 Lab 基本結構

---

*© 2026 Xavier Chan | Last updated: 2026-10-18*
