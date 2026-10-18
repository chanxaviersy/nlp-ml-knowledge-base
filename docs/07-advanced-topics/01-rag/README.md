# 🔍 RAG：檢索增強生成 | Retrieval-Augmented Generation

> 摘要：RAG 通過結合檢索系統和大語言模型，解決 LLM 的知識陳舊、幻覺 hallucination 和缺乏特定領域知識的問題。本章介紹 RAG 的核心組件、實現方式與優化策略。  
> Abstract: RAG (Retrieval-Augmented Generation) combines retrieval systems with LLMs to address knowledge cutoff, hallucination, and domain-specific knowledge gaps. This chapter covers core components, implementations, and optimization strategies.

---

## 1. 為什麼需要 RAG？| Why RAG?

LLM 的三大局限：

| 問題 | 描述 | RAG 的解決方案 |
|------|------|---------------|
| **知識截止** | GPT-4 的知識只到 2023 年 4 月，之後的事實它不知道 | 實時檢索最新資料 |
| **幻覺 (Hallucination)** | 模型可能生成看似合理但事實上錯誤的內容 | 基於檢索到的文檔回答，有據可查 |
| **領域知識缺失** | 通用模型缺乏企業內部、專業領域的專有知識 | 檢索私有知識庫 |
| **無法引用來源** | LLM 無法標注答案的出處 | 直接引用檢索到的文檔段落 |
| **資源限制** | 無法把所有知識都放進模型參數裡 | 外部知識庫，按需檢索 |

---

## 2. RAG 架構總覽 | Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAG Pipeline                                  │
│                                                                         │
│  User Query                                                            │
│      │                                                                 │
│      ▼                                                                 │
│  ┌─────────────┐                                                        │
│  │  Query      │  1. 查詢重寫 / 擴展 / 壓縮                            │
│  │  Processing │     (HyDE, Query Expansion)                           │
│  └──────┬──────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────┐                                                        │
│  │  Retrieval   │  2. 從向量資料庫檢索 Top-K 相關文檔                    │
│  │  (Vector DB) │     FAISS / ChromaDB / Pinecone / Milvus              │
│  └──────┬──────┘                                                        │
│         │  K relevant documents                                         │
│         ▼                                                                │
│  ┌─────────────┐                                                        │
│  │  Re-ranker   │  3. 重排序：更精確地排序檢索結果                       │
│  │  (Optional)  │     (Cross-Encoder, Cohere)                          │
│  └──────┬──────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────┐                                                        │
│  │  Context    │  4. 上下文壓縮 / 去除冗餘                              │
│  │  Compression│     (LLMChainer, Recompactor)                        │
│  └──────┬──────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────┐                                                        │
│  │    LLM       │  5. 基於檢索到的上下文生成答案                         │
│  │  Generation  │     (GPT-4, Claude, 本地模型)                         │
│  └──────┬──────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│   Final Answer + Citations                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心組件實現 | Core Components

### 3.1 文檔處理與分塊 | Document Processing & Chunking

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os

# 文檔加載
loader = PyPDFLoader("paper.pdf")
documents = loader.load()

# 分塊策略：RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每個塊的字符數
    chunk_overlap=50,      # 塊之間的重疊（保持上下文連貫）
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],  # 按優先級分割
    add_start_index=True, # 添加原始文檔中的起始位置
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")
print(f"Example chunk: {chunks[0].page_content[:200]}...")
```

### 3.2 嵌入模型 | Embedding Models

```python
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np

# 使用 HuggingFace 嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

# 測試嵌入
test_text = "自然語言處理是人工智慧的一個重要分支"
vector = embeddings.embed_query(test_text)
print(f"Embedding dimension: {len(vector)}")
print(f"Vector sample: {vector[:5]}...")

# 常見多語言嵌入模型對比
MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": "384dim, 12層, 多語言, 速度快",
    "intfloat/multilingual-e5-large": "1024dim, 多語言, 高精度",
    "BAAI/bge-m3": "1024dim, 多語言, 支持250+語言",
    "text-embedding-3-large": "OpenAI 1536dim, 強大但需API",
}
```

### 3.3 向量資料庫 | Vector Databases

```python
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma

# 方法1：FAISS（本地，高效）
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)
vectorstore.save_local("faiss_index")

# 相似性搜索
query = "什麼是注意力機制？"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"\n[Result {i+1}] Score: {doc.metadata.get('score', 'N/A')}")
    print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"  Content: {doc.page_content[:200]}...")

# 方法2：ChromaDB（本地，API 友好）
# vectorstore = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     persist_directory="chroma_db"
# )
```

### 3.4 完整 RAG 實現 | Full RAG Pipeline

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 初始化 LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,  # RAG 通常低溫度以確保事實性
)

# 定義 RAG prompt template
rag_prompt = PromptTemplate(
    template="""使用以下檢索到的上下文來回答問題。如果上下文中沒有答案，請說「根據檢索到的資料，我找不到相關信息」，不要編造答案。

上下文：
{context}

問題：{question}

回答（請引用相關的上下文內容）：""",
    input_variables=["context", "question"],
)

# 創建 RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff": 將所有文檔塞進 prompt
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 5},  # 檢索 top-5 文檔
    ),
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=True,  # 返回來源文檔
)

# 執行查詢
query = "Transformer 的注意力機制是如何工作的？"
result = qa_chain.invoke({"query": query})

print("=" * 60)
print("Question:", query)
print("=" * 60)
print("Answer:", result["result"])
print("=" * 60)
print("Sources:")
for i, doc in enumerate(result["source_documents"]):
    print(f"  [{i+1}] {doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', '?')})")
```

---

## 4. 高級 RAG 策略 | Advanced RAG Strategies

### 4.1 查詢擴展 | Query Expansion

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# HyDE (Hypothetical Document Embeddings)
# 讓 LLM 先生成一個假設性答案，再根據假答案檢索
hyde_prompt = PromptTemplate(
    template="""根據以下問題，假設性地生成一段詳細的答案（即使你不確定事實，也要生成一個看起來合理的答案）。

問題：{question}

假設性答案：""",
    input_variables=["question"]
)

hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt)
hypothetical_doc = hyde_chain.run(query)
print("HyDE generated:", hypothetical_doc)

# 用假設性答案去檢索，而不是用原始問題
hyde_results = vectorstore.similarity_search(hypothetical_doc, k=3)
```

### 4.2 查詢重寫 | Query Rewriting

```python
# 查詢重寫：將模糊查詢改寫為精確的檢索查詢
rewrite_prompt = PromptTemplate(
    template="""將以下用戶的自然語言問題改寫為一個適合向量檢索的精確查詢。

原問題：{question}

改寫要求：
1. 提取核心實體和關鍵概念
2. 移除口語化表達
3. 如果涉及多個問題，拆分為單一查詢
4. 可添加相關領域術語

改寫後的查詢：""",
    input_variables=["question"]
)
```

### 4.3 重排序 | Re-ranking

```python
from sentence_transformers import CrossEncoder

# 使用 Cross-Encoder 對檢索結果重排序
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query, retrieved_docs, top_k=3):
    """用 Cross-Encoder 重排序"""
    doc_pairs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = cross_encoder.predict(doc_pairs)
    
    # 按分數排序
    scored_docs = list(zip(scores, retrieved_docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    return scored_docs[:top_k]

# 原始檢索
raw_results = vectorstore.similarity_search(query, k=10)
# 重排序
reranked = rerank_documents(query, raw_results, top_k=3)

for score, doc in reranked:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}...")
```

---

## 5. RAG 評估 | Evaluation

### 5.1 評估指標

| 指標 | 衡量內容 | 工具 |
|------|----------|------|
| **Context Precision** | 檢索到的上下文與問題的相關性 | RAGAS |
| **Context Recall** | 檢索到的上下文是否包含完整答案 | RAGAS |
| **Answer Faithfulness** | 生成答案是否忠實於檢索到的上下文 | RAGAS |
| **Answer Relevance** | 生成答案與問題的相關性 | RAGAS |
| **Hit Rate** | Top-K 檢索中是否包含正確文檔 | 簡單指標 |
| **MRR (Mean Reciprocal Rank)** | 正確文檔在結果中的排名 | 檢索指標 |

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall
)
from datasets import Dataset

# 準備評估數據
eval_data = {
    "user_input": ["注意力機制的原理是什麼？", ...],
    "retrieved_contexts": [[context1, context2, ...], ...],
    "response": ["答案是...", ...],
    "reference": ["正確答案是...", ...],
}

dataset = Dataset.from_dict(eval_data)

# 評估
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(result)
```

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────┐
│  RAG 優化檢查清單                                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  □  數據預處理                                           │
│     ├── 清洗 HTML / Markdown / PDF                       │
│     ├── 選擇合適的分塊策略 (chunk_size, overlap)          │
│     └── 保留元數據（標題、頁碼、來源URL）                 │
│                                                          │
│  □  檢索優化                                             │
│     ├── 選擇合適的嵌入模型（多語言、領域適配）            │
│     ├── 混合搜索（keyword + vector）                     │
│     ├── 重排序（Cross-Encoder）                         │
│     └── HyDE / Query Expansion                           │
│                                                          │
│  □  生成優化                                             │
│     ├── 低溫度 (0.0-0.3) 確保事實性                      │
│     ├── Prompt 加入「只根據上下文回答」                  │
│     └── 上下文壓縮，去除無關信息                        │
│                                                          │
│  □  評估                                                 │
│     ├── RAGAS / TruLens 綜合評估                        │
│     ├── Human Evaluation 抽檢                           │
│     └── A/B 測試不同配置                                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 參考文獻 | References

1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
2. Gao, Y., et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv*.
3. Ram, O., et al. (2023). In-Context Retrieval-Augmented Language Models. *TACL*.
4. Rubin, O., et al. (2022). Learning to Retrieve Prompts for In-Context Learning. *NAACL*.

---

*© 2026 chanxaviersy*
