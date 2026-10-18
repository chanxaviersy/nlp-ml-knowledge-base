# 🔍 Lab 05: RAG 問答系統 | RAG-based QA System

> 摘要：本 Lab 構建一個完整的 RAG（檢索增強生成）問答系統，包括文檔加載、向量化、檢索、生成和評估。  
> Abstract: This lab builds a complete RAG QA system including document loading, vectorization, retrieval, generation, and evaluation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chanxaviersy/nlp-ml-knowledge-base/blob/main/docs/08-labs/05-rag-qa-system/lab.ipynb)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)
![No GPU](https://img.shields.io/badge/GPU-not_required-green)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-1c3c3c)

---

## 🎯 目標 | Objectives

- 構建 RAG 系統全流程
- 使用 LangChain 框架
- 整合 FAISS 向量資料庫
- 評估 RAG 性能

---

## 📋 環境準備 | Environment

```bash
pip install langchain langchain-community langchain-openai langchain-huggingface
pip install faiss-cpu sentence-transformers pypdf openai tiktoken
pip install chromadb rank-bm25  # 可選
```

---

## 📊 步驟 1: 準備文檔 | Prepare Documents

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 方法 1：從 PDF 加載
def load_pdf_docs(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs

# 方法 2：從文本文件
def load_text_docs(file_path):
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()

# 方法 3：從目錄
def load_directory_docs(directory):
    loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()

# 方法 4：直接從字符串
from langchain.schema import Document
docs = [
    Document(
        page_content="台積電（TSMC）是全球最大的半導體代工廠，總部位於台灣新竹科學園區，由張忠謀於1987年創立。",
        metadata={"source": "台積電簡介", "page": 1}
    ),
    Document(
        page_content="NVIDIA 是一家以 GPU 和 AI 晶片聞名的美國公司，由黃仁勳於 1993 年創立，總部位於加州聖塔克拉拉。",
        metadata={"source": "NVIDIA 簡介", "page": 1}
    ),
    Document(
        page_content="輝達（NVIDIA）在 AI 晶片市場佔據主導地位，其 H100 GPU 是訓練大型語言模型的首選硬體。",
        metadata={"source": "AI 硬體報告", "page": 5}
    ),
    # ... 更多文檔
]

print(f"加載了 {len(docs)} 個文檔")
```

---

## 🔧 步驟 2: 文檔分塊 | Document Chunking

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每塊大小
    chunk_overlap=50,      # 重疊大小
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(docs)
print(f"分塊數量: {len(chunks)}")
print(f"\n示例塊（前 200 字）:\n{chunks[0].page_content[:200]}")
```

---

## 🔧 步驟 3: 構建向量資料庫 | Build Vector Store

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 選擇嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # 或使用 BGE: "BAAI/bge-m3"
    model_kwargs={"device": "cpu"},  # 或 "cuda"
    encode_kwargs={"normalize_embeddings": True},
)

# 2. 構建 FAISS 向量資料庫
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)

# 3. 保存
vectorstore.save_local("faiss_index")

print(f"向量資料庫大小: {vectorstore.index.ntotal}")
```

---

## 🔧 步驟 4: 相似性搜索 | Similarity Search

```python
# 基本搜索
query = "台積電的創辦人是誰？"
results = vectorstore.similarity_search(query, k=3)

print(f"查詢: {query}\n")
for i, doc in enumerate(results):
    print(f"[結果 {i+1}]")
    print(f"內容: {doc.page_content}")
    print(f"來源: {doc.metadata}\n")

# 帶分數的搜索
results_with_score = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in results_with_score:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}...")
```

---

## 🔧 步驟 5: 構建 RAG Chain

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# 1. LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
)

# 2. Prompt 模板
template = """你是一個準確的問答助手。請基於以下檢索到的上下文回答問題。

要求：
1. 只使用上下文中的信息
2. 如果上下文中沒有答案，明確說「根據提供的資料，我無法回答這個問題」
3. 引用相關來源
4. 用繁體中文回答

上下文：
{context}

問題：{question}

回答："""

prompt = ChatPromptTemplate.from_template(template)

# 3. 構建 RAG Chain
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. 測試
question = "台積電的創辦人是誰？"
answer = rag_chain.invoke(question)
print(f"問題: {question}")
print(f"回答: {answer}")
```

---

## 🔧 步驟 6: 帶來源的 RAG

```python
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

def format_docs(docs):
    """格式化檢索到的文檔"""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知")
        content = doc.page_content
        formatted.append(f"[來源 {i+1}：{source}]\n{content}")
    return "\n\n".join(formatted)

# 構建帶源信息的 RAG
rag_with_sources = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 使用
question = "請介紹台積電和 NVIDIA。"
result = rag_with_sources.invoke(question)
print(result)
```

---

## 🔧 步驟 7: 進階 - 混合搜索

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# 1. BM25 檢索器（基於關鍵詞）
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# 2. 向量檢索器
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. 集成兩個檢索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7],  # BM25: 30%, Vector: 70%
)

# 測試混合檢索
question = "台積電是誰創辦的？"
results = ensemble_retriever.invoke(question)
for i, doc in enumerate(results):
    print(f"[{i+1}] {doc.page_content[:150]}")
```

---

## 🔧 步驟 8: 重排序 (Reranking)

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 1. 加載 Cross-Encoder
model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=model, top_n=3)

# 2. 壓縮檢索器：先檢索 10 個，重排序取 top 3
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)

# 使用
question = "台積電創辦人的背景"
results = compression_retriever.invoke(question)
for i, doc in enumerate(results):
    print(f"[{i+1}] {doc.page_content[:200]}")
```

---

## 🔧 步驟 9: 對話式 RAG

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 1. 對話記憶
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)

# 2. 對話式 RAG
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
    output_key="answer",
)

# 3. 多輪對話
print("=== 多輪對話示例 ===\n")

questions = [
    "台積電是做什麼的？",
    "它的創辦人是誰？",
    "那個人還創辦過其他公司嗎？",  # 依賴上文
]

for q in questions:
    result = qa_chain.invoke({"question": q})
    print(f"問: {q}")
    print(f"答: {result['answer']}")
    print(f"來源: {[d.metadata.get('source') for d in result['source_documents']]}")
    print()
```

---

## 🔧 步驟 10: 評估 RAG 性能

```python
# 準備評估數據集
eval_data = [
    {
        "question": "台積電的創辦人是誰？",
        "ground_truth": "張忠謀",
        "contexts": ["台積電由張忠謀於1987年創立"],
    },
    {
        "question": "NVIDIA 的總部在哪裡？",
        "ground_truth": "加州聖塔克拉拉",
        "contexts": ["NVIDIA 總部位於加州聖塔克拉拉"],
    },
]

# 1. 評估檢索質量
def evaluate_retrieval(eval_data, retriever, k=3):
    """評估 Top-K 命中率"""
    hits = 0
    for item in eval_data:
        retrieved = retriever.invoke(item["question"])[:k]
        retrieved_text = " ".join([d.page_content for d in retrieved])
        if any(g in retrieved_text for g in item["contexts"]):
            hits += 1
    return hits / len(eval_data)

hit_rate = evaluate_retrieval(eval_data, vectorstore.as_retriever(), k=3)
print(f"Hit Rate @3: {hit_rate:.2%}")

# 2. 使用 RAGAS 評估
# pip install ragas
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall
)

# 構建評估數據集
ragas_data = {
    "question": [item["question"] for item in eval_data],
    "contexts": [[item["contexts"]] for item in eval_data],
    "answer": ["張忠謀", "加州聖塔克拉拉"],
    "ground_truth": [item["ground_truth"] for item in eval_data],
}
dataset = Dataset.from_dict(ragas_data)

# 評估
# result = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
# )
# print(result)
```

---

## 🔧 步驟 11: 構建 Web UI（Gradio）

```python
import gradio as gr

def qa_function(question, history):
    """問答函數"""
    result = rag_chain.invoke(question)
    return result

# 構建 Gradio 介面
demo = gr.ChatInterface(
    fn=qa_function,
    title="📚 RAG 問答助手",
    description="基於文檔的問答系統",
    examples=[
        "台積電的創辦人是誰？",
        "NVIDIA 主要做什麼？",
        "AI 晶片市場的領導者是誰？",
    ],
    theme="soft",
)

# 啟動
demo.launch(server_name="0.0.0.0", server_port=7860)
```

---

## 🎓 思考題 | Exercises

1. **更多文檔類型**：加載 Word、Excel、網頁
2. **多語言支持**：加載英文文檔並混合檢索
3. **HyDE**：使用假設性文檔嵌入提升檢索
4. **GraphRAG**：構建知識圖譜增強的 RAG
5. **評估體系**：建立完整的 RAG 評估 pipeline
6. **生產化**：用 FastAPI 部署，Redis 快取，監控

---

## 📊 預期結果 | Expected Results

| 組件 | 評估指標 | 結果 |
|------|----------|------|
| 檢索 | Hit Rate @3 | 80-90% |
| 生成 | Faithfulness | 85-95% |
| | Answer Relevancy | 80-90% |
| | Context Precision | 75-85% |
| 端到端 | 準確率 | 70-85% |

**調優方向**：
1. 更好的分塊策略（按語義、按章節）
2. 更大的嵌入模型（BGE-large）
3. 重排序（Cross-Encoder）
4. HyDE / Query Expansion
5. 更好的 LLM（GPT-4 vs Claude）

---

## 📚 參考 | References

- [LangChain RAG 教程](https://python.langchain.com/docs/use_cases/question_answering/)
- [FAISS 文檔](https://faiss.ai/)
- [RAGAS 評估框架](https://docs.ragas.io/)
- [BGE 嵌入模型](https://huggingface.co/BAAI/bge-m3)

---

## ⚡ 快速開始 | Quick Start

```bash
cd docs/08-labs/05-rag-qa-system

# 1. 安裝
pip install langchain langchain-community faiss-cpu sentence-transformers

# 2. 執行（使用本地嵌入，無需 API key）
python lab.py

# 3. 帶 LLM 的版本（需 OpenAI API key）
export OPENAI_API_KEY="sk-..."
python lab.py --llm openai
```

**互動測試**：
```
問題: 台積電的創辦人是誰？
💡 簡短回答: 張忠謀於 1987 年在台灣新竹創立台積電...
```

---

## 🧪 練習題 | Exercises

| # | 難度 | 題目 |
|---|------|------|
| 1 | ⭐ | 替換嵌入模型為 `BAAI/bge-m3`（效果更好）|
| 2 | ⭐ | 改用 **Hybrid Search**（BM25 + 嵌入）|
| 3 | ⭐⭐ | 加入 **Cross-Encoder Re-ranking** |
| 4 | ⭐⭐ | 嘗試 **HyDE**（Hypothetical Document Embeddings）|
| 5 | ⭐⭐ | 構建 **Multi-hop QA**（多文檔推理）|
| 6 | ⭐⭐⭐ | 加入 **Query Rewriting** / **Step-back Prompting** |
| 7 | ⭐⭐⭐ | 部署為 FastAPI + Streamlit 服務 |
| 8 | ⭐⭐⭐⭐ | 接入 **GraphRAG**（知識圖譜增強 RAG）|

---

## 🏗️ 完整 RAG 架構

```
User Query
    ↓
Query Preprocessing (rewrite, expand, decompose)
    ↓
Hybrid Retrieval (BM25 + Embeddings)
    ↓
Re-ranking (Cross-Encoder)
    ↓
Context Compression
    ↓
LLM Generation
    ↓
Post-processing (citation, fact-check)
```

---

## 📊 評估指標 | Evaluation Metrics

| 指標 | 含義 | 工具 |
|------|------|------|
| **Context Precision** | 檢索文檔相關性 | RAGAS |
| **Context Recall** | 必要信息是否都檢索到 | RAGAS |
| **Faithfulness** | 答案是否忠於上下文 | RAGAS |
| **Answer Relevance** | 答案對問題的相關性 | RAGAS |
| **Hit Rate** | 正確答案是否在 top-k | 自定 |

---

## 🔧 常見優化 | Common Optimizations

| 問題 | 解決方案 |
|------|----------|
| 答案不準確 | 換更好嵌入 / 加入重排序 |
| 召回不足 | 加大 top-k / Hybrid Search |
| 上下文超長 | Context Compression |
| 速度慢 | 緩存 / 預計算 / GPU 加速 |
| 幻覺 | 嚴格 prompt / Self-Check |

---

*© 2026 chanxaviersy*
