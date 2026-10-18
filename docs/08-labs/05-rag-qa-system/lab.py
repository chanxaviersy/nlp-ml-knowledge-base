#!/usr/bin/env python3
"""
Lab 05: RAG 問答系統 - LangChain + FAISS
========================================
構建一個簡單的 RAG 問答系統：載入文檔 → 切分 → 嵌入 → 檢索 → 生成。

Author: chanxaviersy
Date: 2026-10-18
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.utils import set_seed, timer

set_seed(42)

DATA_DIR = Path(__file__).resolve().parent / "data"
DOCS_DIR = DATA_DIR / "documents"


# ============================================================================
# 1. 文檔載入
# ============================================================================

def load_documents(docs_dir: Path = DOCS_DIR) -> list:
    """從目錄載入所有 .txt 文檔"""
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    
    print(f"📂 載入文檔: {docs_dir}")
    loader = DirectoryLoader(
        str(docs_dir), glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"✅ 共載入 {len(docs)} 個文檔")
    return docs


# ============================================================================
# 2. 文檔切分
# ============================================================================

def split_documents(docs: list, chunk_size: int = 300, chunk_overlap: int = 50) -> list:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  切分為 {len(chunks)} 個塊")
    return chunks


# ============================================================================
# 3. 嵌入與向量庫
# ============================================================================

def build_vectorstore(chunks: list, save_path: Path = None):
    """構建 FAISS 向量庫（使用本地嵌入，無需 API key）"""
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    print("🔧 載入嵌入模型 (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    print("📦 構建向量庫...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    if save_path:
        vectorstore.save_local(str(save_path))
        print(f"💾 向量庫已保存: {save_path}")
    return vectorstore


def load_vectorstore(path: Path):
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)


# ============================================================================
# 4. 問答主鏈
# ============================================================================

class SimpleRAG:
    """簡化版 RAG 問答系統 (無需 LLM API)"""
    
    def __init__(self, vectorstore, top_k: int = 3):
        self.vectorstore = vectorstore
        self.top_k = top_k
    
    def retrieve(self, query: str) -> list:
        """檢索 top-k 相關文檔"""
        return self.vectorstore.similarity_search(query, k=self.top_k)
    
    def answer(self, query: str) -> dict:
        """基於檢索結果生成答案（提取式）"""
        docs = self.retrieve(query)
        # 簡單拼接 top-k 摘要
        context = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
        return {
            "query": query,
            "context": context,
            "sources": [d.metadata.get("source", "") for d in docs],
            "answer": docs[0].page_content[:200] if docs else "(無相關文檔)",
        }
    
    def interactive(self):
        """互動問答"""
        print("\n" + "="*60)
        print("🤖 簡易 RAG 問答系統 (輸入 q 退出)")
        print("="*60)
        while True:
            try:
                q = input("\n問題: ").strip()
            except EOFError:
                break
            if not q or q.lower() == "q":
                break
            result = self.answer(q)
            print(f"\n💡 簡短回答: {result['answer']}...")
            print(f"\n📚 參考來源:")
            for i, d in enumerate(self.retrieve(q), 1):
                src = Path(d.metadata.get("source", "")).name
                print(f"  [{i}] {src}: {d.page_content[:100]}...")
        print("\n👋 結束！")


# ============================================================================
# 5. 高級版本：使用 OpenAI / 本地 LLM
# ============================================================================

def build_rag_with_llm(vectorstore, llm_type: str = "openai"):
    """構建完整 RAG（需 LLM）"""
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    
    template = """使用以下上下文回答問題。如果不知道答案就說不知道，不要編造。

上下文:
{context}

問題: {question}

回答:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    if llm_type == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    elif llm_type == "huggingface":
        from langchain_community.llms import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=256)
        llm = HuggingFacePipeline(pipeline=pipe)
    else:
        raise ValueError(f"Unknown llm_type: {llm_type}")
    
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa


# ============================================================================
# 6. 評估 (RAGAS 風格)
# ============================================================================

def evaluate_retrieval(vectorstore, test_queries: list[dict]):
    """評估檢索質量"""
    print("\n" + "="*60)
    print("📊 檢索評估")
    print("="*60)
    correct = 0
    for item in test_queries:
        q, expected_keywords = item["query"], item["expected"]
        docs = vectorstore.similarity_search(q, k=3)
        retrieved = " ".join(d.page_content for d in docs)
        hit = any(kw in retrieved for kw in expected_keywords)
        correct += int(hit)
        status = "✅" if hit else "❌"
        print(f"{status} Q: {q}")
        print(f"   期望關鍵詞: {expected_keywords}")
    print(f"\n命中率: {correct}/{len(test_queries)} = {correct/len(test_queries):.2%}")


# ============================================================================
# 7. 主流程
# ============================================================================

@timer
def main():
    # 1. 載入文檔
    docs = load_documents()
    
    # 2. 切分
    chunks = split_documents(docs)
    
    # 3. 構建向量庫
    index_path = DATA_DIR / "faiss_index"
    if index_path.exists() and any(index_path.iterdir()):
        print("📂 載入已存在的向量庫...")
        vectorstore = load_vectorstore(index_path)
    else:
        vectorstore = build_vectorstore(chunks, save_path=index_path)
    
    # 4. 評估檢索
    test_queries = [
        {"query": "台積電的創辦人是誰？", "expected": ["張忠謀", "1987"]},
        {"query": "NVIDIA 是誰創辦的？", "expected": ["黃仁勳", "1993"]},
        {"query": "什麼是區塊鏈？", "expected": ["去中心化", "帳本"]},
        {"query": "量子計算的單位是什麼？", "expected": ["量子位元", "qubit"]},
        {"query": "Elon Musk 創辦了哪些公司？", "expected": ["Tesla", "SpaceX"]},
    ]
    evaluate_retrieval(vectorstore, test_queries)
    
    # 5. 互動
    rag = SimpleRAG(vectorstore, top_k=3)
    rag.interactive()
    print("\n✅ Lab 05 完成！")


if __name__ == "__main__":
    main()
