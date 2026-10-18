#!/usr/bin/env python3
"""
評估指標工具
============

NLP / ML 通用評估指標：分類、NER、生成、RAG 評估。
"""

import numpy as np
from typing import List, Dict, Optional
from collections import Counter


# ============================================================================
# 1. 分類指標
# ============================================================================

def accuracy(y_true: List, y_pred: List) -> float:
    """準確率"""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0.0


def precision_recall_f1(
    y_true: List[int],
    y_pred: List[int],
    pos_label: int = 1,
    average: str = "binary",
) -> Dict[str, float]:
    """
    計算 Precision / Recall / F1
    
    average:
        - "binary": 二分類
        - "micro": 全局統計
        - "macro": 類別平均
        - "weighted": 加權平均
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    return {
        "precision": precision_score(y_true, y_pred, pos_label=pos_label, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label, average=average, zero_division=0),
    }


def top_k_accuracy(y_true: List[int], y_probs: np.ndarray, k: int = 5) -> float:
    """Top-K 準確率"""
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = sum(1 for i, t in enumerate(y_true) if t in top_k_preds[i])
    return correct / len(y_true)


# ============================================================================
# 2. NER 評估（entity-level）
# ============================================================================

def extract_entities(tags: List[str]) -> List[tuple]:
    """從 BIO/BIOES 標籤中提取實體"""
    entities = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            entity_type = tag[2:]
            start = i
            i += 1
            while i < len(tags) and tags[i].startswith(("I-", "E-")) and tags[i][2:] == entity_type:
                i += 1
            entities.append((start, i, entity_type))
        elif tag.startswith("S-"):
            entities.append((i, i + 1, tag[2:]))
            i += 1
        else:
            i += 1
    return entities


def ner_metrics(
    true_tags: List[List[str]],
    pred_tags: List[List[str]],
) -> Dict[str, float]:
    """
    NER 實體級別 F1 評估
    
    使用 seqeval 更嚴格；此函數提供簡單版本
    """
    true_entities = []
    pred_entities = []
    for tt, pt in zip(true_tags, pred_tags):
        true_entities.append(set((s, e, t) for s, e, t in extract_entities(tt)))
        pred_entities.append(set((s, e, t) for s, e, t in extract_entities(pt)))
    
    # 計算 TP/FP/FN
    all_true = set()
    all_pred = set()
    for te, pe in zip(true_entities, pred_entities):
        all_true |= te
        all_pred |= pe
    
    tp = len(all_true & all_pred)
    fp = len(all_pred - all_true)
    fn = len(all_true - all_pred)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}


# ============================================================================
# 3. 文本生成指標
# ============================================================================

def bleu_score(reference: List[str], hypothesis: str, max_n: int = 4) -> float:
    """
    BLEU 評分（簡化版，建議使用 nltk.translate.bleu_score）
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothing = SmoothingFunction().method1
    try:
        weights = [1.0 / max_n] * max_n
        return sentence_bleu([reference], hypothesis.split(), weights=weights, smoothing_function=smoothing)
    except Exception:
        return 0.0


def rouge_l(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    ROUGE-L (Longest Common Subsequence)
    """
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    
    # LCS
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}


# ============================================================================
# 4. RAG 評估
# ============================================================================

def hit_rate(retrieved_docs: List[List[str]], relevant_docs: List[List[str]], k: int = 1) -> float:
    """
    Hit Rate @ K
    
    retrieved_docs: 檢索到的文檔列表（每查詢一個 list）
    relevant_docs: 相關文檔列表
    """
    if not relevant_docs:
        return 0.0
    hits = 0
    for ret, rel in zip(retrieved_docs, relevant_docs):
        top_k = ret[:k]
        if any(r in top_k for r in rel):
            hits += 1
    return hits / len(relevant_docs)


def mrr(retrieved_docs: List[List[str]], relevant_docs: List[List[str]]) -> float:
    """
    Mean Reciprocal Rank
    """
    rr_sum = 0
    for ret, rel in zip(retrieved_docs, relevant_docs):
        rel_set = set(rel)
        for rank, doc in enumerate(ret, 1):
            if doc in rel_set:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(relevant_docs) if relevant_docs else 0


def ndcg_at_k(retrieved_docs: List[List[str]], relevant_docs: List[List[str]], k: int = 10) -> float:
    """
    NDCG @ K (二級相關性: 相關/不相關)
    """
    ndcg_sum = 0
    for ret, rel in zip(retrieved_docs, relevant_docs):
        rel_set = set(rel)
        # DCG
        dcg = sum(1.0 / np.log2(i + 2) for i, doc in enumerate(ret[:k]) if doc in rel_set)
        # IDCG (理想情況下所有相關都在 top)
        ideal_count = min(len(rel), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_count))
        if idcg > 0:
            ndcg_sum += dcg / idcg
    return ndcg_sum / len(retrieved_docs) if retrieved_docs else 0


# ============================================================================
# 5. 困惑度 / PPL
# ============================================================================

def perplexity(loss: float) -> float:
    """從交叉熵損失計算困惑度"""
    import math
    return math.exp(loss)


# ============================================================================
# 6. CLI Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📊 評估指標工具 Demo")
    print("=" * 60)
    
    # 1. 分類
    y_true = [1, 0, 1, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 1]
    print(f"\n1️⃣  分類: {precision_recall_f1(y_true, y_pred)}")
    
    # 2. NER
    true_tags = [["B-PER", "I-PER", "O", "B-LOC"]]
    pred_tags = [["B-PER", "I-PER", "O", "B-LOC"]]
    print(f"2️⃣  NER: {ner_metrics(true_tags, pred_tags)}")
    
    # 3. ROUGE
    ref = "the cat sat on the mat"
    hyp = "the cat is on the mat"
    print(f"3️⃣  ROUGE-L: {rouge_l(ref, hyp)}")
    
    # 4. PPL
    print(f"4️⃣  PPL (loss=2.5): {perplexity(2.5):.2f}")
    
    print("\n✅ 評估指標模組已加載")
