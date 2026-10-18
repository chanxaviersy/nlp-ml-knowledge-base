#!/usr/bin/env python3
"""
Lab 01: 情感分析 - 傳統機器學習路線
=====================================
使用 TF-IDF + Logistic Regression / SVM 對中文評論進行情感二分類。

Author: chanxaviersy
Date: 2026-10-18
"""

import os
import sys
import re
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 設定中文字型
plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline

# 加入根目錄以便 import
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.utils import (
    set_seed, get_device, set_seed, plot_confusion_matrix,
    plot_metrics_comparison, balance_dataset, evaluate_classification,
    print_evaluation_report,
)

set_seed(42)
RANDOM_STATE = 42


# ============================================================================
# 1. 載入數據
# ============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    優先從 HuggingFace 下載完整 ChnSentiCorp 數據集；若離線，使用本地示例。
    """
    data_dir = Path(__file__).resolve().parent / "data"
    
    try:
        print("📥 嘗試從 HuggingFace 下載 ChnSentiCorp...")
        from datasets import load_dataset
        ds_train = load_dataset("lansinuote/ChnSentiCorp", split="train")
        ds_test = load_dataset("lansinuote/ChnSentiCorp", split="test")
        train_df = pd.DataFrame({"text": ds_train["text"], "label": ds_train["label"]})
        test_df = pd.DataFrame({"text": ds_test["text"], "label": ds_test["label"]})
        print(f"✅ 訓練集: {len(train_df)}, 測試集: {len(test_df)}")
    except Exception as e:
        print(f"⚠️  在線下載失敗 ({e})，使用本地示例...")
        train_df = pd.read_csv(data_dir / "sample_train.csv")
        test_df = pd.read_csv(data_dir / "sample_test.csv")
    
    # 清理空值
    train_df = train_df.dropna(subset=["text"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["text"]).reset_index(drop=True)
    return train_df, test_df


# ============================================================================
# 2. 文本預處理
# ============================================================================

def preprocess_chinese(text: str) -> str:
    """中文預處理：清理 + jieba 分詞"""
    # 清理
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 分詞
    words = jieba.lcut(text)
    return " ".join(w for w in words if len(w) > 1)


def preprocess_corpus(texts: list[str]) -> list[str]:
    """批量預處理"""
    return [preprocess_chinese(t) for t in texts]


# ============================================================================
# 3. 模型定義
# ============================================================================

def build_pipeline(model_name: str = "logreg") -> Pipeline:
    """構建 sklearn Pipeline"""
    tfidf = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    
    if model_name == "logreg":
        clf = LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
        )
    elif model_name == "svm":
        clf = LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


# ============================================================================
# 4. 訓練與評估
# ============================================================================

def train_and_eval(
    model_name: str,
    X_train: list[str], y_train: list[int],
    X_test: list[str], y_test: list[int],
) -> dict:
    """訓練單個模型並評估"""
    print(f"\n{'='*60}\n🤖 訓練模型: {model_name.upper()}\n{'='*60}")
    
    pipe = build_pipeline(model_name)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    metrics = evaluate_classification(y_test, y_pred, average="binary")
    print_evaluation_report(y_test, y_pred, target_names=["負面", "正面"])
    print(f"📊 {model_name}: {metrics}")
    
    return {"pipeline": pipe, "y_pred": y_pred, "metrics": metrics}


def main():
    # 1. 載入
    train_df, test_df = load_data()
    print(f"\n📊 訓練集: {len(train_df)} 條")
    print(f"📊 測試集: {len(test_df)} 條")
    print(f"📊 訓練標籤分佈:\n{train_df['label'].value_counts()}")
    
    # 2. 預處理
    print("\n🔧 預處理文本...")
    X_train = preprocess_corpus(train_df["text"].tolist())
    y_train = train_df["label"].values
    X_test = preprocess_corpus(test_df["text"].tolist())
    y_test = test_df["label"].values
    
    # 3. 訓練多個模型
    results = {}
    for model_name in ["logreg", "svm", "rf"]:
        result = train_and_eval(model_name, X_train, y_train, X_test, y_test)
        results[model_name] = result
    
    # 4. 對比
    summary = {name: r["metrics"] for name, r in results.items()}
    print(f"\n📈 模型對比:")
    print(pd.DataFrame(summary).T)
    
    # 5. 視覺化
    plot_metrics_comparison(summary, metrics=["accuracy", "precision", "recall", "f1"])
    best_name = max(summary, key=lambda k: summary[k]["f1"])
    plot_confusion_matrix(y_test, results[best_name]["y_pred"], labels=["負面", "正面"])
    
    # 6. 互動測試
    print("\n" + "="*60)
    print("🧪 互動測試 (輸入 q 退出)")
    print("="*60)
    best_pipe = results[best_name]["pipeline"]
    while True:
        try:
            text = input("\n請輸入評論: ").strip()
        except EOFError:
            break
        if not text or text.lower() == "q":
            break
        processed = preprocess_chinese(text)
        pred = best_pipe.predict([processed])[0]
        proba = best_pipe.predict_proba([processed])[0] if hasattr(best_pipe, "predict_proba") else [None, None]
        sentiment = "😊 正面" if pred == 1 else "😞 負面"
        confidence = proba[pred] if proba[pred] is not None else "N/A"
        print(f"結果: {sentiment}  (label={pred}, 置信度={confidence:.4f})")
    
    print("\n✅ Lab 01 完成！")


if __name__ == "__main__":
    main()
