#!/usr/bin/env python3
"""
NLP & ML 工具集合

通用輔助函數，涵蓋數據處理、模型評估、可視化等。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# ============================================================================
# 文本處理
# ============================================================================

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截斷文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def batch_texts(texts: List[str], batch_size: int) -> List[List[str]]:
    """將文本列表分批"""
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]


def word_count(text: str) -> int:
    """計算詞數"""
    return len(text.split())


# ============================================================================
# 模型評估
# ============================================================================

def evaluate_classification(
    y_true: List,
    y_pred: List,
    labels: Optional[List[str]] = None,
    average: str = "weighted"
) -> Dict[str, float]:
    """
    計算分類指標
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        labels: 標籤名稱列表
        average: 多分類平均方法
    
    Returns:
        包含各項指標的字典
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def print_evaluation_report(
    y_true: List,
    y_pred: List,
    target_names: Optional[List[str]] = None
) -> None:
    """打印詳細評估報告"""
    print("=" * 60)
    print("分類評估報告")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))


def plot_confusion_matrix(
    y_true: List,
    y_pred: List,
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues"
) -> None:
    """
    繪製混淆矩陣
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        labels: 標籤名稱
        normalize: 是否歸一化
        figsize: 圖形大小
        cmap: 顏色映射
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap=cmap,
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    plt.xlabel('預測標籤', fontsize=12)
    plt.ylabel('真實標籤', fontsize=12)
    plt.title('混淆矩陣' + (' (歸一化)' if normalize else ''), fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 可視化
# ============================================================================

def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (14, 5)
) -> None:
    """
    繪製訓練歷史
    
    Args:
        history: 包含 train_loss, val_loss, train_acc, val_acc 等的字典
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 損失曲線
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curve', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 準確率曲線
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Acc', marker='o', markersize=3)
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Validation Acc', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy Curve', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_attention_weights(
    attention: np.ndarray,
    tokens: List[str],
    head: int = 0,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    可視化注意力權重
    
    Args:
        attention: 形狀 (n_heads, seq_len, seq_len) 或 (seq_len, seq_len)
        tokens: token 列表
        head: 要繪製的頭（多頭注意力時）
    """
    if attention.ndim == 3:
        attn = attention[head]
    else:
        attn = attention
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        attn, xticklabels=tokens, yticklabels=tokens,
        cmap='viridis', annot=False, fmt='.2f',
        cbar_kws={'label': 'Attention Weight'}
    )
    plt.xlabel('Key (Attended To)', fontsize=12)
    plt.ylabel('Query (Attending From)', fontsize=12)
    plt.title(f'Attention Weights (Head {head})', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    比較多個模型的性能
    
    Args:
        results: {model_name: {metric_name: value}}
    """
    import numpy as np
    
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.8 / len(metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in models]
        ax.bar(x + i * width, values, width, label=metric)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 數據處理
# ============================================================================

def split_train_val_test(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    切分訓練/驗證/測試集
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    n = len(data)
    indices = np.arange(n)
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(indices)
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return (
        data[indices[:train_end]],
        data[indices[train_end:val_end]],
        data[indices[val_end:]],
    )


def balance_dataset(
    df: pd.DataFrame,
    label_column: str,
    method: str = "undersample",
    random_state: int = 42
) -> pd.DataFrame:
    """
    平衡數據集
    
    Args:
        df: 數據框
        label_column: 標籤列名
        method: undersample, oversample, smote
    """
    np.random.seed(random_state)
    class_counts = df[label_column].value_counts()
    if method == "undersample":
        min_count = class_counts.min()
        balanced = df.groupby(label_column).apply(
            lambda x: x.sample(min_count, random_state=random_state)
        ).reset_index(drop=True)
    elif method == "oversample":
        max_count = class_counts.max()
        balanced = df.groupby(label_column).apply(
            lambda x: x.sample(max_count, replace=True, random_state=random_state)
        ).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return balanced


# ============================================================================
# 通用工具
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """設置隨機種子以保證可重現性"""
    import random
    import os
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def count_parameters(model) -> int:
    """統計模型參數量"""
    return sum(p.numel() for p in model.parameters())


def get_device():
    """獲取可用設備"""
    try:
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except ImportError:
        return 'cpu'


def timer(func):
    """計時裝飾器"""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"⏱️  {func.__name__} 耗時: {elapsed:.2f}s")
        return result
    return wrapper


if __name__ == "__main__":
    # 簡單測試
    set_seed(42)
    print("✅ NLP & ML 工具模組已加載")
    print(f"📱 設備: {get_device()}")
