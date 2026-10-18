#!/usr/bin/env python3
"""
文本預處理工具
===============

統一的中文/英文文本預處理管道。
"""

import re
import jieba
from typing import List, Optional, Callable


# ============================================================================
# 1. 基礎清理
# ============================================================================

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
HTML_PATTERN = re.compile(r"<[^>]+>")


def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_html: bool = True,
    remove_extra_spaces: bool = True,
) -> str:
    """基礎文本清理"""
    if not isinstance(text, str):
        return ""
    if remove_html:
        text = HTML_PATTERN.sub(" ", text)
    if remove_urls:
        text = URL_PATTERN.sub(" ", text)
    if remove_emails:
        text = EMAIL_PATTERN.sub(" ", text)
    if remove_extra_spaces:
        text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================================
# 2. 中文分詞
# ============================================================================

# 加載自定義詞典（可選）
def load_user_dict(dict_path: str):
    """加載用戶自定義詞典"""
    jieba.load_userdict(dict_path)


def tokenize_chinese(
    text: str,
    cut_all: bool = False,
    use_hmm: bool = True,
    stopwords: Optional[List[str]] = None,
) -> List[str]:
    """
    中文分詞（支持停用詞過濾）
    
    Args:
        text: 輸入文本
        cut_all: 是否全模式分詞
        use_hmm: 是否使用 HMM
        stopwords: 停用詞列表
    
    Returns:
        詞列表
    """
    text = clean_text(text)
    words = jieba.lcut(text, cut_all=cut_all, HMM=use_hmm)
    if stopwords:
        words = [w for w in words if w.strip() and w not in stopwords]
    return words


# ============================================================================
# 3. 停用詞表
# ============================================================================

CHINESE_STOPWORDS = {
    "的", "了", "和", "是", "在", "也", "都", "及", "與", "或",
    "一個", "沒有", "我們", "你們", "他們", "它們", "這個", "那個",
    "因為", "所以", "但是", "然而", "不過", "雖然", "不僅", "而且",
    "如果", "那麼", "如此", "這樣", "那樣", "以及", "以至於",
    "啊", "哦", "呢", "嘛", "吧", "呀", "哎", "嘿", "哼", "唉",
}


def load_stopwords(filepath: Optional[str] = None) -> set:
    """
    加載停用詞表
    
    默認包含常用中文停用詞；可從文件加載更多。
    文件格式：每行一個停用詞
    """
    stopwords = set(CHINESE_STOPWORDS)
    if filepath and Path(filepath).exists():
        from pathlib import Path
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    stopwords.add(w)
    return stopwords


# ============================================================================
# 4. 完整預處理管道
# ============================================================================

def preprocess_pipeline(
    text: str,
    steps: List[str] = None,
    stopwords: Optional[set] = None,
) -> str:
    """
    可配置預處理管道
    
    steps 可選:
        - "clean": 基礎清理
        - "tokenize": 分詞
        - "lower": 小寫化
        - "filter_stopwords": 過濾停用詞
        - "filter_short": 過濾短詞 (len<2)
        - "filter_punct": 過濾標點
    """
    if steps is None:
        steps = ["clean", "tokenize", "filter_stopwords", "filter_short"]
    
    if "clean" in steps:
        text = clean_text(text)
    if "lower" in steps:
        text = text.lower()
    
    if "tokenize" in steps:
        words = tokenize_chinese(text, stopwords=list(stopwords) if stopwords else None)
    else:
        words = text.split()
    
    if "filter_short" in steps:
        words = [w for w in words if len(w) >= 2]
    if "filter_punct" in steps:
        from string import punctuation
        words = [w for w in words if not all(c in punctuation for c in w)]
    
    return " ".join(words)


# ============================================================================
# 5. 統計工具
# ============================================================================

def text_statistics(texts: List[str]) -> dict:
    """計算文本統計信息"""
    import numpy as np
    if not texts:
        return {}
    
    lengths_chars = [len(t) for t in texts]
    lengths_words = [len(t.split()) for t in texts]
    
    return {
        "n_samples": len(texts),
        "char_length": {
            "mean": float(np.mean(lengths_chars)),
            "std": float(np.std(lengths_chars)),
            "min": int(np.min(lengths_chars)),
            "max": int(np.max(lengths_chars)),
            "median": float(np.median(lengths_chars)),
        },
        "word_length": {
            "mean": float(np.mean(lengths_words)),
            "std": float(np.std(lengths_words)),
            "min": int(np.min(lengths_words)),
            "max": int(np.max(lengths_words)),
        },
    }


if __name__ == "__main__":
    # Demo
    sample = "我今天去了一家餐廳，菜真的很好吃，特別是那道紅燒肉！！！ http://example.com"
    print("原始:", sample)
    print("清理:", clean_text(sample))
    print("分詞:", tokenize_chinese(sample))
    print("管道:", preprocess_pipeline(sample))
