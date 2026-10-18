#!/usr/bin/env python3
"""
Lab 02: CNN 文本分類 - PyTorch 從零實現
========================================
使用 TextCNN 對中文文本進行動態卷積分類。

Author: chanxaviersy
Date: 2026-10-18
"""

import os
import re
import sys
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.utils import (
    set_seed, get_device, plot_training_history,
    plot_confusion_matrix, count_parameters, timer,
)

set_seed(42)
DEVICE = get_device()
print(f"📱 設備: {DEVICE}")


# ============================================================================
# 1. 數據準備
# ============================================================================

def preprocess_chinese(text: str) -> list[str]:
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [w for w in jieba.lcut(text) if len(w) > 1]


def load_data():
    data_dir = Path(__file__).resolve().parent / "data"
    try:
        print("📥 下載 ChnSentiCorp...")
        from datasets import load_dataset
        ds = load_dataset("lansinuote/ChnSentiCorp", split="train[:5000]")
        df = pd.DataFrame({"text": ds["text"], "label": ds["label"]})
    except Exception as e:
        print(f"⚠️  在線下載失敗 ({e})，使用本地示例")
        df = pd.read_csv(data_dir / "sample.csv")
    
    df = df.dropna().reset_index(drop=True)
    return df


def build_vocab(texts: list[list[str]], min_freq: int = 2, max_size: int = 30000) -> dict:
    counter = Counter()
    for tokens in texts:
        counter.update(tokens)
    
    # 預留：<pad>=0, <unk>=1
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(text: list[str], vocab: dict, max_len: int = 128) -> list[int]:
    ids = [vocab.get(w, vocab["<unk>"]) for w in text[:max_len]]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


# ============================================================================
# 2. Dataset
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = [encode(preprocess_chinese(t), vocab, max_len) for t in texts]
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.texts[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================================
# 3. TextCNN 模型
# ============================================================================

class TextCNN(nn.Module):
    """
    TextCNN (Kim, 2014) 結構:
    Embedding → 多尺度卷積(n-gram) → MaxPool → Concat → FC
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        n_classes: int = 2,
        kernel_sizes: list[int] = (2, 3, 4, 5),
        n_filters: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)
    
    def forward(self, input_ids):
        # input_ids: (B, L)
        x = self.embedding(input_ids).transpose(1, 2)  # (B, E, L)
        # 多尺度卷積 + ReLU + MaxPool
        outs = []
        for conv in self.convs:
            c = F.relu(conv(x))  # (B, F, L)
            c = F.adaptive_max_pool1d(c, 1).squeeze(-1)  # (B, F)
            outs.append(c)
        # 拼接
        h = torch.cat(outs, dim=1)  # (B, F*K)
        h = self.dropout(h)
        logits = self.fc(h)  # (B, C)
        return logits


# ============================================================================
# 4. 訓練
# ============================================================================

@timer
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        lab = batch["label"].to(device)
        
        optimizer.zero_grad()
        logits = model(ids)
        loss = criterion(logits, lab)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * lab.size(0)
        correct += (logits.argmax(1) == lab).sum().item()
        total += lab.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        lab = batch["label"].to(device)
        logits = model(ids)
        loss = criterion(logits, lab)
        total_loss += loss.item() * lab.size(0)
        pred = logits.argmax(1)
        correct += (pred == lab).sum().item()
        total += lab.size(0)
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(lab.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def main():
    # 1. 數據
    df = load_data()
    print(f"📊 數據: {len(df)} 條")
    print(df["label"].value_counts())
    
    # 2. 分詞 + 詞表
    print("🔧 分詞...")
    texts = [preprocess_chinese(t) for t in df["text"]]
    labels = df["label"].values
    
    vocab = build_vocab(texts, min_freq=2)
    print(f"📚 詞表大小: {len(vocab)}")
    
    # 3. 切分
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels,
    )
    
    train_ds = TextDataset([None] * len(X_tr), y_tr, vocab)
    train_ds.texts = [encode(t, vocab) for t in X_tr]
    test_ds = TextDataset([None] * len(X_te), y_te, vocab)
    test_ds.texts = [encode(t, vocab) for t in X_te]
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    # 4. 模型
    model = TextCNN(vocab_size=len(vocab), n_classes=2).to(DEVICE)
    print(f"🧠 模型參數量: {count_parameters(model):,}")
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 5. 訓練
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_preds = 0, None
    for epoch in range(1, 11):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        va_loss, va_acc, preds, labs = evaluate(model, test_loader, criterion, DEVICE)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"Epoch {epoch:02d} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_preds = preds
    
    # 6. 視覺化
    plot_training_history(history)
    plot_confusion_matrix(labs, best_preds, labels=["負面", "正面"])
    
    print(f"\n✅ 最佳測試準確率: {best_acc:.4f}")
    print("✅ Lab 02 完成！")


if __name__ == "__main__":
    main()
