#!/usr/bin/env python3
"""
Lab 03: Transformer 情感分析 - HuggingFace BERT 微調
====================================================
使用 bert-base-chinese 對 ChnSentiCorp 進行微調。

Author: chanxaviersy
Date: 2026-10-18
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

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
# 1. 載入
# ============================================================================

def load_data():
    data_dir = Path(__file__).resolve().parent / "data"
    try:
        print("📥 下載 ChnSentiCorp...")
        from datasets import load_dataset
        ds_train = load_dataset("lansinuote/ChnSentiCorp", split="train")
        ds_test = load_dataset("lansinuote/ChnSentiCorp", split="test")
        train_df = pd.DataFrame({"text": ds_train["text"], "label": ds_train["label"]})
        test_df = pd.DataFrame({"text": ds_test["text"], "label": ds_test["label"]})
    except Exception as e:
        print(f"⚠️  在線下載失敗 ({e})，使用本地示例")
        train_df = pd.read_csv(data_dir / "sample_train.csv")
        test_df = pd.read_csv(data_dir / "sample_test.csv")
    return train_df.dropna(), test_df.dropna()


# ============================================================================
# 2. Dataset
# ============================================================================

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": self.labels[idx],
        }


# ============================================================================
# 3. 訓練/評估
# ============================================================================

@timer
def train_epoch(model, loader, optimizer, scheduler, criterion, device, log_every=50):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for step, batch in enumerate(loader, 1):
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        lab = batch["label"].to(device)
        
        optimizer.zero_grad()
        out = model(input_ids=ids, attention_mask=am, labels=lab)
        loss = out.loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * lab.size(0)
        correct += (out.logits.argmax(1) == lab).sum().item()
        total += lab.size(0)
        
        if step % log_every == 0:
            print(f"  step {step:4d} | loss={loss.item():.4f} | acc={correct/total:.4f}")
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        lab = batch["label"].to(device)
        out = model(input_ids=ids, attention_mask=am)
        pred = out.logits.argmax(1)
        correct += (pred == lab).sum().item()
        total += lab.size(0)
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(lab.cpu().tolist())
    return correct / total, all_preds, all_labels


def main():
    # 1. 載入
    train_df, test_df = load_data()
    print(f"📊 訓練: {len(train_df)}, 測試: {len(test_df)}")
    
    # 2. 為加速 demo，使用子集
    SUBSET_TRAIN = min(3000, len(train_df))
    SUBSET_TEST = min(1000, len(test_df))
    train_df = train_df.sample(SUBSET_TRAIN, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(SUBSET_TEST, random_state=42).reset_index(drop=True)
    print(f"📊 子集 - 訓練: {len(train_df)}, 測試: {len(test_df)}")
    
    # 3. Tokenizer & Model
    MODEL_NAME = "bert-base-chinese"
    print(f"🤖 載入 {MODEL_NAME}...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    print(f"🧠 參數量: {count_parameters(model):,}")
    
    # 4. DataLoader
    train_ds = SentimentDataset(train_df["text"], train_df["label"], tokenizer)
    test_ds = SentimentDataset(test_df["text"], test_df["label"], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    # 5. 優化器 & 調度器
    EPOCHS = 3
    total_steps = len(train_loader) * EPOCHS
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps,
    )
    
    # 6. 訓練
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_preds, best_labels = 0, None, None
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, None, DEVICE)
        va_acc, preds, labs = evaluate(model, test_loader, DEVICE)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(f"Epoch {epoch}/{EPOCHS} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | val acc={va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_preds, best_labels = preds, labs
    
    # 7. 視覺化
    plot_training_history(history)
    plot_confusion_matrix(best_labels, best_preds, labels=["負面", "正面"])
    
    print(f"\n✅ 最佳驗證準確率: {best_acc:.4f}")
    
    # 8. 互動測試
    print("\n" + "="*60)
    print("🧪 互動測試 (輸入 q 退出)")
    print("="*60)
    model.eval()
    while True:
        try:
            text = input("\n請輸入評論: ").strip()
        except EOFError:
            break
        if not text or text.lower() == "q":
            break
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1)[0]
            pred = out.logits.argmax(1).item()
        sentiment = "😊 正面" if pred == 1 else "😞 負面"
        print(f"結果: {sentiment}  (置信度={probs[pred]:.4f})")
    
    print("\n✅ Lab 03 完成！")


if __name__ == "__main__":
    main()
