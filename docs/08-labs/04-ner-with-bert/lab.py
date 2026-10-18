#!/usr/bin/env python3
"""
Lab 04: BERT 命名實體識別 (NER) - Token Classification
======================================================
使用 bert-base-chinese 對中文文本進行 BIO 序列標註。

Author: chanxaviersy
Date: 2026-10-18
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from seqeval.metrics import classification_report as seq_report, f1_score as seq_f1

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.utils import set_seed, get_device, count_parameters, timer

set_seed(42)
DEVICE = get_device()
print(f"📱 設備: {DEVICE}")


# ============================================================================
# 1. 標籤集
# ============================================================================

TAG2ID = {
    "O": 0,
    "B-PER": 1, "I-PER": 2,
    "B-LOC": 3, "I-LOC": 4,
    "B-ORG": 5, "I-ORG": 6,
    "B-MISC": 7, "I-MISC": 8,
}
ID2TAG = {v: k for k, v in TAG2ID.items()}


# ============================================================================
# 2. 數據
# ============================================================================

def load_bio(filepath: str) -> tuple[list[list[str]], list[list[str]]]:
    """載入 BIO 格式 NER 數據"""
    sentences, tags = [], []
    tokens, tag_seq = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    tags.append(tag_seq)
                    tokens, tag_seq = [], []
                continue
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                tag_seq.append(parts[-1])
    if tokens:
        sentences.append(tokens)
        tags.append(tag_seq)
    return sentences, tags


def load_data():
    data_dir = Path(__file__).resolve().parent / "data"
    train_path = data_dir / "sample_train.txt"
    test_path = data_dir / "sample_test.txt"
    if not train_path.exists():
        # 嘗試下載
        try:
            import urllib.request
            print("📥 下載 MSRA NER 數據...")
            base = "https://raw.githubusercontent.com/lemonhu/NER-data/master/MSRA"
            for f in ["train.txt", "test.txt"]:
                urllib.request.urlretrieve(f"{base}/{f}", data_dir / f"msra_{f}")
            train_path = data_dir / "msra_train.txt"
            test_path = data_dir / "msra_test.txt"
        except Exception as e:
            print(f"⚠️  下載失敗: {e}")
            raise FileNotFoundError(f"請確認 {data_dir} 中有 sample_train.txt")
    return load_bio(str(train_path)), load_bio(str(test_path))


# ============================================================================
# 3. Dataset
# ============================================================================

class NERDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, tag2id, max_len=128):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        words = self.sentences[idx]
        labels = self.tags[idx]
        
        # 使用 is_split_into_words 處理預分詞
        enc = self.tokenizer(
            words, is_split_into_words=True,
            truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        
        # 對齊 labels：subword 仍使用第一個 subword 的標籤
        word_ids = enc.word_ids(0)
        label_ids = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_word:
                label_ids.append(self.tag2id.get(labels[wid], 0))
            else:
                label_ids.append(-100)  # subword 內不計算 loss
            prev_word = wid
        
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


# ============================================================================
# 4. 評估 (使用 seqeval)
# ============================================================================

def align_predictions(preds, labels, id2tag):
    """將預測結果還原為 token 級標籤"""
    pred_tags, true_tags = [], []
    for p_seq, l_seq in zip(preds, labels):
        ps, ts = [], []
        for p, l in zip(p_seq, l_seq):
            if l != -100:
                ps.append(id2tag[p])
                ts.append(id2tag[l])
        pred_tags.append(ps)
        true_tags.append(ts)
    return pred_tags, true_tags


@torch.no_grad()
def evaluate(model, loader, id2tag):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        am = batch["attention_mask"].to(DEVICE)
        lab = batch["labels"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=am)
        pred = out.logits.argmax(-1)
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(lab.cpu().tolist())
    pred_tags, true_tags = align_predictions(all_preds, all_labels, id2tag)
    return pred_tags, true_tags


# ============================================================================
# 5. 主流程
# ============================================================================

def main():
    # 1. 數據
    (train_sents, train_tags), (test_sents, test_tags) = load_data()
    print(f"📊 訓練: {len(train_sents)} 句, 測試: {len(test_sents)} 句")
    print(f"標籤分佈: {set(t for seq in train_tags for t in seq)}")
    
    # 2. 模型
    MODEL_NAME = "bert-base-chinese"
    print(f"🤖 載入 {MODEL_NAME}...")
    from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(TAG2ID), id2label=ID2TAG, label2id=TAG2ID,
    ).to(DEVICE)
    print(f"🧠 參數量: {count_parameters(model):,}")
    
    # 3. DataLoader
    train_ds = NERDataset(train_sents, train_tags, tokenizer, TAG2ID)
    test_ds = NERDataset(test_sents, test_tags, tokenizer, TAG2ID)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)
    
    # 4. 優化
    EPOCHS = 5
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps,
    )
    
    # 5. 訓練
    print("\n🚀 開始訓練...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader, 1):
            ids = batch["input_ids"].to(DEVICE)
            am = batch["attention_mask"].to(DEVICE)
            lab = batch["labels"].to(DEVICE)
            out = model(input_ids=ids, attention_mask=am, labels=lab)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        
        # 驗證
        pred_tags, true_tags = evaluate(model, test_loader, ID2TAG)
        f1 = seq_f1(true_tags, pred_tags)
        print(f"Epoch {epoch}/{EPOCHS} | loss={avg_loss:.4f} | F1={f1:.4f}")
    
    # 6. 最終報告
    print("\n" + "="*60)
    print("📊 最終評估報告")
    print("="*60)
    print(seq_report(true_tags, pred_tags))
    
    # 7. 互動測試
    print("="*60)
    print("🧪 互動測試 (輸入 q 退出)")
    print("="*60)
    model.eval()
    while True:
        try:
            text = input("\n請輸入中文句子: ").strip()
        except EOFError:
            break
        if not text or text.lower() == "q":
            break
        words = list(text)
        enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
            pred_ids = out.logits.argmax(-1)[0].cpu().tolist()
        word_ids = enc.word_ids(0)
        prev = None
        result = []
        for w, wid, p in zip(words, word_ids, pred_ids):
            if wid is not None and wid != prev:
                tag = ID2TAG[p]
                if tag != "O":
                    result.append(f"{w}({tag})")
            prev = wid
        print(f"實體: {', '.join(result) if result else '(無)'}")
    
    print("\n✅ Lab 04 完成！")


if __name__ == "__main__":
    main()
