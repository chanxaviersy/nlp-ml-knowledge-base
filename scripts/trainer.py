#!/usr/bin/env python3
"""
訓練通用工具
============

模型訓練相關工具：訓練循環、優化器工廠、學習率調度器。
"""

import time
from typing import Dict, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# ============================================================================
# 1. 優化器工廠
# ============================================================================

def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    **kwargs,
) -> Optimizer:
    """
    優化器工廠
    
    支持: sgd, adam, adamw, adafactor, lion
    """
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "adafactor":
        from transformers import Adafactor
        return Adafactor(model.parameters(), lr=lr, weight_decay=weight_decay, scale_parameter=False, relative_step=False)
    elif optimizer_name.lower() == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("pip install lion-pytorch")
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


# ============================================================================
# 2. 學習率調度器
# ============================================================================

def get_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = "cosine",
    num_training_steps: int = 1000,
    num_warmup_steps: int = 100,
    **kwargs,
) -> _LRScheduler:
    """學習率調度器工廠"""
    if scheduler_name == "linear":
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif scheduler_name == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif scheduler_name == "constant":
        from transformers import get_constant_schedule_with_warmup
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    elif scheduler_name == "cosine_with_hard_restarts":
        from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
        return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


# ============================================================================
# 3. 訓練器
# ============================================================================

class Trainer:
    """
    通用訓練器（簡化版）
    
    用法：
        trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, device)
        trainer.fit(epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or get_optimizer(model)
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and torch.cuda.is_available()
        
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    def train_one_epoch(self) -> tuple:
        """訓練一個 epoch"""
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in self.train_loader:
            # 將 batch 移到設備
            batch = self._to_device(batch)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    out = self.model(**self._inputs(batch)) if isinstance(batch, dict) else self.model(batch[0])
                    loss = self._compute_loss(out, batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self._forward(batch)
                loss = self._compute_loss(out, batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item() * self._batch_size(batch)
            if hasattr(out, "logits"):
                pred = out.logits.argmax(-1)
            elif isinstance(out, torch.Tensor):
                pred = out.argmax(-1)
            else:
                continue
            
            correct += (pred == self._labels(batch)).sum().item()
            total += self._batch_size(batch)
        
        return total_loss / total, correct / total
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """驗證"""
        if not self.val_loader:
            return 0.0, 0.0
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        for batch in self.val_loader:
            batch = self._to_device(batch)
            out = self._forward(batch)
            loss = self._compute_loss(out, batch)
            total_loss += loss.item() * self._batch_size(batch)
            if hasattr(out, "logits"):
                pred = out.logits.argmax(-1)
            elif isinstance(out, torch.Tensor):
                pred = out.argmax(-1)
            else:
                continue
            correct += (pred == self._labels(batch)).sum().item()
            total += self._batch_size(batch)
        return total_loss / total, correct / total
    
    def fit(self, epochs: int = 10, log_every: int = 1):
        """完整訓練"""
        print(f"🚀 開始訓練 {epochs} 個 epoch...")
        best_val_acc = 0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self.train_one_epoch()
            va_loss, va_acc = self.validate()
            elapsed = time.time() - t0
            
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)
            
            if epoch % log_every == 0:
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                    f"val loss={va_loss:.4f} acc={va_acc:.4f} | "
                    f"time={elapsed:.1f}s"
                )
            if va_acc > best_val_acc:
                best_val_acc = va_acc
        
        print(f"✅ 訓練完成！最佳 val_acc: {best_val_acc:.4f}")
        return self.history
    
    # ---------- 內部輔助 ----------
    def _to_device(self, batch):
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elif isinstance(batch, (tuple, list)):
            return [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in batch]
        return batch
    
    def _inputs(self, batch):
        if isinstance(batch, dict):
            return {k: v for k, v in batch.items() if k != "label" and k != "labels"}
        return batch[:-1]
    
    def _labels(self, batch):
        if isinstance(batch, dict):
            return batch.get("label", batch.get("labels"))
        return batch[-1]
    
    def _batch_size(self, batch):
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    return v.size(0)
        elif isinstance(batch, (tuple, list)):
            return batch[0].size(0)
        return 1
    
    def _forward(self, batch):
        if isinstance(batch, dict):
            return self.model(**{k: v for k, v in batch.items() if k != "label" and k != "labels"})
        return self.model(batch[0])
    
    def _compute_loss(self, out, batch):
        if hasattr(out, "loss"):
            return out.loss
        labels = self._labels(batch)
        return self.criterion(out, labels)


if __name__ == "__main__":
    print("✅ 訓練工具模組已加載")
    print("   使用 Trainer(model, loader, ...) 開始訓練")
