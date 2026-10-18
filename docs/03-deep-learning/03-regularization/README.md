# 🛡️ 正則化與泛化 | Regularization & Generalization

> 摘要：正則化防止模型過擬合訓練資料，提高泛化能力。本章深入探討各種正則化技術、隱式正則化以及理解神經網路泛化的理論。  
> Abstract: Regularization prevents overfitting and improves generalization. This chapter covers regularization techniques, implicit regularization, and theories of neural network generalization.

---

## 1. 過擬合的根源 | Roots of Overfitting

### 1.1 過擬合現象

```
訓練誤差 vs 驗證誤差

  誤差
   ↑
   │         訓練誤差
   │  ──────────────────→  
   │            訓練誤差
   │  ╱╲              ╲
   │ ╱  ╲              ╲
   │╱    ╲              ╲
   │      ╲   驗證誤差   ╱
   │       ╲   ╱╲     ╱
   │        ╲ ╱  ╲   ╱
   │         V    ╲ ╱
   │               V   ← 過擬合開始
   └──────────────────────→ 訓練輪次
```

### 1.2 過擬合的數學描述

$$\text{Generalization Error} = |\text{Train Error} - \text{Test Error}|$$

**Hoeffding 不等式**：
$$P(|E_{in} - E_{out}| > \epsilon) \leq 2 \cdot 2^{d} e^{-2N\epsilon^2}$$

其中 $d$ 是模型複雜度（VC 維），$N$ 是樣本數。

---

## 2. 顯式正則化 | Explicit Regularization

### 2.1 參數範數正則化

**L2 正則化 (Ridge / Weight Decay)**：

$$\mathcal{L}_{reg} = \mathcal{L}_{data} + \lambda \sum_i \theta_i^2$$

**L1 正則化 (Lasso)**：

$$\mathcal{L}_{reg} = \mathcal{L}_{data} + \lambda \sum_i |\theta_i|$$

**Elastic Net**：
$$\mathcal{L}_{reg} = \mathcal{L}_{data} + \lambda_1 \sum_i |\theta_i| + \lambda_2 \sum_i \theta_i^2$$

**L∞ 正則化**：
$$\mathcal{L}_{reg} = \mathcal{L}_{data} + \lambda \max_i |\theta_i|$$

| 類型 | 效果 | 產生稀疏性 |
|------|------|-----------|
| L1 | 軟閾值 | 是 |
| L2 | 權重衰減 | 否 |
| Elastic Net | L1+L2 | 是 |
| Dropout | 神經元級 | 是（隱式） |

### 2.2 Label Smoothing

**原理**：將硬標籤軟化

$$y_{smooth} = (1-\epsilon) y_{one-hot} + \frac{\epsilon}{K}$$

```python
import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    
    def forward(self, logits, target):
        log_probs = torch.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# 使用
criterion = LabelSmoothingLoss(num_classes=10, smoothing=0.1)
```

**效果**：
- 防止模型過度自信
- 提升校準度（calibration）
- 略微提升準確率

---

## 3. 隱式正則化 | Implicit Regularization

### 3.1 SGD 的隱式正則化

即使不加任何正則化項，SGD 本身就傾向於找到**平坦最小值**（flat minima），泛化更好。

```
       損失值
        ↑
        │      ╱╲  尖銳最小值
        │     ╱  ╲
        │   ╱    ╲___
        │  ╱ 平坦最小值 (SGD 傾向)
        │ ╱___________
        └──────────────→ 參數空間
```

### 3.2 早停 | Early Stopping

**早停 = 時間維度的正則化**。

```python
# 簡單實現
best_val_loss = float('inf')
patience = 5
patience_counter = 0
best_model_state = None

for epoch in range(100):
    train_loss = train()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            model.load_state_dict(best_model_state)
            break
```

### 3.3 噪聲注入

| 噪聲類型 | 描述 |
|----------|------|
| **輸入噪聲** | 在輸入 $x$ 上加噪聲 $\mathcal{N}(0, \sigma^2)$ |
| **權重噪聲** | 訓練時對權重加噪聲 |
| **標籤噪聲** | Label Smoothing、Label Flipping |
| **梯度噪聲** | 對梯度加噪聲 |

### 3.4 DropBlock / Spatial Dropout

**DropBlock** 對 CNN 特別有效（相鄰像素高度相關）：

```python
class DropBlock(nn.Module):
    def __init__(self, block_size=7, drop_prob=0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        
        # 計算 gamma
        gamma = self._compute_gamma(x)
        mask = torch.bernoulli(torch.full_like(x[:, :1, :, :], 1 - gamma))
        mask_block = 1 - F.max_pool2d(mask, kernel_size=self.block_size, stride=1)
        mask_block = 1 - F.pad(mask_block, [self.block_size//2]*4, mode='constant', value=0)
        return x * mask_block * mask_block.numel() / mask_block.sum()
```

---

## 4. 結構正則化 | Structural Regularization

### 4.1 參數共享

- **CNN**：卷積核權重共享
- **RNN**：時間步共享
- **Transformer**：Self-Attention 跨位置共享

### 4.2 多任務學習

通過共享表示，相關任務之間的正則化效應。

### 4.3 集成方法

```python
# Snapshot Ensemble: 一個訓練週期保存多個模型
class SnapshotEnsemble:
    def __init__(self, n_snapshots=5):
        self.n_snapshots = n_snapshots
        self.snapshots = []
    
    def save_snapshot(self, model):
        self.snapshots.append(deepcopy(model.state_dict()))
```

### 4.4 知識蒸餾

用大模型（teacher）指導小模型（student）：

$$\mathcal{L}_{KD} = \alpha \mathcal{L}_{CE}(y, \hat{y}_s) + (1-\alpha) T^2 \mathcal{L}_{KL}(p_t/T \| p_s/T)$$

```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=-1),
            F.softmax(teacher_logits / self.T, dim=-1),
            reduction='batchmean'
        ) * (self.T ** 2)
        
        hard_loss = F.cross_entropy(student_logits, labels)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss
```

---

## 5. 數據正則化 | Data Regularization

### 5.1 數據增強 | Data Augmentation

**圖像**：

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5),  # 隨機擦除
])
```

**NLP**：

```python
# 回譯增強
from transformers import MarianMTModel, MarianTokenizer

def back_translate(text, src='en', tgt='fr'):
    # 英 → 法 → 英
    model_name = f'Helsinki-NLP/opus-mt-{src}-{tgt}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # 翻譯
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 翻回
    model_name_back = f'Helsinki-NLP/opus-mt-{tgt}-{src}'
    # ... 類似流程
    return back_translated
```

### 5.2 Mixup 與 CutMix

```python
def mixup(x, y, alpha=0.2):
    """Mixup 增強"""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def cutmix(x, y, alpha=1.0):
    """CutMix 增強"""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    
    H, W = x.shape[2:]
    cut_rat = np.sqrt(1 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x, y, y[idx], lam
```

---

## 6. 多樣性正則化 | Diversity Regularization

### 6.1 Stochastic Depth (DropPath)

```python
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape) < keep_prob
        return x * mask / keep_prob
```

### 6.2 Shakedrop

在殘差連接處加入隨機縮放。

---

## 7. 自適應正則化方法 | Adaptive Regularization

### 7.1 根據訓練動態調整

```python
# 動態 Dropout
class AdaptiveDropout(nn.Module):
    def __init__(self, init_p=0.5):
        super().__init__()
        self.logit_p = nn.Parameter(torch.tensor(init_p).logit())
    
    def forward(self, x):
        p = torch.sigmoid(self.logit_p)
        return F.dropout(x, p=p.item(), training=self.training)
```

### 7.2 根據梯度調整

**梯度 Noise Injection**：

$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L} + \mathcal{N}(0, \sigma_t^2)$$

其中 $\sigma_t^2 = \frac{\eta}{(1+t)^\gamma}$。

---

## 8. 評估正則化效果 | Evaluating Regularization

### 8.1 過擬合診斷

```python
def check_overfitting(train_losses, val_losses, threshold=0.1):
    """簡單的過擬合檢測"""
    if len(train_losses) < 5:
        return "Need more epochs"
    
    recent_train = train_losses[-5:]
    recent_val = val_losses[-5:]
    
    train_trend = np.polyfit(range(5), recent_train, 1)[0]
    val_trend = np.polyfit(range(5), recent_val, 1)[0]
    
    if train_trend < 0 and val_trend > 0:
        return "Overfitting detected"
    elif train_trend > 0 and val_trend > 0:
        return "Underfitting - increase capacity"
    elif train_trend < 0 and val_trend < 0:
        return "Both improving"
    else:
        return "Stable"
```

### 8.2 泛化差距

$$\text{Generalization Gap} = E_{val} - E_{train}$$

越小越好。

---

## 總結 | Summary

| 正則化方法 | 強度 | 通用性 | 計算開銷 |
|-----------|------|--------|----------|
| **L1/L2** | 弱 | 高 | 極低 |
| **Dropout** | 中 | 高 | 低 |
| **Batch/Layer Norm** | 中 | 高 | 低 |
| **Label Smoothing** | 弱 | 高 | 極低 |
| **Data Augmentation** | 強 | 中 | 中 |
| **Mixup/CutMix** | 強 | 中 | 中 |
| **知識蒸餾** | 強 | 中 | 高 |
| **Early Stopping** | 中 | 高 | 極低 |
| **Noise Injection** | 弱 | 中 | 低 |
| **DropBlock** | 中 | 中 | 低 |

**黃金組合**（圖像分類）：
- Weight Decay (1e-4) + Dropout (0.1-0.3) + Data Aug + Label Smoothing + Early Stopping

**黃金組合**（Transformer）：
- AdamW + Weight Decay (0.1) + Dropout (0.1) + Label Smoothing + Warmup

---

## 參考文獻 | References

1. Srivastava, N., et al. (2014). Dropout. *JMLR*.
2. Szegedy, C., et al. (2016). Rethinking the Inception Architecture. *CVPR* (Label Smoothing).
3. Zhang, H., et al. (2018). mixup: Beyond Empirical Risk Minimization. *ICLR*.
4. Yun, S., et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers. *ICCV*.
5. Hinton, G., et al. (2015). Distilling the Knowledge in a Neural Network. *NeurIPS Workshop*.

---

*© 2026 chanxaviersy*
