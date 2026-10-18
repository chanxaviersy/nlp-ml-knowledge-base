# 🚀 深度學習優化 | Optimization

> 摘要：優化是深度學習訓練的核心。本章介紹梯度下降變體（SGD、Momentum、Adam）、學習率調度、正則化技術（Dropout、BatchNorm、Weight Decay）以及訓練技巧。  
> Abstract: Optimization is the heart of deep learning training. This chapter covers gradient descent variants, learning rate schedules, regularization techniques, and training tricks.

---

## 1. 優化挑戰 | Optimization Challenges

### 1.1 非凸優化

神經網路損失函數是非凸的，存在大量局部最小值、鞍點。

### 1.2 病態曲率

損失函數在不同方向上曲率差異巨大（condition number 大）。

### 1.3 梯度消失/爆炸

深層網路中梯度指數級衰減或增長。

---

## 2. 梯度下降變體 | Gradient Descent Variants

### 2.1 三種形式

| 類型 | 每次更新使用樣本 | 優點 | 缺點 |
|------|------------------|------|------|
| **BGD (Batch GD)** | 全部 $N$ | 精確收斂 | 慢、內存大 |
| **SGD** | 1 | 快、可逃離局部最優 | 噪聲大 |
| **Mini-batch GD** | $B$ (32-512) | 平衡 | 主流 |

### 2.2 SGD 與變體

**SGD (Stochastic Gradient Descent)**：

$$\theta \leftarrow \theta - \eta \nabla \mathcal{L}(\theta; x_i, y_i)$$

**Momentum**：

$$v_t = \gamma v_{t-1} + \eta \nabla \mathcal{L}(\theta_{t-1})$$

$$\theta_t = \theta_{t-1} - v_t$$

**Nesterov Accelerated Gradient (NAG)**：

$$v_t = \gamma v_{t-1} + \eta \nabla \mathcal{L}(\theta_{t-1} - \gamma v_{t-1})$$

```python
import torch.optim as optim

# PyTorch 中的優化器
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)
optimizer_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer_nesterov = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

### 2.3 自適應學習率方法

**AdaGrad**（對低頻特徵增大更新）：

$$g_t = \nabla \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

其中 $G_t = \sum_{i=1}^{t} g_i \odot g_i$。

**RMSProp**（解決 AdaGrad 學習率衰減過快）：

$$E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

**Adam**（結合 Momentum + RMSProp）：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(一階矩)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(二階矩)}$$
$$\hat{m}_t = m_t / (1-\beta_1^t), \quad \hat{v}_t = v_t / (1-\beta_2^t)$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

```python
optimizer_adam = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
optimizer_adamw = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
optimizer_lamb = optim.LAMB(model.parameters(), lr=1e-3)  # 大批次首選
```

### 2.4 優化器對比

| 優化器 | 適用場景 |
|--------|----------|
| **SGD + Momentum** | CV 模型、ResNet 訓練 |
| **Adam/AdamW** | Transformer、LLM 微調首選 |
| **AdaGrad** | 稀疏資料（如 NLP 詞嵌入） |
| **RMSProp** | RNN |
| **LAMB** | 大 batch size 訓練 |

---

## 3. 學習率調度 | Learning Rate Scheduling

### 3.1 常見策略

| 策略 | 公式 | 特性 |
|------|------|------|
| **Step Decay** | 每 N 步乘以 $\gamma$ | 階梯式下降 |
| **Exponential** | $\eta_t = \eta_0 \gamma^t$ | 平滑衰減 |
| **Cosine Annealing** | 餘弦曲線降到 0 | 現代主流 |
| **Warmup** | 前 N 步線性增加 | 大模型必備 |
| **One-Cycle** | 先升後降 | 收斂快 |
| **Reduce on Plateau** | 驗證集不變時降低 | 自適應 |

```python
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ExponentialLR, 
    LambdaLR, OneCycleLR, ReduceLROnPlateau
)

# Cosine Annealing with Warmup
num_warmup_steps = 1000
num_training_steps = 10000

def lr_lambda(current_step):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = LambdaLR(optimizer, lr_lambda)
```

### 3.2 學習率可視化

```python
import matplotlib.pyplot as plt
import numpy as np

steps = np.arange(0, 10000)
warmup_steps = 1000
lrs = []
for step in steps:
    if step < warmup_steps:
        lr = step / warmup_steps
    else:
        progress = (step - warmup_steps) / (10000 - warmup_steps)
        lr = 0.5 * (1 + np.cos(np.pi * progress))
    lrs.append(lr)

plt.plot(steps, lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Cosine Annealing with Linear Warmup')
plt.grid()
plt.show()
```

---

## 4. 正則化技術 | Regularization

### 4.1 L1 / L2 正則化

$$\mathcal{L}_{reg} = \mathcal{L}_{data} + \lambda \|\theta\|_p$$

```python
# L1 / L2 / L1+L2
optimizer_l1 = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0)  # 配合 L1 損失
optimizer_l2 = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)  # L2
```

### 4.2 Dropout

**訓練時**：以概率 $p$ 隨機丟棄神經元

```python
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            mask = (torch.rand(x.shape) > self.p).float()
            return x * mask / (1 - self.p)  # Inverted dropout
        return x
```

**推理時**：使用所有神經元（自動切換 model.eval()）。

**變體**：

| 變體 | 描述 |
|------|------|
| **Dropout** | 標準 |
| **DropConnect** | 丟棄連接（權重）而非神經元 |
| **Spatial Dropout** | 對整個特徵圖丟棄（CNN） |
| **DropPath (Stochastic Depth)** | 隨機丟棄整個殘差塊 |
| **Alpha Dropout** | 配合 SELU 保持均值方差 |

### 4.3 Batch Normalization

**訓練時**：對每個 mini-batch 標準化

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

**推理時**：使用移動平均的全局均值/方差

```python
class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

### 4.4 Layer Normalization (Transformer 標配)

```python
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
```

**BN vs LN**：

| 特性 | Batch Norm | Layer Norm |
|------|------------|------------|
| 標準化維度 | 跨 batch 同一特徵 | 跨同一樣本所有特徵 |
| 對 batch size 敏感 | ✅ 是 | ❌ 否 |
| RNN/Transformer | 不適合 | ✅ 首選 |
| CNN | ✅ 首選 | 也能用 |

### 4.5 數據增強 | Data Augmentation

**圖像**：
- 翻轉、裁剪、旋轉
- 顏色抖動
- Mixup：$\tilde{x} = \lambda x_i + (1-\lambda) x_j$
- CutMix：將一塊區域替換為另一張圖的對應區域

**NLP**：
- 詞替換、隨機插入、刪除
- 回譯（Back-translation）
- 上下文增強

```python
# NLP 增強示例
import random

def random_swap(words, n=2):
    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def random_deletion(words, p=0.1):
    if len(words) == 1:
        return words
    return [w for w in words if random.random() > p]
```

### 4.6 Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = EarlyStopping(patience=5)
for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## 5. 梯度處理 | Gradient Tricks

### 5.1 梯度裁剪

防止梯度爆炸：

```python
# 範數裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### 5.2 梯度累積

模擬大 batch：

```python
accumulation_steps = 4
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 6. 現代訓練技巧 | Modern Training Tricks

### 6.1 混合精度訓練 | Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, labels in dataloader:
    optimizer.zero_grad()
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 6.2 梯度檢查點 | Gradient Checkpointing

犧牲計算換記憶體：

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

### 6.3 分布式訓練

```python
# PyTorch DDP 簡化
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│  優化器選擇指南                                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  預訓練 ResNet/ConvNet        → SGD + Momentum + Cosine      │
│  Transformer 訓練            → AdamW + Warmup + Cosine        │
│  GAN 訓練                    → Adam (兩個網路分開學習率)     │
│  RL 訓練                     → Adam 或 RMSProp                │
│  大 batch 訓練 (>=8K)        → LAMB / LARS                   │
│  Few-shot 微調               → Adam + 低學習率                │
│                                                              │
│  正則化優先級：                                                │
│  1. 數據增強 (最有效)                                          │
│  2. Dropout                                                    │
│  3. Weight Decay                                               │
│  4. Batch/Layer Norm                                          │
│  5. Early Stopping                                            │
│  6. Label Smoothing                                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 參考文獻 | References

1. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*.
2. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR* (AdamW).
3. Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*.
4. Ioffe, S., & Szegedy, C. (2015). Batch Normalization. *ICML*.
5. Smith, L. N. (2017). Cyclical Learning Rates for Training Neural Networks. *WACV*.

---

*© 2026 chanxaviersy*
