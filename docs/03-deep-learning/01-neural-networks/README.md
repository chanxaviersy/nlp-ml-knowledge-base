# 🧠 Neural Networks | 神經網路

> 摘要：本章介紹神經網路的基本結構、數學原理與訓練方法。從單層感知器（Perceptron）到多層感知器（MLP），涵蓋前向傳播、反向傳播與各種激活函數。  
> Abstract: This chapter introduces the fundamentals of neural networks — from the perceptron to multi-layer perceptrons (MLPs), covering forward propagation, backpropagation, and activation functions.

---

## 1. 從神經元到感知器 | From Neuron to Perceptron

### 1.1 生物靈感 | Biological Inspiration

人腦神經元：
- **樹突 (Dendrites)**：接收輸入信號
- **細胞核 (Soma)**：處理信號
- **軸突 (Axon)**：傳遞輸出信號

人工神經元是對這一結構的數學抽象。

### 1.2 感知器 | Perceptron (1957, Rosenblatt)

單層感知器是最簡單的神經網路，只能解決線性可分的問題。

$$\hat{y} = \sigma(\mathbf{w} \cdot \mathbf{x} + b)$$

其中 $\sigma$ 是激活函數（二分類時常用階躍函數）。

**感知器學習規則**：

$$\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y}) \mathbf{x}$$

```python
import numpy as np

class Perceptron:
    """單層感知器實現"""
    
    def __init__(self, n_features, learning_rate=0.1):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = learning_rate
    
    def predict(self, x):
        """前向傳播：線性組合 + 符號激活"""
        z = np.dot(x, self.w) + self.b
        return np.where(z >= 0, 1, -1)
    
    def fit(self, X, y, epochs=100):
        """訓練：感知器學習規則"""
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                update = self.lr * (yi - y_pred)
                self.w += update * xi
                self.b += update
                errors += int(yi != y_pred)
            
            if errors == 0:
                print(f"Converged at epoch {epoch}")
                break
        
        return self

# 測試：線性可分的 AND 問題
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([-1, -1, -1, 1])

p = Perceptron(n_features=2, learning_rate=0.1)
p.fit(X_and, y_and)

print("Predictions:", p.predict(X_and))
print("Weights:", p.w, "Bias:", p.b)
```

### 1.3 感知器的局限性 | Limitations

**XOR 問題**（Minsky & Papert, 1969）：感知器無法解決 XOR 這樣的非線性可分問題，這導致了第一次 AI 寒冬。

```
XOR Truth Table:     感知器無法分開：
  Input  Output         ○  ○
  0, 0     0            ○  ●  ← 對角線無法用一條直線分開
  0, 1     1
  1, 0     1
  1, 1     0
```

**解決方案**：多層感知器（MLP）→ 引入隱藏層！

---

## 2. 多層感知器 | Multi-Layer Perceptron (MLP)

### 2.1 網路結構 | Network Architecture

```
輸入層        隱藏層 1        隱藏層 2        輸出層
[ x₁ ]       [ a₁¹ ]         [ a₁² ]         [ ŷ₁ ]
[ x₂ ] ──→   [ a₂¹ ] ──→     [ a₂² ] ──→     [ ŷ₂ ]
[ x₃ ]       [ a₃¹ ]         [ a₃² ]         [ ŷ₃ ]
 ...         [ ... ]         [ ... ]         [ ... ]
[ x_n ]      [ a_h¹ ]       [ a_h² ]       [ ŷ_c ]
```

**前向傳播 (Forward Propagation)**：

$$\mathbf{a}^{(0)} = \mathbf{x}$$

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$

$$\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$$

其中 $l = 1, 2, ..., L$ 為網路層數，$L$ 為總層數。

### 2.2 激活函數 | Activation Functions

| 函數 | 公式 | 輸出範圍 | 優點 | 缺點 |
|------|------|----------|------|------|
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | 二分類輸出 | 梯度消失、not zero-centered |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | zero-centered | 梯度消失 |
| **ReLU** | $\max(0, x)$ | $[0, +\infty)$ | 高效、緩解梯度消失 | Dying ReLU |
| **Leaky ReLU** | $\max(\alpha x, x)$ | $(-\infty, +\infty)$ | 解決 Dying ReLU | 超參數 $\alpha$ |
| **ELU** | $x$ if $x > 0$, $\alpha(e^x-1)$ otherwise | $(-\alpha, +\infty)$ | 更平滑、負值輸出 | 計算稍慢 |
| **GELU** | $x \cdot \Phi(x)$ | $(-\infty, +\infty)$ | Transformer 默認 | 計算複雜 |
| **Softmax** | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | $(0, 1)^K$ | 多分類概率輸出 | 溢出風險 |

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -700, 700)) - 1))

def gelu(x):
    """Gaussian Error Linear Unit — BERT、GPT 等 Transformer 的標準激活函數"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    """多分類輸出的標準激活函數"""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)  # 防止溢出
    return np.exp(x_shifted) / np.sum(np.exp(x_shifted), axis=-1, keepdims=True)

# 測試
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print("Input:", x)
print("Sigmoid:", sigmoid(x))
print("ReLU:", relu(x))
print("GELU:", gelu(x))
print("Softmax([1,2,3]):", softmax(np.array([[1, 2, 3]])))
```

---

## 3. 反向傳播 | Backpropagation

### 3.1 核心思想 | Core Idea

反向傳播是鏈式法則在神經網路中的高效應用，計算損失函數對每個參數的梯度。

**目標**：計算 $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}$ 對所有層 $l$。

### 3.2 梯度推導 | Gradient Derivation

以均方誤差 (MSE) 損失為例：

$$\mathcal{L} = \frac{1}{2} \| \mathbf{y} - \hat{\mathbf{y}} \|^2$$

**輸出層 ($l = L$)**：

$$\frac{\partial \mathcal{L}}{\partial \hat{y}_j} = -(y_j - \hat{y}_j)$$

$$\frac{\partial \mathcal{L}}{\partial z_j^{(L)}} = \frac{\partial \mathcal{L}}{\partial \hat{y}_j} \cdot \sigma'(z_j^{(L)})$$

**誤差項遞推**：

$$\boldsymbol{\delta}^{(l)} = (\mathbf{W}^{(l+1)})^\top \boldsymbol{\delta}^{(l+1)} \odot \sigma'(\mathbf{z}^{(l)})$$

其中 $\odot$ 為逐元素乘法 (Hadamard product)。

**參數梯度**：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^\top$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$

### 3.3 完整實現 | Full Implementation

```python
import numpy as np
from typing import List, Tuple, Callable

class MLP:
    """多層感知器從零實現（NumPy）"""
    
    def __init__(self, layer_sizes: List[int], 
                 activation: Callable = relu,
                 output_activation: Callable = None):
        """
        參數:
            layer_sizes: 每層神經元數量 [input, hidden1, ..., output]
            activation: 隱藏層激活函數
            output_activation: 輸出層激活函數
        """
        self.activation = activation
        self.output_activation = output_activation
        self.weights = []
        self.biases = []
        
        # Xavier 初始化
        for i in range(len(layer_sizes) - 1):
            w_shape = (layer_sizes[i], layer_sizes[i+1])
            # Xavier/Glorot 初始化
            w = np.random.randn(*w_shape) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """前向傳播，返回輸出與所有層的激活值（用於反向傳播）"""
        a = X
        cache = [a]  # 儲存每一層的激活值供反向傳播使用
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            if i < len(self.weights) - 1:
                a = self.activation(z)  # 隱藏層激活
            else:
                a = self.output_activation(z) if self.output_activation else z  # 輸出層
            cache.append((z, a))
        
        return a, cache
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 cache: List) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """反向傳播計算梯度"""
        m = y_true.shape[0]
        grads_w = []
        grads_b = []
        
        # 輸出層誤差 (以 MSE 為例)
        delta = (y_pred - y_true) / m
        delta = delta.reshape(-1, 1) if delta.ndim == 1 else delta
        
        for l in reversed(range(len(self.weights))):
            z, a = cache[l + 1]
            a_prev = cache[l]
            
            # 梯度：dL/dW = a_prev^T · delta
            grad_w = np.dot(a_prev, delta) if a_prev.ndim == 2 else np.outer(a_prev, delta)
            grad_b = np.sum(delta, axis=0)
            
            grads_w.insert(0, grad_w)
            grads_b.insert(0, grad_b)
            
            # 向前傳播誤差（除非是最後一層輸出層）
            if l > 0:
                delta = np.dot(delta, self.weights[l].T)
                # 乘以激活函數的導數
                delta = delta * self.activation_derivative(z)
        
        return grads_w, grads_b
    
    @staticmethod
    def activation_derivative(x):
        """ReLU 的導數"""
        return (x > 0).astype(float)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 1000, lr: float = 0.01,
            batch_size: int = 32, verbose: int = 100):
        """Mini-batch 梯度下降訓練"""
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Mini-batch 隨機打亂
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # 前向傳播
                y_pred, cache = self.forward(X_batch)
                
                # 計算損失 (MSE)
                loss = 0.5 * np.mean((y_pred - y_batch) ** 2)
                epoch_loss += loss
                
                # 反向傳播
                grads_w, grads_b = self.backward(y_batch, y_pred, cache)
                
                # 更新參數
                for i, (dw, db) in enumerate(zip(grads_w, grads_b)):
                    self.weights[i] -= lr * dw
                    self.biases[i] -= lr * db
            
            if verbose and epoch % verbose == 0:
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch:5d} | Loss: {avg_loss:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]


# ─── 實驗：XOR 問題 ───
print("=== XOR Problem ===")
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([[0], [1], [1], [0]], dtype=float)

mlp = MLP([2, 8, 8, 1], activation=relu, output_activation=sigmoid)
mlp.fit(X_xor, y_xor, epochs=5000, lr=0.5, verbose=500)

print("XOR Predictions:")
for x, y in zip(X_xor, y_xor):
    pred = mlp.predict(x.reshape(1, -1))[0, 0]
    print(f"  Input: {x} | True: {y[0]} | Predicted: {pred:.4f}")
```

---

## 4. 訓練技巧 | Training Tips

### 4.1 權重初始化 | Weight Initialization

| 方法 | 公式 | 適用場景 |
|------|------|----------|
| **Zero** | $\mathbf{W} = 0$ | ❌ 錯誤：所有神經元相同 |
| **Random** | $\mathbf{W} \sim \mathcal{N}(0, 1)$ | ❌ 梯度爆炸/消失 |
| **Xavier/Glorot** | $\mathbf{W} \sim \mathcal{N}(0, \frac{2}{n_{in}+n_{out}})$ | Sigmoid, Tanh |
| **He/Kaiming** | $\mathbf{W} \sim \mathcal{N}(0, \frac{2}{n_{in}})$ | ReLU 系列 |
| **Orthogonal** | $\mathbf{W} = \mathbf{Q}$ (正交矩陣) | RNN、LSTM |

### 4.2 Batch Size 的影響

| Batch Size | 優勢 | 劣勢 |
|-----------|------|------|
| SGD (m=1) | 噪聲大，收斂到平坦最小值 | 不穩定，收斂慢 |
| Mini-batch (32-256) | 平衡速度與穩定性 | 需要調整學習率 |
| Full-batch | 精確梯度估計 | 內存限制大，收斂慢 |

### 4.3 Gradient Clipping

防止梯度爆炸：

$$\mathbf{g} = \begin{cases} \mathbf{g} \cdot \frac{threshold}{\|\mathbf{g}\|} & \text{if } \|\mathbf{g}\| > threshold \\ \mathbf{g} & \text{otherwise} \end{cases}$$

---

## 總結 | Summary

```
┌─────────────────────────────────────────────────────────┐
│  神經網路核心流程                                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  輸入 x → [線性組合] → [激活] → [線性組合] → [激活] → 輸出 ŷ
│                ↑                ↑                       │
│               W¹, b¹            W², b²                  │
│                                                         │
│  反向傳播：損失 → ∂L/∂W² → ∂L/∂W¹ → 更新參數           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

| 組件 | 作用 |
|------|------|
| 激活函數 | 引入非線性，使網路能擬合複雜函數 |
| 隱藏層 | 學習特徵表示 |
| 反向傳播 | 高效計算梯度的鏈式法則應用 |
| Xavier/He 初始化 | 保持前向/反向傳播時信號方差穩定 |
| ReLU + Adam | 目前深度學習的默認組合 |

---

## 參考文獻 | References

1. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Ch. 6. MIT Press.
3. LeCun, Y., Bottou, L., Orr, G. B., & Müller, K. R. (2012). Efficient BackProp. *Neural Networks: Tricks of the Trade*.

---

*© 2026 chanxaviersy*
