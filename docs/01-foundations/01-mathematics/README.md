# 📐 Mathematics for ML | ML 數學基礎

> 摘要：本章節涵蓋機器學習所需的數學基礎，包括微積分、機率論與資訊理論。這些數學工具是理解機器學習演算法的核心。  
> Abstract: This chapter covers the mathematical foundations required for machine learning, including calculus, probability theory, and information theory.

---

## 目錄 | Table of Contents

1. [微積分 | Calculus](#1-微積分--calculus)
2. [機率論 | Probability Theory](#2-機率論--probability-theory)
3. [資訊理論 | Information Theory](#3-資訊理論--information-theory)

---

## 1. 微積分 | Calculus

### 1.1 導數與梯度 | Derivatives and Gradients

導數描述函數的瞬時變化率。在 ML 中，梯度是損失函數相對於參數的導數，是優化算法的核心。

$$\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**偏導數 (Partial Derivative)**：多元函數對單一變數的導數

$$\frac{\partial f}{\partial x_i} \bigg|_{x_j, j \neq i} = \lim_{h \to 0} \frac{f(x_1, ..., x_i + h, ..., x_n) - f(x_1, ..., x_n)}{h}$$

**梯度 (Gradient)**：所有偏導數組成的向量

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### 1.2 鏈式法則 | Chain Rule

複合函數的導數，是反向傳播（Backpropagation）的數學基礎。

$$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$

**多維鏈式法則**：

$$\frac{\partial z}{\partial x} = \sum_k \frac{\partial z}{\partial u_k} \cdot \frac{\partial u_k}{\partial x}$$

### 1.3 積分 | Integration

積分是導數的逆運算，在機率密度函數和期望值的計算中至關重要。

$$\int_a^b f(x)\,dx = F(b) - F(a)$$

**多重積分**：

$$\int \int_S f(x,y)\,dx\,dy$$

### 1.4 梯度下降 | Gradient Descent

目標：最小化損失函數 $J(\theta)$

$$\theta_{new} = \theta_{old} - \alpha \nabla J(\theta_{old})$$

其中 $\alpha$ 為學習率 (Learning Rate)。

| 變體 | 描述 |
|------|------|
| SGD | 隨機梯度下降，每個樣本更新一次 |
| Mini-batch GD | 每批次更新，更常見 |
| Batched GD | 全量資料計算梯度，收斂更穩但慢 |

**代碼實現 | Code Implementation:**

```python
import numpy as np

def gradient_descent(gradient_fn, x_init, learning_rate=0.01, n_iterations=1000, tol=1e-6):
    """
    梯度下降優化器
    
    參數:
        gradient_fn: 梯度函數
        x_init: 初始參數
        learning_rate: 學習率 α
        n_iterations: 最大迭代次數
        tol: 收斂閾值
    """
    x = x_init.copy()
    history = [x.copy()]
    
    for i in range(n_iterations):
        grad = gradient_fn(x)
        x = x - learning_rate * grad
        history.append(x.copy())
        
        # 收斂檢查
        if np.linalg.norm(grad) < tol:
            print(f"Converged at iteration {i}")
            break
    
    return x, history
```

---

## 2. 機率論 | Probability Theory

### 2.1 基本概念 | Basic Concepts

**機率分佈 | Probability Distribution**:

$$P(X = x) \quad \text{或} \quad p(x)$$

**期望值 | Expected Value**:

$$\mathbb{E}[X] = \sum_x x \cdot P(X=x) \quad \text{(離散)}$$
$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f(x)\,dx \quad \text{(連續)}$$

**方差 | Variance**:

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**標準差 | Standard Deviation**:

$$\sigma_X = \sqrt{\text{Var}(X)}$$

### 2.2 常見分佈 | Common Distributions

| 分佈 | 參數 | 期望值 | 方差 | 應用場景 |
|------|------|--------|------|----------|
| Bernoulli | $p$ | $p$ | $p(1-p)$ | 二分類 |
| Binomial | $n, p$ | $np$ | $np(1-p)$ | n次試驗成功次數 |
| Gaussian/Normal | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ | 廣泛應用，自然現象 |
| Poisson | $\lambda$ | $\lambda$ | $\lambda$ | 稀有事件計數 |
| Exponential | $\lambda$ | $1/\lambda$ | $1/\lambda^2$ | 等待時間 |
| Beta | $\alpha, \beta$ | $\frac{\alpha}{\alpha+\beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ | 機率的先驗分佈 |

### 2.3 貝葉斯定理 | Bayes' Theorem

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

在 ML 中的意義：

| 術語 | 意義 |
|------|------|
| $P(A)$ | **先驗機率 (Prior)** — 觀察數據前對 A 的信念 |
| $P(A|B)$ | **後驗機率 (Posterior)** — 觀察數據 B 後對 A 的更新信念 |
| $P(B|A)$ | **似然 (Likelihood)** — 在 A 成立的條件下，B 出現的可能性 |
| $P(B)$ | **證據 (Evidence)** — B 的總體邊緣機率 |

**範例 | Example:**

```python
import numpy as np
from scipy import stats

# 貝葉斯更新示例：垃圾郵件分類
# P(spam|contains "free") = P(contains "free"|spam) * P(spam) / P(contains "free")

prior_spam = 0.3        # P(spam) = 30% of emails are spam
prior_ham = 0.7         # P(ham) = 70%

likelihood_free_given_spam = 0.9   # P("free"|spam) = 90%
likelihood_free_given_ham = 0.05   # P("free"|ham) = 5%

# 邊緣機率
p_free = likelihood_free_given_spam * prior_spam + likelihood_free_given_ham * prior_ham

# 後驗機率
posterior_spam = (likelihood_free_given_spam * prior_spam) / p_free
posterior_ham = (likelihood_free_given_ham * prior_ham) / p_free

print(f"P(spam|free) = {posterior_spam:.4f}")   # ≈ 0.8909
print(f"P(ham|free)  = {posterior_ham:.4f}")   # ≈ 0.1091
```

### 2.4 最大似然估計 | Maximum Likelihood Estimation (MLE)

給定觀察數據 $\mathcal{D} = \{x_1, ..., x_n\}$，找參數 $\theta$ 使似然函數最大化：

$$\hat{\theta}_{MLE} = \arg\max_\theta \mathcal{L}(\theta; \mathcal{D}) = \arg\max_\theta \prod_{i=1}^{n} p(x_i | \theta)$$

為計算方便，通常最大化 **對數似然 (Log Likelihood)**：

$$\hat{\theta}_{MLE} = \arg\max_\theta \sum_{i=1}^{n} \log p(x_i | \theta)$$

---

## 3. 資訊理論 | Information Theory

### 3.1 熵 | Entropy

資訊熵衡量隨機變數的不確定性（資訊量）：

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x) = \mathbb{E}[-\log p(X)]$$

- $\log_2$: 單位為 bits
- $\log_e$: 單位為 nats

**性質**：
- $H(X) \geq 0$（非負性）
- 均勻分佈時 $H(X)$ 最大
- 確定性分佈時 $H(X) = 0$

### 3.2 交叉熵 | Cross-Entropy

衡量兩個分佈 $p$（真實）和 $q$（預測）之間的差異：

$$H(p, q) = -\sum_x p(x) \log q(x) = H(p) + D_{KL}(p \| q)$$

在分類問題中，交叉熵損失（Cross-Entropy Loss）：
$$L_{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

### 3.3 KL 散度 | Kullback-Leibler Divergence

衡量從分佈 $q$ 近似 $p$ 的資訊損失：

$$D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

**非對稱性**：$D_{KL}(p \| q) \neq D_{KL}(q \| p)$

### 3.4 互資訊 | Mutual Information

兩個隨機變數之間的共享資訊量：

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = D_{KL}(p(X,Y) \| p(X)p(Y))$$

```python
import numpy as np

def entropy(x):
    """計算離散熵 H(X)"""
    # 計算經驗分佈
    _, counts = np.unique(x, return_counts=True)
    p = counts / len(x)
    # 避免 log(0)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def kl_divergence(p, q):
    """計算 KL 散度 D_KL(p || q)"""
    p = np.array(p)
    q = np.array(q)
    # 確保概率和為1
    p = p / p.sum()
    q = q / q.sum()
    # 避免 log(0)
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

# 示例
p = [0.5, 0.5]   # 均勻分佈
q = [0.9, 0.1]   # 偏移分佈

print(f"H(p) = {entropy(p):.4f} bits")           # 1.0
print(f"D_KL(p||q) = {kl_divergence(p, q):.4f}") # > 0
print(f"D_KL(q||p) = {kl_divergence(q, p):.4f}") # 非對稱
```

---

## 總結 | Summary

| 概念 | 機器學習中的應用 |
|------|----------------|
| 導數 / 梯度 | 反向傳播、神經網路訓練 |
| 偏導數 | 多變數模型（所有深度學習） |
| 鏈式法則 | 反向傳播算法 |
| 機率論 | 生成模型、貝葉斯方法、不確定性估計 |
| MLE | 監督學習參數估計 |
| 熵 | 決策樹、資訊增益、語言模型困惑度 |
| 交叉熵 | 分類損失函數 (CE Loss) |
| KL 散度 | 知識蒸餾、分佈對齊 |

---

## 參考文獻 | References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Murphy, K. P. (2022). *Probabilistic Machine Learning*. MIT Press.
3. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley.
4. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.

---

*© 2026 chanxaviersy | 本章節由 AI 輔助生成與校對*
