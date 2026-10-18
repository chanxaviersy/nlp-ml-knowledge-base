# 📊 Linear Algebra for ML | 機器學習線性代數

> 摘要：本章節介紹機器學習所需的線性代數基礎，包括向量、矩陣運算、特徵值分解及奇異值分解（SVD），這些是神經網路和大多數 ML 算法的數學骨架。  
> Abstract: This chapter introduces linear algebra foundations for machine learning, covering vectors, matrices, eigenvalues, and SVD — the mathematical backbone of neural networks and ML algorithms.

---

## 1. 向量 | Vectors

### 1.1 定義與基本運算 | Definition & Basic Operations

向量 $\mathbf{v} \in \mathbb{R}^n$ 是 $n$ 個實數的有序陣列：

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

**基本運算**：

| 運算 | 公式 | 代碼 |
|------|------|------|
| 加法 | $(\mathbf{u} + \mathbf{v})_i = u_i + v_i$ | `u + v` |
| 標量乘法 | $(c\mathbf{v})_i = c \cdot v_i$ | `c * v` |
| 點積 | $\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i$ | `np.dot(u, v)` |
| 範數 | $\\|\mathbf{v}\\| = \sqrt{\sum_i v_i^2}$ | `np.linalg.norm(v)` |

### 1.2 向量範數 | Vector Norms

| 範數 | 符號 | 定義 | 應用 |
|------|------|------|------|
| L1 | $\\|\mathbf{v}\\|_1$ | $\sum_i \|v_i\|$ | Lasso 回歸、稀疏性 |
| L2 (Euclidean) | $\\|\mathbf{v}\\|_2$ | $\sqrt{\sum_i v_i^2}$ | Ridge 回歸、距離度量 |
| L∞ | $\\|\mathbf{v}\\|_{\infty}$ | $\max_i \|v_i\|$ | 魯棒估計 |

### 1.3 向量幾何意義 | Geometric Interpretation

$$\cos \theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \cdot \|\mathbf{v}\|_2}$$

- **正交 (Orthogonal)**：$\mathbf{u} \cdot \mathbf{v} = 0$
- **夾角 (Angle)**：$\theta = \arccos(\cos \theta)$

```python
import numpy as np

def cosine_similarity(u, v):
    """計算兩個向量的餘弦相似度"""
    u, v = np.array(u), np.array(v)
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# 示例：詞向量相似度
word_embedding = {
    "king": np.array([0.8, 0.2, 0.1]),
    "queen": np.array([0.7, 0.3, 0.1]),
    "man": np.array([0.6, 0.8, 0.0]),
    "woman": np.array([0.5, 0.9, 0.1]),
}

print("king vs queen:", cosine_similarity(word_embedding["king"], word_embedding["queen"]))
print("king vs man:",   cosine_similarity(word_embedding["king"], word_embedding["man"]))
# king - man + woman ≈ queen (經典的 word2vec 類比)
analogy = word_embedding["king"] - word_embedding["man"] + word_embedding["woman"]
print("king - man + woman:", analogy)
```

---

## 2. 矩陣 | Matrices

### 2.1 定義與類型 | Definition & Types

矩陣 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 是 $m$ 行 $n$ 列的 2D 陣列：

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

**特殊矩陣**：

| 類型 | 定義 | 性質 |
|------|------|------|
| 方陣 (Square) | $m = n$ | 可有特徵值 |
| 對稱 (Symmetric) | $\mathbf{A} = \mathbf{A}^\top$ | 實數特徵值 |
| 反對稱 (Skew-symmetric) | $\mathbf{A} = -\mathbf{A}^\top$ | 純虛數特徵值 |
| 正定 (Positive Definite) | $\mathbf{x}^\top \mathbf{A} \mathbf{x} > 0, \forall \mathbf{x} \neq 0$ | 優化收斂保證 |
| 正交 (Orthogonal) | $\mathbf{A}^\top \mathbf{A} = \mathbf{I}$ | 保持向量範數 |
| 單位 (Identity) | $\mathbf{I}_{ij} = \delta_{ij}$ | 中性元素 |

### 2.2 矩陣運算 | Matrix Operations

**乘法**：

$$(\mathbf{A}\mathbf{B})_{ij} = \sum_k A_{ik} B_{kj}$$

**轉置 (Transpose)**：

$$(\mathbf{A}^\top)_{ij} = A_{ji}$$

**逆矩陣 (Inverse)**：

$$\mathbf{A}^{-1} \mathbf{A} = \mathbf{A} \mathbf{A}^{-1} = \mathbf{I}$$

**跡 (Trace)**：

$$\text{tr}(\mathbf{A}) = \sum_{i} A_{ii} = \text{特徵值之和}$$

**行列式 (Determinant)**：

$$\det(\mathbf{A}) = \prod_i \lambda_i \quad \text{(特徵值的積)}$$

### 2.3 矩陣範數 | Matrix Norms

| 範數 | 定義 |
|------|------|
| Frobenius | $\\|\mathbf{A}\\|_F = \sqrt{\sum_{ij} A_{ij}^2}$ |
| 誘導 L2 | $\\|\mathbf{A}\\|_2 = \sigma_{\max}(\mathbf{A})$ (最大奇異值) |
| 核範數 | $\\|\mathbf{A}\\|_* = \sum_i \sigma_i$ (奇異值之和，用於低秩近似) |

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B:\n", A + B)
print("A @ B (matmul):\n", A @ B)
print("A.T (transpose):\n", A.T)
print("np.linalg.inv(A):\n", np.linalg.inv(A))
print("np.linalg.det(A):", np.linalg.det(A))
print("np.linalg.norm(A, 'fro'):", np.linalg.norm(A, 'fro'))
print("np.trace(A):", np.trace(A))
```

---

## 3. 特徵值與奇異值 | Eigenvalues & Singular Values

### 3.1 特徵值分解 | Eigenvalue Decomposition (EVD)

對於方陣 $\mathbf{A} \in \mathbb{R}^{n \times n}$：

$$\mathbf{A} \mathbf{v} = \lambda \mathbf{v}$$

- $\lambda$：特徵值 (Eigenvalue)
- $\mathbf{v}$：特徵向量 (Eigenvector)

**譜定理 (Spectral Theorem)**：對稱矩陣可對角化為：

$$\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^\top$$

其中 $\mathbf{Q}$ 的列是標準正交特徵向量，$\mathbf{\Lambda} = \text{diag}(\lambda_1, ..., \lambda_n)$。

### 3.2 奇異值分解 | Singular Value Decomposition (SVD)

任何矩陣 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 都可以分解為：

$$\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$$

- $\mathbf{U} \in \mathbb{R}^{m \times m}$：左奇異向量（特徵向量 of $\mathbf{A}\mathbf{A}^\top$）
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$：奇異值（對角非負實數）
- $\mathbf{V} \in \mathbb{R}^{n \times n}$：右奇異向量（特徵向量 of $\mathbf{A}^\top \mathbf{A}$）

**截斷 SVD（低秩近似）**：

$$\mathbf{A} \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^\top, \quad k < \min(m,n)$$

最小化 Frobenius 範數誤差，是 PCA 和推薦系統的核心。

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 奇異值分解
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print("U:\n", U)
print("Singular values:", s)
print("Vt:\n", Vt)

# 重構（驗證）
Sigma = np.diag(s)
A_reconstructed = U @ Sigma @ Vt
print("Reconstructed A:\n", A_reconstructed)

# 截斷 SVD（k=2 低秩近似）
k = 2
A_lowrank = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
print("Low-rank (k=2) approximation:\n", A_lowrank)

# PCA 視角：用 SVD 做主成分分析
X = np.random.randn(100, 5)  # 100 samples, 5 features
U, s, Vt = np.linalg.svd(X, full_matrices=False)
components = Vt[:2, :]        # 前兩個主成分
projected = X @ components.T # 投影到前兩個主成分
print("Components shape:", components.shape)
print("Projected shape:", projected.shape)
```

---

## 4. 奇異值在 ML 中的應用 | SVD Applications in ML

### 4.1 主成分分析 (PCA)

PCA 等價於 SVD 的數據投影：

$$\text{PCA}: \mathbf{X} \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^\top$$

選擇前 $k$ 個奇異值解釋的方差比例：

$$\text{解釋方差} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2}, \quad r = \text{rank}(\mathbf{X})$$

### 4.2 詞嵌入降維 | Word Embedding Dimensionality Reduction

詞向量矩陣 $\mathbf{E} \in \mathbb{R}^{|V| \times d}$ 可以用 SVD 降維到 $k \ll d$：

$$\mathbf{E}_k = \mathbf{U}_k \mathbf{\Sigma}_k$$

### 4.3 協同過濾 | Collaborative Filtering (推薦系統)

用户-物品評分矩陣 $\mathbf{R} \in \mathbb{R}^{m \times n}$ 的 SVD 補全：

$$\hat{\mathbf{R}} = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^\top$$

---

## 5. 矩陣微積分 | Matrix Calculus

### 5.1 基本公式 | Key Identities

在深度學習中極為重要的矩陣微分恆等式：

| 運算 | 結果 |
|------|------|
| $\frac{\partial}{\partial \mathbf{x}}(\mathbf{b}^\top \mathbf{x})$ | $\mathbf{b}$ |
| $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^\top \mathbf{x})$ | $2\mathbf{x}$ |
| $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^\top \mathbf{A} \mathbf{x})$ | $(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ |
| $\frac{\partial}{\partial \mathbf{X}}(\text{tr}(\mathbf{A}\mathbf{X}))$ | $\mathbf{A}^\top$ |
| $\frac{\partial}{\partial \mathbf{X}}(\log\det(\mathbf{X}))$ | $(\mathbf{X}^{-1})^\top$ |

### 5.2 梯度與反向傳播 | Gradients & Backpropagation

神經網路的損失函數梯度：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \mathbf{\delta}^{(l)} \cdot (\mathbf{a}^{(l-1)})^\top$$

其中 $\mathbf{\delta}^{(l)} = (\mathbf{W}^{(l+1)})^\top \mathbf{\delta}^{(l+1)} \odot \sigma'(\mathbf{z}^{(l)})$ 是誤差信號。

```python
import numpy as np

def matrix_gradient_example():
    """展示矩陣微積分在線性回歸中的應用"""
    
    # 線性回歸：y = Xw + b
    # 損失：L = (1/2N) ||y - Xw||^2
    # 梯度：dL/dw = X^T(y - Xw) / N
    
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    true_w = np.random.randn(10, 1)
    y = X @ true_w + 0.1 * np.random.randn(1000, 1)
    
    w = np.zeros((10, 1))
    lr = 0.01
    
    for epoch in range(1000):
        pred = X @ w
        error = pred - y
        grad = X.T @ error / len(y)
        w = w - lr * grad
        
        if epoch % 200 == 0:
            loss = 0.5 * np.mean(error ** 2)
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")
    
    return w

w_learned = matrix_gradient_example()
print(f"\nTrue weights:     {true_w.flatten()[:5]}...")
print(f"Learned weights:  {w_learned.flatten()[:5]}...")
```

---

## 6. 總結速查表 | Quick Reference

```
┌─────────────────────────────────────────────────────────────┐
│  深度學習中的線性代數速查                                   │
├──────────────────────┬──────────────────────────────────────┤
│  概念                │  應用場景                             │
├──────────────────────┼──────────────────────────────────────┤
│  向量點積             │  注意力機制 (Q·K^T)                   │
│  矩陣乘法             │  全連接層 (Y = XW + B)               │
│  Softmax             │  注意力權重標準化                     │
│  特徵值 / SVD         │  PCA、詞嵌入降維、網路穩定性分析      │
│  奇異值               │  條件數 (神經網路收斂性)               │
│  正交矩陣             │  Layer Normalization (歸一化)         │
│  範數                 │  損失函數 (L1/L2)、梯度裁剪           │
│  行列式               │  機率密度函數中的Jacobian             │
│  偽逆 (Moore-Penrose) │  最小二乘解、神經網路正交初始化       │
└──────────────────────┴──────────────────────────────────────┘
```

---

## 參考文獻 | References

1. Strang, G. (2023). *Linear Algebra for Everyone*. Wellesley-Cambridge Press.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Ch. 2. MIT Press.
3. Petersen, K. B., & Pedersen, M. S. (2008). *The Matrix Cookbook*. Technical University of Denmark.

---

*© 2026 chanxaviersy*
