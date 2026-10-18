# 🔄 無監督學習 | Unsupervised Learning

> 摘要：無監督學習從無標籤資料中發現隱藏結構。本章介紹聚類算法（K-Means、層次聚類、DBSCAN）、降維方法（PCA、t-SNE、UMAP）以及關聯規則學習。  
> Abstract: Unsupervised learning discovers hidden structures from unlabeled data. This chapter covers clustering (K-Means, Hierarchical, DBSCAN), dimensionality reduction (PCA, t-SNE, UMAP), and association rules.

---

## 1. 聚類 | Clustering

### 1.1 K-Means

**目標**：將資料分為 K 個簇，使簇內方差最小：

$$\min \sum_{k=1}^{K} \sum_{x \in C_k} \|x - \mu_k\|^2$$

**算法**：
1. 隨機初始化 K 個中心點
2. 分配：每個點歸到最近的中心
3. 更新：重新計算每個簇的中心
4. 重複 2-3 直到收斂

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成資料
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# K-Means 聚類
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
y_pred = kmeans.fit_predict(X)

# 可視化
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=30)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label='Centroids')
plt.legend()
plt.title("K-Means Clustering")
plt.show()
```

**選擇 K 值：肘部法則 + 輪廓係數**

```python
from sklearn.metrics import silhouette_score

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    y_pred = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, y_pred))

# 繪製
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(K_range, inertias, 'bo-')
ax[0].set_xlabel('K'); ax[0].set_ylabel('Inertia')
ax[0].set_title('Elbow Method')
ax[1].plot(K_range, silhouettes, 'go-')
ax[1].set_xlabel('K'); ax[1].set_ylabel('Silhouette Score')
ax[1].set_title('Silhouette Analysis')
plt.show()
```

### 1.2 層次聚類 | Hierarchical Clustering

構建嵌套簇的樹狀結構（dendrogram）：

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# 凝聚式層次聚類
Z = linkage(X, method='ward')

# 樹狀圖
plt.figure(figsize=(10, 5))
dendrogram(Z, no_labels=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

# 切割成 4 個簇
clusters = fcluster(Z, t=4, criterion='maxclust')
```

**連結方法**：
- **Single linkage**：$d(C_1, C_2) = \min_{x \in C_1, y \in C_2} d(x, y)$ — 容易拉鏈效應
- **Complete linkage**：$d(C_1, C_2) = \max_{x \in C_1, y \in C_2} d(x, y)$
- **Average linkage**：平均值
- **Ward's method**：最小化簇內方差

### 1.3 DBSCAN：基於密度的聚類

**核心思想**：高密度區域形成簇，低密度區域為噪聲。

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
y_pred = dbscan.fit_predict(X)

n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
n_noise = (y_pred == -1).sum()
print(f"Clusters: {n_clusters}, Noise points: {n_noise}")
```

**優點**：
- 不需要指定 K
- 能發現任意形狀的簇
- 能識別噪聲點

**缺點**：
- 對 $\epsilon$ 和 `min_samples` 敏感
- 高維資料效果差

### 1.4 其他聚類算法

| 算法 | 特點 |
|------|------|
| **Gaussian Mixture (GMM)** | 概率模型，軟分配 |
| **Mean Shift** | 自動確定 K，計算慢 |
| **Spectral Clustering** | 基於圖論，適合非凸 |
| **Affinity Propagation** | 不需 K，自動選擇範例 |

---

## 2. 降維 | Dimensionality Reduction

### 2.1 主成分分析 (PCA)

**目標**：找到資料方差最大的正交方向（主成分）。

**算法**：
1. 中心化資料
2. 計算協方差矩陣 $\Sigma = \frac{1}{N} X^\top X$
3. 特徵值分解 $\Sigma = V \Lambda V^\top$
4. 取前 $k$ 個特徵向量作為投影方向

$$\mathbf{X}_{reduced} = \mathbf{X} V_k$$

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

# PCA
pca = PCA(n_components=0.95)  # 保留 95% 方差
X_reduced = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
```

### 2.2 t-SNE：視覺化首選

t-SNE 通過保持局部結構來可視化高維資料：

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=digits.target, cmap='tab10', s=10)
plt.colorbar(scatter)
plt.title("t-SNE Visualization of Digits")
plt.show()
```

**注意**：
- ✅ 用於**視覺化**（2D/3D）
- ❌ 不應該用於特徵工程或聚類輸入
- 對 `perplexity` 參數敏感

### 2.3 UMAP：更快更穩定

```python
# pip install umap-learn
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_2d = reducer.fit_transform(X)
```

**UMAP vs t-SNE**：
- ✅ UMAP 訓練快、可重複、保留全局結構
- ❌ t-SNE 視覺化效果更平滑

### 2.4 降維方法對比

| 方法 | 類型 | 保留 | 用途 |
|------|------|------|------|
| **PCA** | 線性 | 全局方差 | 特徵工程、壓縮 |
| **Kernel PCA** | 非線性 | 全局 | 非線性資料 |
| **LDA** | 線性 | 類別可分性 | 有監督降維 |
| **t-SNE** | 非線性 | 局部 | 視覺化 |
| **UMAP** | 非線性 | 局部+部分全局 | 視覺化、特徵 |
| **Autoencoder** | 非線性 | 通過重建 | 深度特徵 |

---

## 3. 關聯規則學習 | Association Rules

### 3.1 Apriori 算法

**經典應用**：購物籃分析

**支持度 (Support)**：$P(A \cap B)$
**置信度 (Confidence)**：$P(B | A) = P(A \cap B) / P(A)$
**提升度 (Lift)**：$P(B | A) / P(B)$

```python
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# 示例：交易資料
transactions = [
    ['牛奶', '麵包', '雞蛋'],
    ['牛奶', '麵包'],
    ['牛奶', '雞蛋'],
    ['麵包', '雞蛋', '啤酒'],
    ['牛奶', '麵包', '雞蛋', '啤酒'],
]

# One-hot 編碼
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# 挖掘頻繁項集
freq_items = apriori(df, min_support=0.4, use_colnames=True)
print(freq_items)

# 關聯規則
rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

### 3.2 FP-Growth

比 Apriori 更高效的算法：

```python
from mlxtend.frequent_patterns import fpgrowth
freq_items = fpgrowth(df, min_support=0.4, use_colnames=True)
```

---

## 4. 異常檢測 | Anomaly Detection

| 方法 | 描述 | 適用 |
|------|------|------|
| **統計方法** | 假設常態分佈，超過 3σ 視為異常 | 單變量、低維 |
| **Isolation Forest** | 隨機切分，異常易被隔離 | 高維、通用 |
| **One-Class SVM** | 學習正常資料的邊界 | 中等維度 |
| **Local Outlier Factor (LOF)** | 基於局部密度的偏差 | 不均勻密度 |
| **Autoencoder** | 重建誤差大的視為異常 | 複雜模式 |

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.1, random_state=42)
y_pred = iso.fit_predict(X)
# -1: 異常, 1: 正常
```

---

## 總結 | Summary

| 任務 | 算法 | 適用 |
|------|------|------|
| **聚類** | K-Means | 球形簇、大資料 |
| | DBSCAN | 任意形狀、識別噪聲 |
| | 層次 | 小資料、需要層次結構 |
| | GMM | 概率聚類、橢圓形 |
| **降維** | PCA | 線性、特徵工程 |
| | t-SNE | 2D/3D 視覺化 |
| | UMAP | 視覺化 + 保留全局 |
| | Autoencoder | 複雜非線性 |
| **關聯** | Apriori | 經典購物籃 |
| | FP-Growth | 大資料 |
| **異常** | Isolation Forest | 高維 |
| | LOF | 局部異常 |

---

## 參考文獻 | References

1. MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations.
2. Ester, M., et al. (1996). A Density-Based Algorithm for Discovering Clusters. *KDD*.
3. van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*.
4. McInnes, L., et al. (2018). UMAP: Uniform Manifold Approximation and Projection.

---

*© 2026 chanxaviersy*
