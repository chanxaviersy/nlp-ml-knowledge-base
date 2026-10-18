# 🎯 推薦系統 | Recommender Systems

> 摘要：推薦系統預測用戶對物品的偏好，從海量資訊中提供個性化推薦。本章介紹協同過濾、矩陣分解、深度學習推薦模型、序列推薦以及 LLM 時代的推薦系統。  
> Abstract: Recommender systems predict user preferences to provide personalized recommendations. This chapter covers collaborative filtering, matrix factorization, deep learning, sequential recommenders, and LLM-era systems.

---

## 1. 推薦系統類型 | Types of Recommender Systems

### 1.1 三大範式

```
┌──────────────────────────────────────────────────────────────┐
│  推薦系統三劍客                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │  協同過濾         │  │  基於內容         │                 │
│  │  Collaborative    │  │  Content-based   │                 │
│  │  Filtering        │  │                  │                 │
│  │  ── "相似用戶"    │  │  ── "相似物品"    │                 │
│  │  ── 行為數據      │  │  ── 特徵匹配      │                 │
│  └──────────────────┘  └──────────────────┘                 │
│           │                       │                          │
│           └───────────┬───────────┘                          │
│                       ↓                                      │
│            ┌──────────────────┐                              │
│            │  混合方法         │                              │
│            │  Hybrid          │                              │
│            │  ── 加權、級聯    │                              │
│            │  ── 特徵組合      │                              │
│            └──────────────────┘                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 關鍵問題

| 問題 | 描述 | 指標 |
|------|------|------|
| **冷啟動** | 新用戶/新物品無歷史數據 | 啟動率 |
| **稀疏性** | 用戶-物品互動矩陣極度稀疏 | 稀疏度 |
| **長尾** | 熱門少，冷門多 | 多樣性 |
| **可解釋性** | 為什麼推薦這個 | 解釋準確度 |
| **多樣性** | 不要總推薦同類 | 多樣性指標 |
| **驚喜度** | 推薦意外但滿意 | Serendipity |

---

## 2. 數據形式 | Data Representation

### 2.1 用戶-物品互動矩陣

$$R = \begin{bmatrix} r_{1,1} & r_{1,2} & \cdots & r_{1,n} \\ r_{2,1} & r_{2,2} & \cdots & r_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ r_{m,1} & r_{m,2} & \cdots & r_{m,n} \end{bmatrix}$$

其中 $r_{u,i}$ 是用戶 $u$ 對物品 $i$ 的評分或隱式反饋。

### 2.2 反饋類型

| 類型 | 例子 | 特點 |
|------|------|------|
| **顯式反饋** | 評分（1-5星）、點讚/踩 | 準確但稀疏 |
| **隱式反饋** | 點擊、瀏覽、停留時間 | 豐富但噪聲 |
| **多行為** | 點擊+加購+購買 | 反映真實興趣 |

---

## 3. 協同過濾 | Collaborative Filtering

### 3.1 基於用戶的協同過濾

**核心思想**：相似的用戶有相似的偏好。

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserBasedCF:
    """基於用戶的協同過濾"""
    
    def __init__(self, ratings_matrix):
        self.R = ratings_matrix  # (n_users, n_items)
        self.user_similarity = None
    
    def compute_similarity(self, method='cosine'):
        """計算用戶相似度"""
        if method == 'cosine':
            # 處理 0 值（無評分）
            self.user_similarity = cosine_similarity(self.R)
        elif method == 'pearson':
            # Pearson 相關
            self.user_similarity = np.corrcoef(self.R)
        elif method == 'adjusted_cosine':
            # 調整餘弦（減去用戶均值）
            user_means = self.R[self.R > 0].mean()  # 全局均值
            R_centered = self.R - user_means
            R_centered[R_centered < 0] = 0
            self.user_similarity = cosine_similarity(R_centered)
        return self.user_similarity
    
    def predict(self, user_idx, item_idx, k=10):
        """預測用戶 u 對物品 i 的評分"""
        # 找到評分過物品 i 的用戶
        rated_users = np.where(self.R[:, item_idx] > 0)[0]
        if len(rated_users) == 0:
            return self.R[self.R > 0].mean()
        
        # 計算相似度
        sims = self.user_similarity[user_idx, rated_users]
        
        # 取 top-k
        top_k_idx = np.argsort(sims)[-k:]
        top_k_users = rated_users[top_k_idx]
        top_k_sims = sims[top_k_idx]
        
        # 加權預測
        top_k_ratings = self.R[top_k_users, item_idx]
        if np.sum(np.abs(top_k_sims)) == 0:
            return self.R[self.R > 0].mean()
        
        return np.dot(top_k_sims, top_k_ratings) / np.sum(np.abs(top_k_sims))
    
    def recommend(self, user_idx, n_recommendations=10):
        """為用戶推薦 top-N 物品"""
        n_items = self.R.shape[1]
        predictions = np.array([
            self.predict(user_idx, i) for i in range(n_items)
        ])
        return np.argsort(predictions)[::-1][:n_recommendations]
```

### 3.2 基於物品的協同過濾

**核心思想**：用戶喜歡相似的物品。

```python
class ItemBasedCF:
    """基於物品的協同過濾"""
    
    def __init__(self, ratings_matrix):
        self.R = ratings_matrix
        self.item_similarity = None
    
    def compute_similarity(self, shrinkage=100):
        """計算物品相似度（帶 shrinkage）"""
        n_items = self.R.shape[1]
        self.item_similarity = np.zeros((n_items, n_items))
        
        for i in range(n_items):
            for j in range(n_items):
                if i == j:
                    self.item_similarity[i, j] = 1
                    continue
                
                # 共同評分的用戶
                common = (self.R[:, i] > 0) & (self.R[:, j] > 0)
                n_common = common.sum()
                
                if n_common < 3:
                    self.item_similarity[i, j] = 0
                    continue
                
                # 餘弦相似度 + shrinkage
                vi = self.R[common, i]
                vj = self.R[common, j]
                sim = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-9)
                # Shrinkage：共同評分用戶少時降低相似度
                sim_shrunk = sim * n_common / (n_common + shrinkage)
                self.item_similarity[i, j] = sim_shrunk
    
    def predict(self, user_idx, item_idx, k=20):
        # 該用戶評分過的物品
        rated_items = np.where(self.R[user_idx] > 0)[0]
        if len(rated_items) == 0:
            return self.R[self.R > 0].mean()
        
        # 物品相似度
        sims = self.item_similarity[item_idx, rated_items]
        top_k_idx = np.argsort(sims)[-k:]
        top_k_items = rated_items[top_k_idx]
        top_k_sims = sims[top_k_idx]
        top_k_ratings = self.R[user_idx, top_k_items]
        
        return np.dot(top_k_sims, top_k_ratings) / np.sum(np.abs(top_k_sims) + 1e-9)
```

### 3.3 矩陣分解 | Matrix Factorization

**核心思想**：將 $R$ 分解為用戶向量 $P$ 和物品向量 $Q$：

$$R \approx P \cdot Q^\top$$

其中 $P \in \mathbb{R}^{m \times k}$，$Q \in \mathbb{R}^{n \times k}$，$k$ 是隱向量維度。

```python
class MatrixFactorization:
    """矩陣分解（帶偏置項）"""
    
    def __init__(self, n_users, n_items, n_factors=20, lr=0.01, 
                 reg=0.02, n_epochs=50):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        
        # 參數
        self.P = np.random.normal(0, 0.1, (n_users, n_factors))  # 用戶向量
        self.Q = np.random.normal(0, 0.1, (n_items, n_factors))  # 物品向量
        self.bu = np.zeros(n_users)  # 用戶偏置
        self.bi = np.zeros(n_items)  # 物品偏置
        self.mu = 0  # 全局均值
    
    def fit(self, R, mask):
        """訓練"""
        self.mu = R[mask].mean()
        for epoch in range(self.n_epochs):
            for u in range(self.n_users):
                for i in range(self.n_items):
                    if mask[u, i] == 0:
                        continue
                    
                    # 預測
                    pred = self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                    err = R[u, i] - pred
                    
                    # 梯度更新
                    self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                    self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                    self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                    self.Q[i] += self.lr * (err * self.P[u] - self.reg * self.Q[i])
    
    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
```

### 3.4 SVD 與 FunkSVD

```python
# FunkSVD（隱式反饋，不分解為三個矩陣）
def funk_svd(R, mask, n_factors=20, n_epochs=50, lr=0.01, reg=0.02):
    """
    Funk SVD：只學習 P 和 Q
    """
    n_users, n_items = R.shape
    P = np.random.normal(0, 0.1, (n_users, n_factors))
    Q = np.random.normal(0, 0.1, (n_items, n_factors))
    
    for epoch in range(n_epochs):
        for u in range(n_users):
            for i in range(n_items):
                if mask[u, i] == 0:
                    continue
                pred = P[u] @ Q[i]
                err = R[u, i] - pred
                P[u] += lr * (err * Q[i] - reg * P[u])
                Q[i] += lr * (err * P[u] - reg * Q[i])
    
    return P, Q
```

---

## 4. 進階矩陣分解 | Advanced MF

### 4.1 SVD++

加入隱式反饋（瀏覽、收藏等）：

```python
class SVDpp:
    """SVD++ 帶隱式反饋"""
    
    def __init__(self, n_users, n_items, n_factors=20):
        self.P = np.random.normal(0, 0.1, (n_users, n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, n_factors))
        self.y = np.random.normal(0, 0.1, (n_items, n_factors))  # 隱式反饋
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
    
    def fit(self, R, mask, implicit_feedback):
        """implicit_feedback[u] = 用戶 u 評分過的物品集合"""
        mu = R[mask].mean()
        for epoch in range(50):
            for u in range(self.n_users):
                N_u = implicit_feedback[u]  # 用戶 u 評分過的物品
                # 隱式反饋貢獻
                y_sum = self.y[N_u].sum(axis=0) / np.sqrt(len(N_u)) if len(N_u) > 0 else 0
                
                for i in range(self.n_items):
                    if mask[u, i] == 0:
                        continue
                    pred = mu + self.bu[u] + self.bi[i] + (self.P[u] + y_sum) @ self.Q[i]
                    err = R[u, i] - pred
                    # 更新參數...
```

### 4.2 ALS (交替最小二乘)

適合大規模稀疏矩陣：

```python
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# pip install implicit
model = AlternatingLeastSquares(
    factors=64,
    regularization=0.01,
    iterations=15,
    use_gpu=False,
)

# 訓練（隱式反饋）
model.fit(user_items_sparse_matrix)

# 推薦
user_id = 0
recommended_items, scores = model.recommend(user_id, user_items_sparse_matrix[user_id], N=10)
```

---

## 5. 深度學習推薦 | Deep Learning

### 5.1 Neural Collaborative Filtering (NCF)

```python
import torch
import torch.nn as nn

class NCF(nn.Module):
    """Neural Collaborative Filtering"""
    
    def __init__(self, n_users, n_items, mf_dim=8, mlp_dim=32, dropout=0.2):
        super().__init__()
        # 矩陣分解部分
        self.user_mf = nn.Embedding(n_users, mf_dim)
        self.item_mf = nn.Embedding(n_items, mf_dim)
        
        # MLP 部分
        self.user_mlp = nn.Embedding(n_users, mlp_dim)
        self.item_mlp = nn.Embedding(n_items, mlp_dim)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        
        # 最終預測
        self.predict = nn.Linear(mf_dim + 16, 1)
    
    def forward(self, user, item):
        # MF 部分
        mf = self.user_mf(user) * self.item_mf(item)
        
        # MLP 部分
        mlp_input = torch.cat([self.user_mlp(user), self.item_mlp(item)], dim=-1)
        mlp_out = self.mlp(mlp_input)
        
        # 拼接
        x = torch.cat([mf, mlp_out], dim=-1)
        return torch.sigmoid(self.predict(x).squeeze(-1))
```

### 5.2 Wide & Deep (Google)

結合記憶（Wide）和泛化（Deep）：

```python
class WideDeep(nn.Module):
    """Wide & Deep 模型"""
    
    def __init__(self, n_users, n_items, n_features, mlp_dims=[256, 128, 64]):
        super().__init__()
        # Wide 部分：線性模型
        self.wide = nn.Linear(n_features, 1)
        
        # Deep 部分：深度網路
        self.user_emb = nn.Embedding(n_users, 32)
        self.item_emb = nn.Embedding(n_items, 32)
        layers = []
        input_dim = 64 + n_features - 2  # 嵌入 + 其他特徵
        for dim in mlp_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            input_dim = dim
        self.deep = nn.Sequential(*layers)
        self.deep_final = nn.Linear(mlp_dims[-1], 1)
    
    def forward(self, user, item, features):
        # Wide
        wide_out = self.wide(features).squeeze(-1)
        
        # Deep
        user_vec = self.user_emb(user)
        item_vec = self.item_emb(item)
        deep_input = torch.cat([user_vec, item_vec, features], dim=-1)
        deep_out = self.deep_final(self.deep(deep_input)).squeeze(-1)
        
        return wide_out + deep_out
```

### 5.3 DeepFM

**FM (Factorization Machines)** 自動學習特徵交互，與 Deep 結合：

```python
class DeepFM(nn.Module):
    """DeepFM: FM + Deep"""
    
    def __init__(self, n_features, n_factors=10, mlp_dims=[256, 128]):
        super().__init__()
        self.n_features = n_features
        self.n_factors = n_factors
        
        # FM 一階
        self.linear = nn.Linear(n_features, 1)
        
        # FM 二階
        self.V = nn.Parameter(torch.randn(n_features, n_factors) * 0.01)
        
        # Deep
        layers = []
        input_dim = n_features
        for dim in mlp_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            input_dim = dim
        layers.append(nn.Linear(mlp_dims[-1], 1))
        self.deep = nn.Sequential(*layers)
    
    def forward(self, x):
        # FM 一階
        linear_part = self.linear(x).squeeze(-1)
        
        # FM 二階：$(1/2) \sum ((V^T x)^2 - V^T x^2)$
        interactions = 0.5 * torch.sum(
            torch.pow(x @ self.V, 2) - (x ** 2) @ (self.V ** 2),
            dim=-1
        )
        
        # Deep
        deep_part = self.deep(x).squeeze(-1)
        
        # Sigmoid
        return torch.sigmoid(linear_part + interactions + deep_part)
```

### 5.4 兩塔模型 (Two-Tower)

```python
class TwoTower(nn.Module):
    """雙塔模型：User Tower + Item Tower"""
    
    def __init__(self, user_feat_dim, item_feat_dim, embed_dim=64):
        super().__init__()
        # 用戶塔
        self.user_tower = nn.Sequential(
            nn.Linear(user_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
        # 物品塔
        self.item_tower = nn.Sequential(
            nn.Linear(item_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, user_features, item_features):
        u = self.user_tower(user_features)
        i = self.item_tower(item_features)
        # 餘弦相似度
        u = nn.functional.normalize(u, dim=-1)
        i = nn.functional.normalize(i, dim=-1)
        logits = (u @ i.T) / self.temperature
        return logits
    
    def encode_user(self, user_features):
        return nn.functional.normalize(self.user_tower(user_features), dim=-1)
    
    def encode_item(self, item_features):
        return nn.functional.normalize(self.item_tower(item_features), dim=-1)
```

### 5.5 DIN (Deep Interest Network)

注意力機制捕捉用戶歷史興趣：

```python
class DINAttention(nn.Module):
    """DIN 局部激活單元"""
    
    def __init__(self, embed_dim, mlp_dims=[256, 128, 64]):
        super().__init__()
        # 輸入：候選物品 + 歷史物品 + 外積
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, mlp_dims[0]),
            nn.PReLU(),
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.PReLU(),
            nn.Linear(mlp_dims[1], 1),
        )
    
    def forward(self, candidate, history, mask=None):
        """
        candidate: (batch, embed_dim)
        history: (batch, seq_len, embed_dim)
        mask: (batch, seq_len) - 1 為有效，0 為填充
        """
        seq_len = history.size(1)
        # 擴展 candidate
        candidate = candidate.unsqueeze(1).expand(-1, seq_len, -1)
        # 外積
        outer = candidate * history
        # 拼接所有特徵
        attn_input = torch.cat([candidate, history, candidate - history, outer], dim=-1)
        # 計算注意力分數
        scores = self.mlp(attn_input).squeeze(-1)  # (batch, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        # 加權求和
        output = (weights.unsqueeze(-1) * history).sum(dim=1)
        return output
```

---

## 6. 序列推薦 | Sequential Recommendation

### 6.1 SASRec (Self-Attentive Sequential Recommendation)

```python
class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation"""
    
    def __init__(self, n_items, max_seq_len=50, embed_dim=64, 
                 n_heads=2, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # 嵌入
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer 編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, 
            dim_feedforward=256, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Layer Norm
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, seq):
        """
        seq: (batch, seq_len) - 物品 ID 序列
        """
        batch_size, seq_len = seq.shape
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        
        # 嵌入
        item_emb = self.item_emb(seq)
        pos_emb = self.pos_emb(positions)
        x = item_emb + pos_emb
        x = self.layer_norm(x)
        
        # 因果遮罩
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()
        
        # Transformer 編碼
        x = self.encoder(x, mask=causal_mask)
        return x
    
    def predict_next(self, seq, candidate_items):
        """預測下一個物品"""
        x = self.forward(seq)  # (batch, seq_len, embed_dim)
        last_hidden = x[:, -1, :]  # 取最後位置
        candidate_emb = self.item_emb(candidate_items)  # (batch, n_candidates, embed_dim)
        scores = (last_hidden.unsqueeze(1) * candidate_emb).sum(dim=-1)
        return scores
```

### 6.2 BERT4Rec

```python
class BERT4Rec(nn.Module):
    """BERT 風格的序列推薦"""
    
    def __init__(self, n_items, max_seq_len=50, embed_dim=64, 
                 n_heads=2, n_layers=2, mask_prob=0.15):
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.mask_token = n_items  # [MASK] token
        
        # 嵌入
        self.item_emb = nn.Embedding(n_items + 2, embed_dim, padding_idx=0)  # +1 for [MASK]
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=256, dropout=0.2,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, seq, labels=None):
        """訓練：隨機遮蓋 + 預測"""
        if self.training and labels is not None:
            # 隨機遮蓋
            mask = torch.bernoulli(torch.full(seq.shape, self.mask_prob)).bool()
            mask &= (seq != 0)  # 不遮蓋 padding
            seq_masked = seq.clone()
            seq_masked[mask] = self.mask_token
        else:
            seq_masked = seq
            mask = None
        
        # ... 編碼和預測
        return predictions
```

### 6.3 其他序列推薦

| 模型 | 架構 | 特色 |
|------|------|------|
| **GRU4Rec** | GRU | 早期 RNN 序列推薦 |
| **Caser** | CNN | 水平和垂直卷積 |
| **SASRec** | Transformer | 自回歸 |
| **BERT4Rec** | BERT | Cloze 預訓練 |
| **TiSASRec** | Transformer | 時間感知 |
| **LightSANs** | Transformer | 輕量化 |

---

## 7. 大語言模型推薦 | LLM-based Recommendation

### 7.1 三大範式

```python
# 1. LLM 作為特徵工程
class LLMFeatureEncoder:
    def encode_item_description(self, text):
        """用 LLM 編碼物品描述"""
        # 構建 prompt
        prompt = f"為以下物品生成結構化特徵：\n{text}\n\n特徵："
        features = llm.invoke(prompt)
        return parse_features(features)

# 2. LLM 作為零樣本推薦器
def zero_shot_recommend(user_history, candidate_items):
    prompt = f"""基於用戶的歷史行為，推薦最相關的物品。

用戶歷史：{user_history}
候選物品：{candidate_items}

推薦前 3 個最相關的物品："""
    return llm.invoke(prompt)

# 3. LLM 微調推薦
class RecLLM(nn.Module):
    def __init__(self, base_llm, n_items):
        super().__init__()
        self.llm = base_llm
        # 物品 ID 嵌入
        self.item_emb = nn.Embedding(n_items, base_llm.config.hidden_size)
        # 融合層
        self.fusion = nn.Linear(base_llm.config.hidden_size * 2, 
                                 base_llm.config.hidden_size)
```

### 7.2 主流 LLM 推薦模型

| 模型 | 特點 |
|------|------|
| **P5** | 統一 prompt 的多任務預訓練 |
| **M6-Rec** | 阿里多模態推薦 |
| **ChatRec** | 對話式推薦 |
| **InstructRec** | 指令微調推薦 |
| **GPT4Rec** | 生成式推薦 |
| **TallRec** | Lora 微調推薦 |

---

## 8. 評估指標 | Evaluation Metrics

### 8.1 預測準確度

| 指標 | 公式 | 適用 |
|------|------|------|
| **RMSE** | $\sqrt{\frac{1}{N}\sum (r_{ui} - \hat{r}_{ui})^2}$ | 顯式反饋評分預測 |
| **MAE** | $\frac{1}{N}\sum \|r_{ui} - \hat{r}_{ui}\|$ | 顯式反饋 |

### 8.2 Top-N 推薦

```python
def precision_at_k(recommended, relevant, k):
    """P@K"""
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(relevant)) / k

def recall_at_k(recommended, relevant, k):
    """R@K"""
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(relevant)) / len(relevant) if len(relevant) > 0 else 0

def ndcg_at_k(recommended, relevant, k):
    """NDCG@K"""
    recommended_k = recommended[:k]
    dcg = sum([1 / np.log2(i + 2) for i, r in enumerate(recommended_k) if r in relevant])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(relevant)))])
    return dcg / idcg if idcg > 0 else 0

def hit_rate(recommended, relevant):
    """HR"""
    return 1 if any(r in relevant for r in recommended) else 0

def map_at_k(recommended, relevant, k):
    """MAP@K (Mean Average Precision)"""
    recommended_k = recommended[:k]
    score = 0
    hits = 0
    for i, r in enumerate(recommended_k):
        if r in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(k, len(relevant)) if len(relevant) > 0 else 0
```

### 8.3 多樣性與新穎度

```python
def diversity(recommended, item_similarity):
    """推薦列表的內部多樣性"""
    n = len(recommended)
    if n <= 1:
        return 0
    diversity_sum = 0
    for i in range(n):
        for j in range(i + 1, n):
            diversity_sum += (1 - item_similarity[recommended[i], recommended[j]])
    return 2 * diversity_sum / (n * (n - 1))

def coverage(all_recommendations, n_items):
    """目錄覆蓋率"""
    unique_recommended = set()
    for rec_list in all_recommendations:
        unique_recommended.update(rec_list)
    return len(unique_recommended) / n_items
```

### 8.4 線上 A/B 測試指標

| 指標 | 描述 |
|------|------|
| **CTR** | 點擊率 |
| **CVR** | 轉化率 |
| **GMV** | 商品交易總額 |
| **Engagement** | 用戶參與時長 |
| **Retention** | 用戶留存 |

---

## 9. 工業實踐 | Industry Practice

### 9.1 多階段架構

```
候選生成 (Candidate Generation)    海量物品 → 數百候選
   ↓
粗排 (Pre-Ranking)               數百 → 數十
   ↓
精排 (Ranking)                   數十 → top K
   ↓
重排 (Re-ranking)                多樣性、業務規則
```

### 9.2 特徵工程

| 特徵類型 | 例子 |
|----------|------|
| 用戶特徵 | 年齡、性別、興趣標籤 |
| 物品特徵 | 類別、價格、銷量 |
| 上下文 | 時間、地點、設備 |
| 行為特徵 | 點擊序列、瀏覽時長 |
| 統計特徵 | 用戶活躍度、物品熱度 |

### 9.3 冷啟動解決方案

| 場景 | 方案 |
|------|------|
| **新用戶** | 問卷引導、社交信息、流行推薦 |
| **新物品** | 基於內容特徵、屬性匹配 |
| **新領域** | 跨域遷移、元學習 |

### 9.4 開源工具

| 工具 | 描述 |
|------|------|
| **Surprise** | Python 經典推薦庫 |
| **Implicit** | 隱式反饋 ALS |
| **LightFM** | 混合推薦 |
| **RecBole** | 統一推薦框架 |
| **Microsoft Recommenders** | 微軟推薦庫 |
| **TorchRec** | PyTorch 工業級 |

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│  推薦系統發展路線                                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  經典協同過濾 (1990s-2000s)                                   │
│  ├── User-based CF                                          │
│  ├── Item-based CF                                          │
│  └── Matrix Factorization (SVD)                             │
│                                                              │
│  深度學習 (2015+)                                              │
│  ├── NCF, DeepFM, Wide&Deep                                 │
│  ├── DIN (注意力)                                            │
│  ├── 兩塔模型 (檢索)                                          │
│  └── 多任務學習 (ESMM)                                        │
│                                                              │
│  序列推薦 (2018+)                                              │
│  ├── GRU4Rec, Caser                                         │
│  ├── SASRec, BERT4Rec                                       │
│  └── 強化學習推薦                                           │
│                                                              │
│  LLM 時代 (2023+)                                             │
│  ├── LLM 作為特徵                                            │
│  ├── Prompt 推薦                                             │
│  ├── 微調 LLM 做推薦                                        │
│  └── 對話式推薦                                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

| 場景 | 推薦算法 |
|------|----------|
| 評分預測 | SVD++, NMF |
| 隱式反饋 (點擊) | ALS, BPR, Two-Tower |
| 序列推薦 | SASRec, BERT4Rec |
| 多目標 | ESMM, MMOE |
| 冷啟動 | 基於內容 + 跨域 |
| 大模型時代 | LLM 微調 + 傳統結合 |

---

## 參考文獻 | References

1. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer*.
2. He, X., et al. (2017). Neural Collaborative Filtering. *WWW*.
3. Guo, H., et al. (2017). DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. *IJCAI*.
4. Kang, W. C., & McAuley, J. (2018). Self-Attentive Sequential Recommendation. *ICDM* (SASRec).
5. Sun, F., et al. (2019). BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformers. *CIKM*.
6. Zhou, G., et al. (2018). Deep Interest Network for Click-Through Rate Prediction. *KDD* (DIN).

---

*© 2026 chanxaviersy*
