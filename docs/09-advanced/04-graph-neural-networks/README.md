# 🕸️ 圖神經網路 | Graph Neural Networks (GNN)

> 摘要：圖神經網路處理圖結構數據，學習節點、邊和整圖的表示。本章介紹圖基礎、GNN 主流架構 (GCN、GAT、GraphSAGE)、圖表示學習以及在推薦、分子、社交網路等領域的應用。  
> Abstract: Graph Neural Networks process graph-structured data, learning representations for nodes, edges, and entire graphs. This chapter covers graph basics, GNN architectures, representation learning, and applications.

---

## 1. 圖基礎 | Graph Basics

### 1.1 圖的數學表示

**無向圖** $G = (V, E)$：
- $V$：節點集合
- $E$：邊集合

**鄰接矩陣 (Adjacency Matrix)** $A \in \mathbb{R}^{N \times N}$：

$$A_{ij} = \begin{cases} 1 & \text{if } (i, j) \in E \\ 0 & \text{otherwise} \end{cases}$$

**度矩陣 (Degree Matrix)** $D$：

$$D_{ii} = \sum_j A_{ij}$$

**特徵矩陣** $X \in \mathbb{R}^{N \times d}$：每個節點 $d$ 維特徵。

### 1.2 圖的類型

| 類型 | 描述 | 例子 |
|------|------|------|
| **同質圖** | 一種節點、一種邊 | 引文網路 |
| **異質圖** | 多種節點/邊 | 學術圖（作者-論文-領域）|
| **有向圖** | 邊有方向 | 引用、網頁鏈接 |
| **時空圖** | 圖隨時間變化 | 交通網路 |
| **知識圖譜** | 實體-關係-實體 | Wikidata |
| **動態圖** | 結構隨時間演變 | 社交網路 |

### 1.3 為什麼需要 GNN？

傳統深度學習處理的是**規則數據**（圖像、序列、表格），但現實中很多數據是**不規則的圖結構**：

- 社交網路（用戶 = 節點，關注 = 邊）
- 分子（原子 = 節點，化學鍵 = 邊）
- 知識圖譜（實體 = 節點，關係 = 邊）
- 交通網路（路口 = 節點，道路 = 邊）
- 推薦系統（用戶-物品二部圖）

CNN 和 RNN 難以直接處理，需要專門的 GNN 架構。

---

## 2. 圖表示學習 | Graph Representation Learning

### 2.1 節點特徵的挑戰

| 挑戰 | 描述 |
|------|------|
| **非歐幾里得** | 節點沒有固定順序和鄰居數 |
| **鄰接關係** | 節點的語義依賴於其鄰居 |
| **多尺度** | 局部、全局結構都需要考慮 |
| **動態性** | 結構和特徵都可能變化 |

### 2.2 圖學習任務

```
┌──────────────────────────────────────────────────────────────┐
│  GNN 任務分類                                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  節點級 (Node-level)                                          │
│  ├── 節點分類 (Node Classification)                           │
│  ├── 節點回歸 (Node Regression)                               │
│  └── 節點聚類 (Node Clustering)                               │
│                                                              │
│  邊級 (Edge-level)                                            │
│  ├── 鏈接預測 (Link Prediction)                                │
│  ├── 邊分類 (Edge Classification)                              │
│  └── 關係預測                                                 │
│                                                              │
│  圖級 (Graph-level)                                           │
│  ├── 圖分類 (Graph Classification)                            │
│  ├── 圖回歸 (Graph Regression)                                │
│  └── 圖生成 (Graph Generation)                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 經典 GNN 架構 | Classic GNN Architectures

### 3.1 GCN (Graph Convolutional Network, 2017)

**核心思想**：通過鄰居聚合更新節點表示。

**卷積操作**：

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

其中：
- $\tilde{A} = A + I$（加自環）
- $\tilde{D}$ 是 $\tilde{A}$ 的度矩陣
- $H^{(l)}$ 是第 $l$ 層節點特徵
- $W^{(l)}$ 是可學習權重

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """單層 GCN"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        x: (N, in_features)
        adj: (N, N) 鄰接矩陣（含自環）
        """
        # 對稱歸一化
        degree = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[d_inv_sqrt == float('inf')] = 0.0
        D = torch.diag(d_inv_sqrt)
        adj_normalized = D @ adj @ D
        
        # 線性變換
        support = x @ self.weight
        # 聚合
        output = adj_normalized @ support + self.bias
        return output


class GCN(nn.Module):
    """完整 GCN 模型"""
    
    def __init__(self, n_features, n_hidden, n_classes, n_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        # 第一層
        self.layers.append(GCNLayer(n_features, n_hidden))
        # 中間層
        for _ in range(n_layers - 2):
            self.layers.append(GCNLayer(n_hidden, n_hidden))
        # 最後一層
        self.layers.append(GCNLayer(n_hidden, n_classes))
        self.dropout = dropout
    
    def forward(self, x, adj):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
```

### 3.2 GAT (Graph Attention Network, 2018)

**核心思想**：不同鄰居有不同重要性，用注意力機制學習權重。

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_k]))}$$

```python
class GATLayer(nn.Module):
    """GAT 層（單頭）"""
    
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.dropout = dropout
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        """
        h: (N, in_features)
        adj: (N, N) 鄰接矩陣
        """
        Wh = self.W(h)  # (N, out_features)
        N = Wh.size(0)
        
        # 計算注意力係數
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # (N*N, out)
        Wh_repeated_alternating = Wh.repeat(N, 1)  # (N*N, out)
        all_combinations = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # (N*N, 2*out) -> (N*N, 1)
        e = self.leakyrelu(torch.matmul(all_combinations, self.a).squeeze(-1))
        
        # reshape 為 (N, N)
        e = e.view(N, N)
        
        # 遮罩非鄰居
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        
        # 加權聚合
        h_prime = torch.matmul(attention, Wh)
        return h_prime


class GAT(nn.Module):
    """多頭 GAT"""
    
    def __init__(self, n_features, n_hidden, n_classes, n_heads=8, n_layers=2, dropout=0.6):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # 輸入層：多頭
        self.gat_layers.append(
            GATMultiHead(n_features, n_hidden, n_heads, dropout)
        )
        # 中間層
        for _ in range(n_layers - 2):
            self.gat_layers.append(
                GATMultiHead(n_hidden * n_heads, n_hidden, n_heads, dropout)
            )
        # 輸出層：單頭
        self.out_layer = GATLayer(n_hidden * n_heads, n_classes, dropout)
    
    def forward(self, x, adj):
        h = x
        for layer in self.gat_layers:
            h = layer(h, adj)
        return self.out_layer(h, adj)
```

### 3.3 GraphSAGE (2017)

**核心思想**：通過鄰居採樣學習節點表示，便於歸納式學習（inductive）。

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\mathbf{W}^{(l)} \cdot \text{CONCAT}(\mathbf{h}_i^{(l)}, \text{AGG}(\{\mathbf{h}_j^{(l)} : j \in \mathcal{N}(i)\}))\right)$$

```python
class SAGEAggregator(nn.Module):
    """GraphSAGE 聚合器"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)
    
    def forward(self, x, neighbor_agg):
        # x: 自身特徵, neighbor_agg: 鄰居聚合特徵
        combined = torch.cat([x, neighbor_agg], dim=-1)
        return self.linear(combined)


class GraphSAGE(nn.Module):
    """GraphSAGE 模型"""
    
    def __init__(self, n_features, n_hidden, n_classes, n_layers=2, 
                 sample_sizes=[25, 10], dropout=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.sample_sizes = sample_sizes
        
        # 聚合器
        self.aggregators = nn.ModuleList()
        in_dim = n_features
        for i in range(n_layers):
            out_dim = n_hidden if i < n_layers - 1 else n_classes
            self.aggregators.append(SAGEAggregator(in_dim, out_dim))
            in_dim = n_hidden if i < n_layers - 1 else n_classes
        
        self.dropout = dropout
    
    def sample_neighbors(self, adj, node_idx, sample_size):
        """鄰居採樣"""
        neighbors = []
        for node in node_idx:
            adj_nodes = torch.where(adj[node] > 0)[0]
            if len(adj_nodes) == 0:
                neighbors.append(torch.tensor([]))
            elif len(adj_nodes) <= sample_size:
                neighbors.append(adj_nodes)
            else:
                idx = torch.randperm(len(adj_nodes))[:sample_size]
                neighbors.append(adj_nodes[idx])
        return neighbors
    
    def aggregate(self, x, adj, neighbors):
        """鄰居聚合（均值）"""
        agg_features = []
        for node, nbrs in zip(range(x.size(0)), neighbors):
            if len(nbrs) == 0:
                agg_features.append(torch.zeros(x.size(1), device=x.device))
            else:
                agg_features.append(x[nbrs].mean(dim=0))
        return torch.stack(agg_features)
    
    def forward(self, x, adj):
        h = x
        for i in range(self.n_layers):
            neighbors = self.sample_neighbors(adj, torch.arange(x.size(0)), 
                                              self.sample_sizes[i])
            neighbor_agg = self.aggregate(h, adj, neighbors)
            h = self.aggregators[i](h, neighbor_agg)
            if i < self.n_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
```

### 3.4 GIN (Graph Isomorphism Network, 2019)

**核心思想**：WL 測試啟發，理論上最具有區分力的 GNN：

$$\mathbf{h}_i^{(l+1)} = \text{MLP}\left((1 + \epsilon^{(l)}) \mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(l)}\right)$$

```python
class GINLayer(nn.Module):
    """GIN 層"""
    
    def __init__(self, in_features, out_features, eps=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )
        self.eps = eps
    
    def forward(self, x, adj):
        # 鄰居聚合
        neighbor_sum = torch.sparse.mm(adj, x)
        # 更新
        out = self.mlp((1 + self.eps) * x + neighbor_sum)
        return out
```

### 3.5 算法對比

| 特性 | GCN | GAT | GraphSAGE | GIN |
|------|-----|-----|-----------|-----|
| **聚合方式** | 歸一化求和 | 注意力加權 | 採樣+拼接 | 求和+MLP |
| **訓練方式** | Transductive | Transductive | Inductive | Transductive |
| **可解釋性** | 中 | 高 | 中 | 低 |
| **大圖支持** | 弱 | 弱 | 強 | 弱 |
| **理論能力** | 中 | 中 | 中 | 強 |

---

## 4. 圖的讀出與圖級任務 | Graph-level Tasks

### 4.1 圖讀出 (Readout / Pooling)

```python
class GraphReadout(nn.Module):
    """圖級表示學習（多種讀出方式）"""
    
    def __init__(self, hidden_dim, method='sum'):
        super().__init__()
        self.method = method
        if method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
    
    def forward(self, node_features, batch_idx):
        """
        node_features: (N, hidden_dim)
        batch_idx: (N,) - 每個節點屬於哪個圖
        """
        if self.method == 'sum':
            return global_add_pool(node_features, batch_idx)
        elif self.method == 'mean':
            return global_mean_pool(node_features, batch_idx)
        elif self.method == 'max':
            return global_max_pool(node_features, batch_idx)
        elif self.method == 'attention':
            # 基於注意力的池化
            weights = self.attention(node_features)
            weighted = node_features * weights
            return global_sum_pool(weighted, batch_idx) / global_sum_pool(weights, batch_idx)
```

### 4.2 圖分類器

```python
class GraphClassifier(nn.Module):
    """圖分類模型（GIN + Readout）"""
    
    def __init__(self, n_features, n_hidden, n_classes, n_layers=3):
        super().__init__()
        self.atom_embedding = nn.Linear(n_features, n_hidden)
        self.gin_layers = nn.ModuleList([
            GINLayer(n_hidden, n_hidden) for _ in range(n_layers)
        ])
        self.readout = GraphReadout(n_hidden, method='sum')
        self.classifier = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(n_hidden, n_classes),
        )
    
    def forward(self, x, adj, batch_idx):
        h = self.atom_embedding(x)
        for layer in self.gin_layers:
            h = F.relu(layer(h, adj))
        graph_repr = self.readout(h, batch_idx)
        return self.classifier(graph_repr)
```

---

## 5. 圖自監督學習 | Graph Self-Supervised Learning

### 5.1 對比學習

**GRACE (Graph Contrastive Learning)**：

```python
class GRACE(nn.Module):
    """圖對比學習"""
    
    def __init__(self, encoder, hidden_dim, n_features, proj_hidden=256):
        super().__init__()
        self.encoder = encoder  # GCN
        # 投影頭
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, proj_hidden),
            nn.PReLU(),
            nn.Linear(proj_hidden, proj_hidden),
        )
    
    def augment(self, x, adj, drop_edge_p=0.2, mask_feat_p=0.3):
        """圖增強"""
        # 隨機丟邊
        mask = torch.bernoulli(torch.full(adj.shape, 1 - drop_edge_p)).to(adj.device)
        adj_aug = adj * mask
        # 隨機遮蓋特徵
        feat_mask = torch.bernoulli(torch.full(x.shape, 1 - mask_feat_p)).to(x.device)
        x_aug = x * feat_mask
        return x_aug, adj_aug
    
    def forward(self, x, adj):
        # 兩個增強視角
        x1, adj1 = self.augment(x, adj)
        x2, adj2 = self.augment(x, adj)
        
        # 編碼
        z1 = self.proj_head(self.encoder(x1, adj1))
        z2 = self.proj_head(self.encoder(x2, adj2))
        
        # 餘弦相似度
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        return z1, z2
    
    def contrastive_loss(self, z1, z2, temperature=0.5):
        """InfoNCE 損失"""
        N = z1.size(0)
        sim = torch.mm(z1, z2.T) / temperature  # (N, N)
        labels = torch.arange(N, device=z1.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss
```

### 5.2 圖預訓練

| 方法 | 描述 |
|------|------|
| **Node Property Prediction** | 預測節點屬性 |
| **Context Prediction** | 預測節點周圍上下文 |
| **Masking** | 遮蓋節點/邊特徵 |
| **GPT-GNN** | 生成式預訓練 |
| **GraphMAE** | 圖自動編碼器 |

---

## 6. 異質圖 | Heterogeneous Graphs

異質圖含多種類型節點和邊。

```python
class HAN(nn.Module):
    """Heterogeneous Graph Attention Network"""
    
    def __init__(self, meta_paths, in_dims, hidden_dim, out_dim, n_heads=8):
        super().__init__()
        self.meta_paths = meta_paths
        # 為每個元路徑的節點類型分別編碼
        self.type_encoders = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) for ntype, in_dim in in_dims.items()
        })
        # 語義級注意力
        self.semantic_attention = nn.Linear(hidden_dim, 1)
        # 多頭注意力
        self.n_heads = n_heads
    
    def forward(self, features_dict, adj_dict):
        """
        features_dict: {node_type: features}
        adj_dict: {meta_path: adjacency}
        """
        semantic_embeddings = {}
        for meta_path, adj in adj_dict.items():
            # 節點級注意力聚合
            z = self.attention_aggregate(features_dict, adj, meta_path)
            semantic_embeddings[meta_path] = z
        
        # 語義級注意力：不同元路徑權重
        meta_embeddings = torch.stack(list(semantic_embeddings.values()), dim=1)
        attn = F.softmax(self.semantic_attention(meta_embeddings).squeeze(-1), dim=1)
        final = (meta_embeddings * attn.unsqueeze(-1)).sum(dim=1)
        
        return final
```

---

## 7. 時空圖神經網路 | Spatio-Temporal GNN

```python
class STGNN(nn.Module):
    """時空圖神經網路（用於交通預測）"""
    
    def __init__(self, n_features, hidden_dim, n_timesteps):
        super().__init__()
        # 時間卷積
        self.tcn = nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1)
        # 圖卷積
        self.gcn = GCNLayer(hidden_dim, hidden_dim)
        # 門控循環單元（用於時間序列）
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # 預測頭
        self.predictor = nn.Linear(hidden_dim, n_timesteps)
    
    def forward(self, x, adj):
        """
        x: (batch, time, nodes, features)
        """
        # 時間卷積
        # ...
        # 圖卷積聚合
        # ...
        # 預測未來時段
        return self.predictor(...)
```

---

## 8. 圖生成模型 | Graph Generation

```python
class GraphGenerator(nn.Module):
    """簡化的圖生成模型（用於分子生成）"""
    
    def __init__(self, n_features, hidden_dim, n_nodes):
        super().__init__()
        self.n_nodes = n_nodes
        # 節點生成
        self.node_gen = nn.LSTM(n_features, hidden_dim, batch_first=True)
        # 邊預測
        self.edge_pred = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, z):
        """從潛在向量 z 生成圖"""
        # 生成節點嵌入
        h, _ = self.node_gen(z.unsqueeze(0).repeat(self.n_nodes, 1).unsqueeze(0))
        h = h.squeeze(0)  # (n_nodes, hidden)
        
        # 預測邊
        edges = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                edge_prob = self.edge_pred(torch.cat([h[i], h[j]]))
                edges.append((i, j, edge_prob))
        
        return edges
```

---

## 9. 圖神經網路工具庫 | GNN Libraries

| 庫 | 框架 | 特色 |
|----|------|------|
| **PyTorch Geometric (PyG)** | PyTorch | 最流行、研究常用 |
| **DGL** | 通用 | 支持多框架 |
| **GraphSAGE** | PyTorch | 經典 |
| **NetworkX** | Python | 圖分析、算法 |
| **StellarGraph** | TF/Keras | 工業級 |
| **tf-gnn** | TF | Google |
| **CogDL** | PyTorch | 學術 |
| **OGB** | 通用 | 基準數據集 |

```python
# PyTorch Geometric 範例
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, DataLoader

# 構建圖數據
x = torch.randn(100, 32)  # 100 個節點，每個 32 維特徵
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
data = Data(x=x, edge_index=edge_index)

# GCN 模型
class GCN_PyG(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN_PyG(32, 64, 7)
output = model(data)
```

---

## 10. 應用領域 | Applications

| 領域 | 應用 |
|------|------|
| **社交網路** | 用戶分類、好友推薦、社區發現 |
| **推薦系統** | 用戶-物品圖、序列推薦 |
| **分子化學** | 分子性質預測、藥物發現 |
| **交通** | 路況預測、出行時間估計 |
| **知識圖譜** | 實體鏈接、關係抽取、問答 |
| **生物信息** | 蛋白質結構、基因調控網路 |
| **金融** | 反欺詐、風險傳播 |
| **3D 視覺** | 點雲、3D 物體識別 |
| **代碼** | 抽象語法樹、代碼生成 |
| **電商** | 商品關聯、購物籃 |

### 10.1 分子性質預測

```python
# 分子 SMILES -> 圖 -> GNN -> 性質
from rdkit import Chem
from torch_geometric.data import Data

def smiles_to_graph(smiles):
    """SMILES 轉圖數據"""
    mol = Chem.MolFromSmiles(smiles)
    
    # 節點特徵（原子）
    atom_features = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            atom.GetIsAromatic() * 1,
            atom.GetTotalNumHs(),
        ]
        atom_features.append(feat)
    
    # 邊（化學鍵）
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # 無向
    
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

# 用 GNN 預測分子性質
molecule_data = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")  # 阿司匹林
prediction = model(molecule_data)
```

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│  GNN 發展路線                                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  經典方法 (2005-2016)                                          │
│  ├── 圖核方法 (Graph Kernels)                                 │
│  ├── 隨機遊走 (DeepWalk, node2vec)                            │
│  └── 矩陣分解 (Graph Factorization)                          │
│                                                              │
│  深度 GNN (2017+)                                              │
│  ├── GCN (2017) - 圖卷積                                      │
│  ├── GraphSAGE (2017) - 歸納式 + 採樣                         │
│  ├── GAT (2018) - 注意力                                       │
│  ├── GIN (2019) - 理論最優                                    │
│  ├── SGC (2019) - 簡化版                                      │
│  └── ChebNet / APPNP (多項式濾波)                              │
│                                                              │
│  大規模 / 工業 (2019+)                                         │
│  ├── ClusterGCN, GraphSAINT                                  │
│  ├── OGB Benchmark                                            │
│  ├── GNN 與 LLM 結合                                          │
│  └── Graph Foundation Models                                  │
│                                                              │
│  前沿方向                                                      │
│  ├── 異質圖 (HAN, HGT)                                        │
│  ├── 動態圖 (TGN, DyRep)                                      │
│  ├── 時空圖 (Traffic)                                          │
│  ├── 大模型 + 圖 (GraphGPT)                                   │
│  └── 圖語言模型                                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

| 任務 | 推薦模型 |
|------|----------|
| 節點分類（小圖） | GCN, GAT |
| 節點分類（大圖） | GraphSAGE, ClusterGCN |
| 圖分類 | GIN, GCN + Readout |
| 鏈接預測 | VGAE, GAT |
| 大規模圖 | GraphSAGE, GraphSAINT |
| 異質圖 | HAN, HGT |
| 動態圖 | TGN, JODIE |
| 分子 | GIN, MPNN, SchNet |

---

## 參考文獻 | References

1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR* (GCN).
2. Hamilton, W. L., et al. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS* (GraphSAGE).
3. Veličković, P., et al. (2018). Graph Attention Networks. *ICLR* (GAT).
4. Xu, K., et al. (2019). How Powerful are Graph Neural Networks? *ICLR* (GIN).
5. Ying, R., et al. (2018). Hierarchical Graph Representation Learning with Differentiable Pooling. *NeurIPS*.
6. Zhu, Y., et al. (2020). A Survey on Graph Neural Networks. *IEEE TNNLS*.
7. Wu, Z., et al. (2021). A Comprehensive Survey on Graph Neural Networks. *IEEE TNNLS*.

---

*© 2026 chanxaviersy*
