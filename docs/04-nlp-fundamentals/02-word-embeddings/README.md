# 🔠 詞嵌入 | Word Embeddings

> 摘要：詞嵌入將離散的詞符映射到連續的向量空間，使語義相近的詞在向量空間中距離相近。本章介紹 Word2Vec、GloVe、FastText 等經典方法和現代預訓練嵌入模型。  
> Abstract: Word embeddings map discrete tokens to continuous vector spaces, where semantically similar words are close. This chapter covers Word2Vec, GloVe, FastText, and modern pretrained embeddings.

---

## 1. 為什麼需要詞嵌入？| Why Embeddings?

### 1.1 詞袋模型的問題

| 問題 | 描述 |
|------|------|
| **稀疏性** | 詞表大小 V，句子表示是 1×V 稀疏向量 |
| **維度災難** | V 越大，向量越長 |
| **無語義** | "king" 和 "queen" 是兩個完全獨立的維度，無相似度 |
| **OOV 問題** | 沒見過的詞無法表示 |

### 1.2 詞嵌入的優勢

- **稠密低維**：通常 100-1024 維
- **語義編碼**：相似詞在向量空間中距離近
- **可計算類比**：`king - man + woman ≈ queen`
- **下游友好**：直接餵入神經網路

---

## 2. Word2Vec (Mikolov et al., 2013)

### 2.1 兩種架構

**CBOW (Continuous Bag-of-Words)**：用上下文詞預測中心詞

$$P(w_t | w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})$$

**Skip-Gram**：用中心詞預測上下文詞

$$P(w_{t+j} | w_t), \quad j \in \{-c, ..., -1, 1, ..., c\}$$

| 架構 | 速度 | 效果 |
|------|------|------|
| CBOW | 快 | 小數據集較好 |
| Skip-Gram | 慢 | 大數據集、低頻詞效果好 |

### 2.2 訓練目標

**Skip-Gram 目標函數**：

$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

**Softmax 計算**：

$$P(w_O | w_I) = \frac{\exp(\mathbf{v}_{w_O}^\top \mathbf{v}_{w_I})}{\sum_{w=1}^{V} \exp(\mathbf{v}_w^\top \mathbf{v}_{w_I})}$$

### 2.3 優化技巧：負採樣 (Negative Sampling)

原始 softmax 計算量 $O(V)$，負採樣將其近似為二元分類問題：

$$\log \sigma(\mathbf{v}_{w_O}^\top \mathbf{v}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-\mathbf{v}_{w_i}^\top \mathbf{v}_{w_I})]$$

```python
import gensim
from gensim.models import Word2Vec

# 訓練語料
sentences = [
    ["i", "love", "natural", "language", "processing"],
    ["deep", "learning", "is", "a", "subset", "of", "machine", "learning"],
    ["transformers", "are", "powerful", "models", "for", "nlp"],
    # ... 更多語料
]

# 訓練 Skip-Gram 模型
model = Word2Vec(
    sentences,
    vector_size=100,    # 嵌入維度
    window=5,           # 上下文窗口大小
    min_count=1,        # 最小詞頻
    workers=4,          # 並行線程數
    sg=1,               # 1=Skip-Gram, 0=CBOW
    epochs=10,
    negative=5,         # 負採樣數量
    sample=1e-3,        # 高頻詞下採樣閾值
)

# 使用
vector = model.wv["language"]  # 詞向量
similar = model.wv.most_similar("language", topn=5)
print("Most similar to 'language':", similar)

# 詞類比：king - man + woman ≈ queen
result = model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=3
)
print("king - man + woman ≈", result)
```

### 2.4 從零實現 Skip-Gram

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np

class SkipGramModel(nn.Module):
    """Skip-Gram 模型的簡化實現"""
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # 中心詞嵌入矩陣
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 上下文詞嵌入矩陣
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化
        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.target_embeddings.weight, -init_range, init_range)
        nn.init.zeros_(self.context_embeddings.weight)
    
    def forward(self, target, context, negatives):
        """
        target: (batch,)
        context: (batch,)
        negatives: (batch, n_negatives)
        """
        # 中心詞向量
        t_emb = self.target_embeddings(target)  # (batch, dim)
        # 正樣本上下文
        c_emb = self.context_embeddings(context)  # (batch, dim)
        # 負樣本
        n_emb = self.context_embeddings(negatives)  # (batch, n_neg, dim)
        
        # 正樣本分數（內積）
        pos_score = (t_emb * c_emb).sum(dim=1)
        # 負樣本分數
        neg_score = torch.bmm(n_emb, t_emb.unsqueeze(2)).squeeze(2)
        
        return pos_score, neg_score


def train_skipgram(corpus, embedding_dim=50, epochs=5, batch_size=256, 
                   n_negatives=5, learning_rate=0.01, window=2):
    """訓練 Skip-Gram 模型"""
    
    # 建立詞表
    word_counts = Counter(corpus)
    vocab = {w: i for i, w in enumerate(word_counts.keys())}
    idx_to_word = {i: w for w, i in vocab.items()}
    vocab_size = len(vocab)
    
    # 構建 (target, context) 訓練對
    pairs = []
    for i, word in enumerate(corpus):
        word_idx = vocab[word]
        for j in range(max(0, i - window), min(len(corpus), i + window + 1)):
            if i != j:
                pairs.append((word_idx, vocab[corpus[j]]))
    
    print(f"Vocabulary: {vocab_size}, Training pairs: {len(pairs)}")
    
    # 模型與優化器
    model = SkipGramModel(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 訓練
    for epoch in range(epochs):
        total_loss = 0
        np.random.shuffle(pairs)
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            targets = torch.tensor([p[0] for p in batch])
            contexts = torch.tensor([p[1] for p in batch])
            
            # 隨機負採樣
            negatives = torch.randint(0, vocab_size, (len(batch), n_negatives))
            
            # 前向
            pos_score, neg_score = model(targets, contexts, negatives)
            
            # 損失：對比損失 (max-margin)
            pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-9).mean()
            neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-9).mean()
            loss = pos_loss + neg_loss
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(pairs):.4f}")
    
    # 提取詞向量
    embeddings = model.target_embeddings.weight.detach().numpy()
    return embeddings, vocab, idx_to_word


# 示範訓練
corpus = ("the quick brown fox jumps over the lazy dog "
          "the dog barks at the fox "
          "natural language processing is fun "
          "deep learning is a powerful technique").split()

embeddings, vocab, idx_to_word = train_skipgram(corpus, embedding_dim=20, epochs=50)
print("'language' embedding:", embeddings[vocab["language"]])
```

---

## 3. GloVe (Global Vectors, 2014)

### 3.1 核心思想

結合**全局共現統計**和**局部上下文窗口**，基於共現矩陣 $\mathbf{X}$：

$$X_{ij} = \text{詞 } j \text{ 在詞 } i \text{ 的上下文中出現的次數}$$

### 3.2 目標函數

$$\mathcal{L} = \sum_{i,j=1}^{V} f(X_{ij}) (\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

權重函數 $f(x)$ 對高頻詞降權：

$$f(x) = \begin{cases} (x/x_{max})^\alpha & \text{if } x < x_{max} \\ 1 & \text{otherwise} \end{cases}$$

通常 $x_{max} = 100$，$\alpha = 0.75$。

```python
# 使用預訓練 GloVe
import gensim.downloader

# 下載 GloVe 預訓練模型
glove_vectors = gensim.downloader.load("glove-wiki-gigaword-100")
print(glove_vectors.most_similar("language"))
print("king - man + woman =", glove_vectors.most_similar(
    positive=["king", "woman"], negative=["man"]
)[:3])
```

---

## 4. FastText (Bojanowski et al., 2017)

### 4.1 核心創新

FastText 將每個詞表示為**字符 n-gram** 的集合，解決 OOV 問題：

```python
# "where" → 字符 n-grams (n=3):
# <wh, whe, her, ere, re> + <where>
```

### 4.2 詞表示

$$\mathbf{v}_w = \sum_{g \in G_w} \mathbf{z}_g$$

$G_w$ 是詞 $w$ 的所有 n-gram 集合（含詞本身）。

**優點**：
- ✅ 處理 OOV（罕見詞、新詞）
- ✅ 形態學豐富的語言效果好（芬蘭語、土耳其語）
- ✅ 訓練快

```python
from gensim.models import FastText

# 訓練 FastText
model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=3,        # 最小字符 n-gram
    max_n=6,        # 最大字符 n-gram
    epochs=10,
)

# 處理 OOV
oov_vector = model.wv["newwordthatdoesnotexist"]
similar = model.wv.most_similar("newwordthatdoesnotexist", topn=5)
```

---

## 5. 預訓練詞嵌入對比 | Comparison

| 特性 | Word2Vec | GloVe | FastText |
|------|----------|-------|----------|
| **發表年份** | 2013 | 2014 | 2017 |
| **訓練方式** | 局部窗口 | 全局共現矩陣 | 字符 n-gram |
| **OOV 處理** | ❌ | ❌ | ✅ |
| **訓練速度** | 中 | 中 | 較慢 |
| **效果** | 好 | 好 | 更好（特別是低資源語言） |
| **存儲大小** | 中 | 中 | 大（含 n-gram） |

---

## 6. 上下文相關詞嵌入 | Contextual Embeddings

傳統詞嵌入（如 Word2Vec）是**靜態的**："bank" 在 "river bank" 和 "bank account" 中是同一向量。**上下文嵌入**解決這個問題。

### 6.1 ELMo (2018)

使用雙向 LSTM 產生上下文相關嵌入：

$$\text{ELMo}_k = \gamma \sum_{j=0}^{L} s_j \mathbf{h}_{k,j}$$

不同層學習不同特徵：
- 淺層：語法
- 深層：語義

### 6.2 BERT 詞嵌入

BERT 為每個 token 在不同上下文中產生不同向量：

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")

# 同一個詞在不同句子中
sentences = [
    "我在河邊的**銀行**存錢。",
    "我坐在**銀行**旁邊的長椅上。",
]

for sent in sentences:
    inputs = tokenizer(sent, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # 找到"銀行"對應的 token 索引
    tokens = tokenizer.tokenize(sent)
    bank_idx = tokens.index("銀") if "銀" in tokens else 0
    
    # 不同層的嵌入
    all_hidden = outputs.hidden_states  # tuple of (layer_count+1, batch, seq, dim)
    print(f"\nSentence: {sent}")
    print(f"Layer 0 embedding of 銀: {all_hidden[0][0, bank_idx+1, :5].numpy()}")
    print(f"Layer 12 embedding of 銀: {all_hidden[12][0, bank_idx+1, :5].numpy()}")
    # 注意：相同詞在不同上下文中，特別是深層，會有顯著不同的嵌入
```

---

## 7. 句子嵌入 | Sentence Embeddings

### 7.1 Sentence-BERT (SBERT)

對 BERT 進行修改，使其能產生語義完整的句子向量：

```python
from sentence_transformers import SentenceTransformer

# 載入預訓練多語言句子嵌入模型
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 編碼句子
sentences = [
    "機器學習是人工智慧的一個分支。",
    "深度學習屬於機器學習的範疇。",
    "今天天氣真好，適合出門散步。",
]
embeddings = model.encode(sentences)

# 計算相似度
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(embeddings)
print("句子相似度矩陣：")
for i, s1 in enumerate(sentences):
    for j, s2 in enumerate(sentences):
        if j > i:
            print(f"  {s1[:10]}... vs {s2[:10]}...: {similarity[i, j]:.4f}")
```

### 7.2 BGE / MTEB Leaderboard

| 模型 | 維度 | 多語言 | MTEB 排名 |
|------|------|--------|-----------|
| BAAI/bge-large-en-v1.5 | 1024 | 英文 | 高 |
| BAAI/bge-m3 | 1024 | 多語言 | 高 |
| intfloat/e5-large-v2 | 1024 | 英文 | 高 |
| sentence-transformers/all-mpnet-base-v2 | 768 | 英文 | 中高 |
| OpenAI text-embedding-3-large | 3072 | 多語言 | 最高 |

---

## 8. 詞嵌入的評估 | Evaluation

### 8.1 內在評估 | Intrinsic Evaluation

- **詞類比 (Word Analogy)**：a:b :: c:?
- **詞相似度 (Word Similarity)**：與人類標註的相似度相關性
- **概念分類 (Concept Categorization)**：聚類正確率

### 8.2 外在評估 | Extrinsic Evaluation

將詞嵌入作為特徵，評估下游任務（分類、NER、QA）的性能。

```python
# 詞類比評估
def analogy_test(model, a, b, c, topn=5):
    """
    評估 a:b :: c:?
    例：king:man :: queen:?
    """
    result = model.wv.most_similar(
        positive=[b, c], negative=[a], topn=topn
    )
    return result

print(analogy_test(trained_model, "man", "king", "woman"))
# 期望輸出：queen, princess, monarch...
```

### 8.3 偏差問題 | Bias in Embeddings

詞嵌入可能包含社會偏見（如性別、種族）：

```python
# 經典的性別偏見測試
# doctor - man + woman ≈ nurse（這反映了訓練語料的社會偏見）
```

**緩解方法**：
- 訓練資料去偏
- 後處理去偏（Bolukbasi et al., 2016）
- 構建更平衡的語料

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│  詞嵌入發展演進                                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  靜態詞嵌入 (Static Embeddings)                              │
│  ├── Word2Vec (2013) - 局部窗口, 預測任務                    │
│  ├── GloVe (2014)   - 全局共現矩陣分解                      │
│  └── FastText (2017) - 字符 n-gram, 處理 OOV                │
│                                                              │
│  上下文嵌入 (Contextual Embeddings)                           │
│  ├── ELMo (2018)    - 雙向 LSTM, 特徵融合                    │
│  ├── BERT (2018)    - Transformer, 真正上下文相關             │
│  └── GPT/GPT-2/3    - 單向 Decoder 嵌入                      │
│                                                              │
│  句子/文檔嵌入 (Sentence/Document Embeddings)                │
│  ├── Sentence-BERT  - Siamese 網路微調                        │
│  ├── SimCSE         - 對比學習                                │
│  ├── BGE / M3       - 多語言 SOTA                            │
│  └── E5 / GTE       - 通用文本嵌入                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 參考文獻 | References

1. Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. *arXiv*.
2. Pennington, J., et al. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*.
3. Bojanowski, P., et al. (2017). Enriching Word Vectors with Subword Information. *TACL*.
4. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL*.
5. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.

---

*© 2026 chanxaviersy*
