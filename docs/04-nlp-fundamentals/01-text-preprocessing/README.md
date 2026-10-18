# 🔤 NLP 文本預處理 | Text Preprocessing

> 摘要：文本預處理是 NLP 流程的第一步，包括分詞、正規化、停用詞過濾等操作。高質量的預處理能顯著提升下游模型性能。  
> Abstract: Text preprocessing is the first step in NLP pipelines, including tokenization, normalization, and stopword filtering. High-quality preprocessing can dramatically improve downstream model performance.

---

## 1. 文本預處理流程 | Preprocessing Pipeline

```
原始文本 (Raw Text)
   ↓
1. 清理 (Cleaning)        → 去除 HTML、URL、特殊字符
   ↓
2. 正規化 (Normalization)  → 大小寫、Unicode、全半形
   ↓
3. 分詞 (Tokenization)     → 詞級、子詞級、字符級
   ↓
4. 詞形還原 (Lemmatization) → 詞根、詞性標註
   ↓
5. 停用詞過濾 (Stopwords)   → 移除常見無意義詞
   ↓
6. 詞性標註 (POS Tagging)   → 標註每個詞的語法角色
   ↓
7. 命名實體識別 (NER)       → 識別專有名詞
   ↓
預處理後的 Token 列表
```

---

## 2. 文本清理 | Text Cleaning

```python
import re
import unicodedata

def clean_text(text):
    """通用文本清理"""
    # 1. 移除 HTML 標籤
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. 移除 URL
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 3. 移除電子郵件
    text = re.sub(r'\S+@\S+', '', text)
    
    # 4. 移除表情符號
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # 表情
        u"\U0001F300-\U0001F5FF"  # 符號和象形文字
        u"\U0001F680-\U0001F6FF"  # 交通和地圖
        u"\U0001F1E0-\U0001F1FF"  # 旗幟
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # 5. 移除多餘空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 繁簡中文處理
def normalize_chinese(text):
    """繁簡轉換、全半形統一"""
    # OpenCC 進行繁簡轉換
    import opencc
    converter = opencc.OpenCC('t2s')  # 繁體轉簡體
    text = converter.convert(text)
    
    # 全形轉半形
    text = unicodedata.normalize('NFKC', text)
    
    return text
```

---

## 3. 分詞 | Tokenization

### 3.1 英文分詞

```python
import nltk
import spacy

# nltk 下載
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

# 方法1：NLTK
from nltk.tokenize import word_tokenize, sent_tokenize
text = "Hello world! How are you doing today? NLP is fascinating."
print("Sentences:", sent_tokenize(text))
print("Words:", word_tokenize(text))

# 方法2：spaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print("spaCy tokens:", [token.text for token in doc])
```

### 3.2 中文分詞

```python
import jieba

# 基本分詞
text = "自然語言處理是人工智慧的一個重要方向。"
words = jieba.lcut(text)
print(words)
# ['自然', '語言', '處理', '是', '人工智慧', '的', '一個', '重要', '方向', '。']

# 載入自訂詞典
jieba.load_userdict("user_dict.txt")
# 格式：詞 詞頻 詞性

# 詞性標註
import jieba.posseg as pseg
words = pseg.cut("我愛自然語言處理。")
for word, flag in words:
    print(f"{word}\t{flag}")
```

### 3.3 子詞分詞 | Subword Tokenization (BPE, WordPiece, Unigram)

現代 Transformer 普遍使用子詞分詞，能平衡詞表大小和 OOV 問題。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

text = "自然語言處理很有趣"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['自', '然', '語', '言', '處', '理', '很', '有', '趣']

# 編碼為模型輸入
encoded = tokenizer(text, return_tensors="pt")
print(encoded)
# {'input_ids': tensor([[101, 4692, 3979, ...]]), 'attention_mask': tensor([[1, 1, ...]])}

# 解碼
decoded = tokenizer.decode(encoded["input_ids"][0])
print(decoded)
```

**BPE (Byte Pair Encoding) 算法**：

```python
def learn_bpe(corpus, num_merges=10):
    """簡化版 BPE 學習算法"""
    # 初始化詞表：每個字符獨立
    vocab = set()
    for word in corpus:
        vocab.update(word.split())
    
    for i in range(num_merges):
        # 統計相鄰 token 對的頻率
        pairs = {}
        for word in corpus:
            tokens = word.split()
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j+1])
                pairs[pair] = pairs.get(pair, 0) + 1
        
        if not pairs:
            break
        
        # 合併頻率最高的對
        best_pair = max(pairs, key=pairs.get)
        new_token = "".join(best_pair)
        vocab.add(new_token)
        print(f"Merge {i+1}: {best_pair} -> {new_token}")
        
        # 更新語料庫
        corpus = [w.replace(" ".join(best_pair), new_token) for w in corpus]
    
    return vocab

corpus = ["low low low low lowest lowest newer newer newer newer newer wider wider wider"]
learn_bpe(corpus, num_merges=10)
```

---

## 4. 詞形還原 | Lemmatization & Stemming

```python
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

words = ["running", "ran", "runs", "easily", "fairly", "better"]

print("Original | Stem | Lemma")
for word in words:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word, pos='v')  # 'v' 動詞
    print(f"{word:10s} | {stem:8s} | {lemma}")
```

| 詞 | 詞幹 (Stem) | 詞元 (Lemma) |
|----|------------|--------------|
| running | run | running (詞性需正確) |
| ran | ran | run |
| better | better | good |

**重要**：詞形還原需要詞性標註，否則可能不準確。

```python
# 結合詞性標註
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The children were running quickly.")
for token in doc:
    print(f"{token.text:12s} | POS: {token.pos_:6s} | Lemma: {token.lemma_}")
```

---

## 5. 停用詞過濾 | Stopword Filtering

```python
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# 英文停用詞
stop_words_en = set(stopwords.words('english'))
print(f"English stopwords count: {len(stop_words_en)}")
print(list(stop_words_en)[:20])

# 中文停用詞（自定義或使用哈工大停用詞表）
chinese_stopwords = set([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一個",
    "上", "也", "很", "到", "說", "要", "去", "你", "會", "著", "沒有", "看", "好",
    "自己", "這", "那", "嗎", "吧", "啊", "呢", "哦", "嗯"
])

def remove_stopwords(tokens, stopwords):
    return [t for t in tokens if t.lower() not in stopwords and t not in stopwords]

# 注意：對於 Transformer 模型，通常不移除停用詞
# 因為注意力機制能自動學習哪些詞重要
```

---

## 6. N-Gram 特徵 | N-Gram Features

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba

corpus = [
    "我喜歡自然語言處理",
    "自然語言處理很有趣",
    "深度學習是機器學習的分支",
    "機器學習和深度學習都很重要"
]

# 對中文先分詞
corpus_cut = [" ".join(jieba.lcut(doc)) for doc in corpus]

# 1. 詞袋模型 (Bag-of-Words)
bow = CountVectorizer()
X_bow = bow.fit_transform(corpus_cut)
print("BoW vocabulary:", bow.get_feature_names_out())
print("BoW matrix:\n", X_bow.toarray())

# 2. TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus_cut)
print("TF-IDF matrix:\n", X_tfidf.toarray().round(3))

# 3. N-gram 特徵
bigram = CountVectorizer(ngram_range=(1, 2))
X_bigram = bigram.fit_transform(corpus_cut)
print("Bigrams:", bigram.get_feature_names_out())
```

---

## 7. 詞性標註 | POS Tagging

```python
import nltk
import jieba.posseg as pseg

# 英文
nltk.download('averaged_perceptron_tagger_eng')
tokens = nltk.word_tokenize("I love natural language processing.")
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)
# [('I', 'PRP'), ('love', 'VBP'), ('natural', 'JJ'), ...]

# 中文
words = pseg.cut("我正在學習自然語言處理。")
for word, flag in words:
    print(f"{word}\t{flag}")
# 我 r
# 正在 t
# 學習 v
# 自然語言處理 n
# 。 w
```

**常用詞性標籤 (Penn Treebank)**：

| 標籤 | 詞性 | 例子 |
|------|------|------|
| NN | 名詞（單數） | dog, book |
| NNS | 名詞（複數） | dogs, books |
| VB | 動詞（原形） | run |
| VBD | 動詞（過去式） | ran |
| VBG | 動詞（現在分詞） | running |
| JJ | 形容詞 | beautiful |
| RB | 副詞 | quickly |
| IN | 介詞 | in, on, at |
| DT | 限定詞 | the, a, an |

---

## 8. 命名實體識別 | Named Entity Recognition (NER)

```python
import spacy
nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. is planning to open a new office in Tokyo by 2026. Tim Cook is the CEO."
doc = nlp(text)

print(f"{'Entity':20s} | {'Type':10s} | Description")
print("-" * 60)
for ent in doc.ents:
    print(f"{ent.text:20s} | {ent.label_:10s} | {spacy.explain(ent.label_)}")
# Apple Inc.           | ORG        | Companies, agencies, institutions
# Tokyo                | GPE        | Geopolitical entities
# 2026                 | DATE       | Absolute or relative dates
# Tim Cook             | PERSON     | People
```

**中文 NER**：

```python
# 使用 HuggingFace
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# 加載中文 NER 模型
model_name = "bert-base-chinese"  # 或 dbmdz/bert-large-chinese-cased-whole-word-masking
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# 或使用專門的 NER 模型
ner_pipeline = pipeline("ner", model="ckiplab/bert-base-chinese-ner")

text = "張三在台北的台積電工作。"
results = ner_pipeline(text)
for r in results:
    print(f"{r['word']}\t{r['entity']}\t{r['score']:.4f}")
```

---

## 9. 中文特殊處理 | Chinese-specific Preprocessing

```python
# 1. 繁簡轉換
import opencc

converter_t2s = opencc.OpenCC('t2s')  # 繁體 → 簡體
converter_s2t = opencc.OpenCC('s2t')  # 簡體 → 繁體

text = "繁體中文轉換測試"
print("簡體:", converter_t2s.convert(text))

# 2. 全半形轉換
import unicodedata

def full_to_half(text):
    """全形字符轉半形"""
    return ''.join(c for c in text if not unicodedata.east_asian_width(c) == 'W')

# 3. 中文標點處理
import re

def normalize_punctuation(text):
    """統一標點符號"""
    text = re.sub(r'[,，]+', ',', text)  # 統一逗號
    text = re.sub(r'[.。]+', '.', text)  # 統一句號
    text = re.sub(r'!！+', '!', text)
    text = re.sub(r'\?？+', '?', text)
    return text
```

---

## 10. 預處理最佳實踐 | Best Practices

```
┌──────────────────────────────────────────────────────────────┐
│  文本預處理最佳實踐                                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ✅ 應該做：                                                  │
│  • 統一編碼 (UTF-8)                                          │
│  • 移除 HTML/XML 標籤                                       │
│  • 統一標點符號                                              │
│  • 處理特殊字符（表情符號、URL、@mentions）                  │
│  • 對 BERT/Transformer：只做基本清理，不要做詞幹化           │
│                                                              │
│  ❌ 不應該做：                                                │
│  • 對深度學習模型做停用詞過濾（會丟失信息）                  │
│  • 過度詞幹化（會合併不同語義的詞）                          │
│  • 移除數字（除非確定無用）                                  │
│  • 過度清理（保留情緒、表情等微妙的語義信息）                │
│                                                              │
│  ⚠️ 視情況而定：                                              │
│  • 繁簡轉換（依任務需求）                                    │
│  • 大小寫轉換（英文 NER 不要轉，BERT 內部處理）             │
│  • 分詞粒度（中文一般用詞級，罕見詞用字符級）                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 總結 | Summary

| 預處理步驟 | 傳統 ML | 深度學習/Transformer |
|-----------|---------|---------------------|
| 文本清理 | ✅ 必要 | ✅ 必要 |
| 分詞 | ✅ 必要 | ✅ 子詞級 |
| 大小寫轉換 | ✅ 必要 | ❌ 不需要 (BERT 內部處理) |
| 停用詞過濾 | ✅ 必要 | ❌ 通常不過濾 |
| 詞幹化/詞形還原 | ✅ 必要 | ❌ 通常不需要 |
| 詞性標註 | 特徵工程用 | ❌ 通常不需要 |
| NER | 特徵工程用 | ❌ 端到端學習 |

---

## 參考文獻 | References

1. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.).
2. Sennrich, R., et al. (2016). Neural Machine Translation of Rare Words with Subword Units. *ACL*.
3. Wu, Y., et al. (2016). Google's Neural Machine Translation System. *arXiv*.
4. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL*.

---

*© 2026 chanxaviersy*
