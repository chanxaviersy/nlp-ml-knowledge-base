# Lab 01: 情感分析 | Sentiment Analysis

> 摘要：本 Lab 使用 TF-IDF + 傳統 ML 算法（Naive Bayes、SVM、邏輯回歸）對中文評論進行情感分類（正面/負面）。  
> Abstract: This lab uses TF-IDF and traditional ML algorithms to classify Chinese reviews as positive or negative.

---

## 🎯 目標 | Objectives

- 學習中文文本預處理
- 掌握 TF-IDF 特徵提取
- 比較不同 ML 算法在文本分類上的表現
- 評估模型性能

---

## 📋 環境準備 | Environment

```bash
pip install scikit-learn jieba pandas numpy matplotlib
```

---

## 📊 數據集 | Dataset

我們將使用一個小型的中文情感評論數據集：

```python
import pandas as pd

# 範例數據（實際使用時可以從更大數據集載入）
data = {
    "text": [
        # 正面
        "這部電影真的太棒了，演員演技精湛，劇情扣人心弦",
        "非常滿意這次購物，商品質量好，物流也很快",
        "餐廳的菜色美味，服務態度也很好，強烈推薦",
        "酒店環境優美，房間乾淨整潔，前台服務很貼心",
        "這本書寫得很好，深入淺出，讓人受益匪淺",
        "手機性能強大，拍照效果出色，性價比高",
        "演唱會現場氣氛熱烈，歌手表現精彩",
        "老師講課生動有趣，收穫很多",
        "公園風景優美，適合休閒散步",
        "產品質量過硬，用起來很順手",
        # 負面
        "非常失望，東西品質差，與描述完全不符",
        "服務態度極差，等了兩個小時還沒處理",
        "電影劇情混亂，完全看不懂在演什麼",
        "食物難吃，價格還特別貴，絕對不會再來",
        "房間有異味，衛生條件堪憂",
        "書的內容空洞，浪費錢",
        "手機經常卡頓，電池續航也很差",
        "演唱會音響效果差，體驗很糟糕",
        "課程內容枯燥，完全沒有收穫",
        "公園環境髒亂，到處都是垃圾",
    ],
    "label": [1] * 10 + [0] * 10,  # 1=positive, 0=negative
}

df = pd.DataFrame(data)
print(f"資料量: {len(df)}")
print(f"類別分佈:\n{df['label'].value_counts()}")
```

**更大型的數據集**：
- ChnSentiCorp（中文情感分析）
- 網易雲音樂評論
- 豆瓣影評

---

## 🔧 步驟 1: 文本預處理 | Text Preprocessing

```python
import re
import jieba

# 加載停用詞
def load_stopwords(path=None):
    """加載中文停用詞"""
    stopwords = set([
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一個",
        "上", "也", "很", "到", "說", "要", "去", "你", "會", "著", "沒有", "看", "好",
        "自己", "這", "那", "嗎", "吧", "啊", "呢", "哦", "嗯", "嗎", "把", "讓", "給",
    ])
    if path:
        with open(path, 'r', encoding='utf-8') as f:
            stopwords.update([line.strip() for line in f])
    return stopwords

STOPWORDS = load_stopwords()

def preprocess_text(text):
    """中文文本預處理"""
    # 1. 清理特殊字符
    text = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text)  # 保留中文、字母、數字
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. 中文分詞
    words = jieba.lcut(text)
    
    # 3. 過濾停用詞和單字符
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    
    return ' '.join(words)

# 測試
sample = "這部電影真的太棒了，演員演技精湛！"
print(f"原文: {sample}")
print(f"處理後: {preprocess_text(sample)}")
```

---

## 🔧 步驟 2: 特徵提取 | Feature Extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1. 預處理所有文本
df['processed'] = df['text'].apply(preprocess_text)

# 2. 切分訓練測試集
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['processed'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)

# 3. TF-IDF 特徵
tfidf = TfidfVectorizer(
    max_features=5000,    # 最大特徵數
    ngram_range=(1, 2),   # 使用 1-gram 和 2-gram
    min_df=1,
    max_df=0.95,
)

X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)

print(f"訓練集形狀: {X_train.shape}")
print(f"測試集形狀: {X_test.shape}")
print(f"詞彙表大小: {len(tfidf.vocabulary_)}")
print(f"前 20 個特徵: {list(tfidf.vocabulary_.items())[:20]}")
```

---

## 🔧 步驟 3: 訓練多個模型 | Train Multiple Models

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import time

# 定義模型
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear SVM": LinearSVC(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# 訓練並評估
results = {}

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "train_time": train_time,
    }
    
    print(f"\n=== {name} ===")
    print(f"訓練時間: {train_time:.3f}s")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"\n分類報告:\n{classification_report(y_test, y_pred, target_names=['負面', '正面'])}")
```

---

## 🔧 步驟 4: 結果比較 | Results Comparison

```python
import matplotlib.pyplot as plt
import numpy as np

# 結果可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左圖：性能指標
names = list(results.keys())
metrics = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(names))
width = 0.2

for i, metric in enumerate(metrics):
    values = [results[n][metric] for n in names]
    axes[0].bar(x + i * width, values, width, label=metric)

axes[0].set_xlabel('Model')
axes[0].set_ylabel('Score')
axes[0].set_title('Model Performance Comparison')
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(names, rotation=15)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, 1.1])

# 右圖：訓練時間
times = [results[n]['train_time'] for n in names]
axes[1].bar(names, times, color='coral')
axes[1].set_ylabel('Training Time (s)')
axes[1].set_title('Training Time')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 🔧 步驟 5: 最佳模型詳細分析 | Best Model Analysis

```python
# 選取最佳模型（這裡假設 LR 最佳）
best_model = LogisticRegression(max_iter=1000, random_state=42)
best_model.fit(X_train, y_train)

# 1. 混淆矩陣
import seaborn as sns
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['負面', '正面'], yticklabels=['負面', '正面'])
plt.xlabel('預測')
plt.ylabel('實際')
plt.title('混淆矩陣')
plt.show()

# 2. 最重要的特徵（單詞）
feature_names = tfidf.get_feature_names_out()
coefficients = best_model.coef_[0]
top_positive = np.argsort(coefficients)[-15:]  # 正面指標
top_negative = np.argsort(coefficients)[:15]   # 負面指標

print("\n=== 預測正面的關鍵詞 ===")
for idx in reversed(top_positive):
    print(f"  {feature_names[idx]:15s} (權重: {coefficients[idx]:.4f})")

print("\n=== 預測負面的關鍵詞 ===")
for idx in top_negative:
    print(f"  {feature_names[idx]:15s} (權重: {coefficients[idx]:.4f})")
```

---

## 🔧 步驟 6: 預測新文本 | Predict on New Text

```python
def predict_sentiment(text, model, vectorizer, preprocessor):
    """預測單個文本的情感"""
    processed = preprocessor(text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    
    sentiment = "正面 😊" if prediction == 1 else "負面 😞"
    confidence = probability[prediction]
    
    return sentiment, confidence

# 測試新文本
test_texts = [
    "這家餐廳的菜真的非常好吃，下次還會再來",
    "完全浪費錢，質量太差了",
    "普通一般，沒什麼特別的感覺",
    "體驗極佳，強烈推薦給大家",
    "非常糟糕的一次購物體驗",
]

print("\n=== 新文本預測 ===")
for text in test_texts:
    sentiment, conf = predict_sentiment(text, best_model, tfidf, preprocess_text)
    print(f"文本: {text}")
    print(f"預測: {sentiment} (信心: {conf:.2%})\n")
```

---

## 📈 步驟 7: 進階優化 | Advanced Optimization

```python
# 1. 調整 TF-IDF 參數
from sklearn.pipeline import Pipeline

# 嘗試不同 n-gram
for ngram in [(1, 1), (1, 2), (1, 3)]:
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=ngram, max_features=5000)),
        ('lr', LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X_train_text, y_train)
    score = pipe.score(X_test_text, y_test)
    print(f"ngram={ngram}: Accuracy={score:.4f}")

# 2. 嘗試字元 n-gram（對中文也有效）
char_tfidf = TfidfVectorizer(
    analyzer='char_wb',  # 字符級別
    ngram_range=(2, 4),
    max_features=5000,
)
X_train_char = char_tfidf.fit_transform(X_train_text)
X_test_char = char_tfidf.transform(X_test_text)

lr_char = LogisticRegression(max_iter=1000)
lr_char.fit(X_train_char, y_train)
print(f"\n字元 n-gram 準確率: {lr_char.score(X_test_char, y_test):.4f}")
```

---

## 🎓 思考題 | Exercises

1. 嘗試載入 **ChnSentiCorp** 數據集，觀察大規模數據下各模型的表現差異
2. 將 `TfidfVectorizer` 換成 `CountVectorizer` 比較效果
3. 嘗試調整 `max_features`、`min_df`、`max_df` 等參數
4. 使用 `cross_val_score` 進行 5 折交叉驗證
5. 嘗試 **BERT** 微調（見 Lab 03）

---

## 📊 實驗結果範例 | Sample Results

| 模型 | Accuracy | Precision | Recall | F1 | 訓練時間 |
|------|----------|-----------|--------|-----|----------|
| Naive Bayes | 0.83 | 0.86 | 0.80 | 0.83 | 0.005s |
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 | 0.02s |
| Linear SVM | 0.83 | 0.86 | 0.80 | 0.83 | 0.005s |
| Random Forest | 0.83 | 0.86 | 0.80 | 0.83 | 0.2s |

**注意**：小數據集結果波動大，實際應用需要更大數據集驗證。

---

## 📚 參考 | References

- [scikit-learn 文本分類文檔](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [ChnSentiCorp 數據集](https://github.com/SophonPlus/ChineseNlpCorpus)
- [jieba 中文分詞](https://github.com/fxsjy/jieba)

---

*© 2026 chanxaviersy*
