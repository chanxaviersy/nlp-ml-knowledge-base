# 📐 監督學習 | Supervised Learning

> 摘要：監督學習從帶標籤的訓練資料中學習輸入到輸出的映射。本章涵蓋線性回歸、邏輯回歸、決策樹、SVM、KNN、隨機森林等經典方法。  
> Abstract: Supervised learning learns mappings from inputs to outputs from labeled training data. This chapter covers linear regression, logistic regression, decision trees, SVM, KNN, and random forests.

---

## 1. 監督學習總覽 | Overview

### 1.1 基本設定

給定訓練集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$，學習一個函數 $f: \mathcal{X} \rightarrow \mathcal{Y}$，使預測誤差最小。

### 1.2 任務分類

| 任務 | 輸出類型 | 例子 |
|------|----------|------|
| **分類 (Classification)** | 離散類別 | 垃圾郵件、圖像分類 |
| **回歸 (Regression)** | 連續數值 | 房價預測、溫度預測 |
| **排序 (Ranking)** | 有序列表 | 搜尋引擎、推薦 |

---

## 2. 線性回歸 | Linear Regression

### 2.1 模型

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$$

### 2.2 損失函數

**均方誤差 (MSE)**：

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

### 2.3 解析解

$$\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 生成資料
np.random.seed(42)
X = np.random.randn(200, 1) * 5
y = 3 * X.squeeze() + 2 + np.random.randn(200) * 2

# 訓練
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 評估
y_pred = model.predict(X_test)
print(f"Slope: {model.coef_[0]:.4f}")  # ≈ 3
print(f"Intercept: {model.intercept_:.4f}")  # ≈ 2
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### 2.4 多項式回歸

處理非線性關係：將特徵 $x$ 擴展為 $[1, x, x^2, ..., x^d]$

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model.fit(X_train, y_train)
```

**注意**：多項式次數過高會過擬合，需要正則化。

---

## 3. 邏輯回歸 | Logistic Regression

### 3.1 模型

雖然名字帶"回歸"，實際上是**分類**算法：

$$P(y=1 | x) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}$$

### 3.2 損失函數

**二元交叉熵 (Binary Cross-Entropy)**：

$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1-y_i) \log (1-\hat{y}_i) \right]$$

### 3.3 多分類

使用 **Softmax** 函數：

$$P(y=k | x) = \frac{e^{\mathbf{w}_k^\top \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^\top \mathbf{x}}}$$

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix

# 加載鳶尾花資料集
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200, multi_class='multinomial')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("混淆矩陣：\n", confusion_matrix(y_test, y_pred))
```

---

## 4. 決策樹 | Decision Tree

### 4.1 結構

```
                [Feature X < 5?]
                /              \
              Yes               No
              /                  \
    [Feature Y < 3?]      [Feature Z > 7?]
      /        \            /          \
    Yes        No         Yes          No
    /            \         /              \
 Leaf         Leaf      Leaf            Leaf
```

### 4.2 分裂準則

**信息增益 (ID3/C4.5)**：

$$IG(D, A) = H(D) - H(D | A)$$

**基尼不純度 (CART)**：

$$Gini(D) = 1 - \sum_{k=1}^{K} p_k^2$$

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    criterion='gini',
    random_state=42,
)
model.fit(X_train, y_train)

# 可視化
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=iris.feature_names, 
          class_names=iris.target_names, filled=True)
plt.show()
```

### 4.3 剪枝 | Pruning

防止過擬合：
- **預剪枝 (Pre-pruning)**：限制樹深度、最小葉子樣本數
- **後剪枝 (Post-pruning)**：cost-complexity pruning $\alpha |T|$

---

## 5. 支援向量機 | Support Vector Machine (SVM)

### 5.1 線性 SVM

目標：找到最大化間隔 (margin) 的超平面：

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i$$

$$\text{s.t. } y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

### 5.2 核方法 | Kernel Trick

通過核函數 $K(\mathbf{x}, \mathbf{x}')$ 將資料映射到高維空間，常見核：

| 核函數 | 公式 | 適用 |
|--------|------|------|
| **線性** | $K(x, x') = x^\top x'$ | 線性可分 |
| **多項式** | $K(x, x') = (x^\top x' + c)^d$ | 中等複雜度 |
| **RBF (高斯)** | $K(x, x') = e^{-\gamma \|x-x'\|^2}$ | 非線性、默認首選 |
| **Sigmoid** | $K(x, x') = \tanh(\alpha x^\top x' + c)$ | 模擬神經網路 |

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# SVM 對特徵尺度敏感
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train_scaled, y_train)
print(f"Accuracy: {model.score(X_test_scaled, y_test):.4f}")
```

---

## 6. K-近鄰 | K-Nearest Neighbors (KNN)

### 6.1 原理

"物以類聚"：測試樣本的類別由最近的 K 個訓練樣本投票決定。

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5,
    metric='minkowski',  # 默認歐氏距離
    weights='distance',  # 加權投票
)
model.fit(X_train_scaled, y_train)
```

### 6.2 K 值選擇

| K 值 | 偏差 / 方差 | 適用 |
|------|-------------|------|
| K=1 | 低偏差、高方差 | 噪聲資料容易過擬合 |
| K=N | 高偏差、低方差 | 永遠預測多數類 |
| K=√N | 平衡 | 默認起點 |

**距離度量**：
- 歐氏距離：$\sqrt{\sum (x_i - y_i)^2}$
- 曼哈頓距離：$\sum |x_i - y_i|$
- 餘弦距離：$1 - \cos(\theta)$

---

## 7. 集成方法 | Ensemble Methods

### 7.1 Bagging：隨機森林

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',  # 每棵樹看 sqrt(d) 個特徵
    random_state=42,
)
model.fit(X_train, y_train)
```

**特徵重要性**：

```python
import pandas as pd

importances = pd.Series(model.feature_importances_, index=iris.feature_names)
importances.sort_values(ascending=True).plot(kind='barh')
```

### 7.2 Boosting：梯度提升

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
)
model.fit(X_train, y_train)
```

### 7.3 Stacking

```python
from sklearn.ensemble import StackingClassifier

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('xgb', xgb.XGBClassifier(n_estimators=100)),
    ('svm', SVC(probability=True)),
]
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
)
```

---

## 8. 樸素貝葉斯 | Naive Bayes

基於貝葉斯定理 + 條件獨立假設：

$$P(y | x_1, ..., x_n) \propto P(y) \prod_{i=1}^{n} P(x_i | y)$$

```python
from sklearn.naive_bayes import MultinomialNB, GaussianNB

# 高斯 NB（連續特徵）
model = GaussianNB()
model.fit(X_train, y_train)

# 多項式 NB（離散計數，如文本）
model = MultinomialNB()
```

---

## 總結 | Summary

| 算法 | 適用場景 | 優點 | 缺點 |
|------|----------|------|------|
| **線性回歸** | 連續值預測 | 簡單、可解釋 | 只能捕捉線性 |
| **邏輯回歸** | 二/多分類 | 概率輸出、快速 | 線性決策邊界 |
| **決策樹** | 分類/回歸 | 可解釋、無需特徵縮放 | 易過擬合 |
| **SVM** | 中小資料集 | 核方法強大 | 大資料慢、需要調參 |
| **KNN** | 小資料集 | 簡單、無訓練 | 預測慢、維度災難 |
| **隨機森林** | 通用 | 強基線、穩健 | 推理較慢、不易解釋 |
| **XGBoost** | 結構化資料 | Kaggle 利器 | 調參多、易過擬合 |
| **樸素貝葉斯** | 文本分類 | 極快、適合高維 | 條件獨立假設強 |

---

## 參考文獻 | References

1. Hastie, T., et al. (2009). *The Elements of Statistical Learning*. Springer.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
3. Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*.

---

*© 2026 chanxaviersy*
