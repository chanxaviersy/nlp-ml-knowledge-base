# 📏 模型評估 | Model Evaluation

> 摘要：模型評估是機器學習流程的關鍵環節，決定了模型是否能泛化到未見資料。本章介紹分類、回歸、聚類的評估指標，交叉驗證方法，以及超參數調優策略。  
> Abstract: Model evaluation is crucial for generalization. This chapter covers evaluation metrics for classification, regression, and clustering, cross-validation methods, and hyperparameter tuning.

---

## 1. 分類評估指標 | Classification Metrics

### 1.1 混淆矩陣 | Confusion Matrix

|  | 預測正 | 預測負 |
|--|--------|--------|
| **實際正** | TP (真陽性) | FN (假陰性) |
| **實際負** | FP (假陽性) | TN (真陰性) |

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

y_true = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 0, 1, 1, 1, 1]

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
```

### 1.2 核心指標

| 指標 | 公式 | 含義 |
|------|------|------|
| **Accuracy** | $\frac{TP+TN}{TP+TN+FP+FN}$ | 整體正確率 |
| **Precision** | $\frac{TP}{TP+FP}$ | 預測為正的樣本中實際為正的比例 |
| **Recall (Sensitivity, TPR)** | $\frac{TP}{TP+FN}$ | 實際為正的樣本中被正確預測的比例 |
| **Specificity (TNR)** | $\frac{TN}{TN+FP}$ | 實際為負的樣本中被正確預測的比例 |
| **F1 Score** | $\frac{2 \cdot P \cdot R}{P + R}$ | Precision 和 Recall 的調和平均 |
| **F-beta** | $(1+\beta^2) \frac{PR}{\beta^2 P + R}$ | 加權 F1，$\beta > 1$ 重視 Recall |

### 1.3 多分類指標

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, multilabel_confusion_matrix
)

# 加權平均
print("Macro F1:", f1_score(y_true, y_pred, average='macro'))     # 各類別權重相同
print("Micro F1:", f1_score(y_true, y_pred, average='micro'))     # 全局計算
print("Weighted F1:", f1_score(y_true, y_pred, average='weighted'))  # 按樣本數加權
```

### 1.4 ROC 曲線與 AUC

```python
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

# 二分類
y_scores = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend()
plt.grid(True)
plt.show()
```

### 1.5 Precision-Recall 曲線

當類別不平衡時，PR 曲線比 ROC 更有信息量：

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_test, y_scores)
ap = average_precision_score(y_test, y_scores)

plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()
```

### 1.6 多分類 ROC

```python
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# One-hot 編碼
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = model.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 繪製每個類別
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()
```

---

## 2. 回歸評估指標 | Regression Metrics

| 指標 | 公式 | 特性 |
|------|------|------|
| **MAE** | $\frac{1}{N}\sum \|y_i - \hat{y}_i\|$ | 對離群值穩健 |
| **MSE** | $\frac{1}{N}\sum (y_i - \hat{y}_i)^2$ | 受離群值影響大 |
| **RMSE** | $\sqrt{\text{MSE}}$ | 與 y 同尺度 |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | 解釋方差比例 |
| **MAPE** | $\frac{1}{N}\sum \|\frac{y_i - \hat{y}_i}{y_i}\|$ | 百分比誤差 |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
print(f"MSE:  {mean_squared_error(y_true, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
print(f"R²:   {r2_score(y_true, y_pred):.4f}")
```

---

## 3. 聚類評估 | Clustering Metrics

### 3.1 內部指標（無真實標籤）

| 指標 | 描述 |
|------|------|
| **輪廓係數 (Silhouette)** | 衡量樣本與自身簇 vs 相鄰簇的相似度，[-1, 1] |
| **Calinski-Harabasz** | 簇間方差 / 簇內方差，越大越好 |
| **Davies-Bouldin** | 簇間相似度，越小越好 |
| **DBCV** | 基於密度，適用 DBSCAN |

### 3.2 外部指標（有真實標籤）

| 指標 | 描述 |
|------|------|
| **ARI (Adjusted Rand Index)** | 調整蘭德指數，[-1, 1] |
| **NMI (Normalized Mutual Info)** | 正規化互資訊，[0, 1] |
| **V-measure** | 同質性與完整性的調和平均 |
| **Fowlkes-Mallows** | 配對精度/召回的幾何平均 |

```python
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)

# 內部
print(f"Silhouette: {silhouette_score(X, y_pred):.4f}")
print(f"Calinski-Harabasz: {calinski_harabasz_score(X, y_pred):.2f}")
print(f"Davies-Bouldin: {davies_bouldin_score(X, y_pred):.4f}")

# 外部
print(f"ARI: {adjusted_rand_score(y_true, y_pred):.4f}")
print(f"NMI: {normalized_mutual_info_score(y_true, y_pred):.4f}")
```

---

## 4. 交叉驗證 | Cross-Validation

### 4.1 K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(random_state=42)

scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"每折分數: {scores}")
print(f"平均分數: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 4.2 分層 K-Fold（類別不平衡）

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

### 4.3 時間序列交叉驗證

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    # 永遠用過去預測未來
```

### 4.4 Leave-One-Out

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
```

---

## 5. 超參數調優 | Hyperparameter Tuning

### 5.1 Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
best_model = grid_search.best_estimator_
```

### 5.2 Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'learning_rate': uniform(0.01, 0.3),  # 對 XGBoost
}

random_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(),
    param_distributions=param_distributions,
    n_iter=50,  # 隨機採樣 50 組
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
)
random_search.fit(X_train, y_train)
```

### 5.3 Bayesian Optimization

```python
# pip install optuna
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }
    model = xgb.XGBClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Best params: {study.best_params_}")
print(f"Best score: {study.best_value:.4f}")
```

### 5.4 三種方法對比

| 方法 | 速度 | 效果 | 適用 |
|------|------|------|------|
| **Grid Search** | 慢 | 全局最優 | 參數少 |
| **Random Search** | 快 | 接近最優 | 參數多 |
| **Bayesian** | 中 | 樣本高效 | 評估昂貴 |

---

## 6. 偏差-方差權衡 | Bias-Variance Tradeoff

```
                    總誤差
                      ↓
   ┌─────────────────────────────────────────┐
   │           Bias² + Variance + Noise       │
   └─────────────────────────────────────────┘

   高偏差（欠擬合）    平衡    高方差（過擬合）
   ┌────┐         ┌────┐         ┌────┐
   │    │         │ ✓  │         │ ╱╲ │
   │    │         │    │         │╱  ╲│
   └────┘         └────┘         └────┘
   簡單模型      適當複雜       過於複雜
```

**診斷方法**：

| 訓練誤差 | 驗證誤差 | 結論 |
|----------|----------|------|
| 高 | 高 | 高偏差（欠擬合） |
| 低 | 高 | 高方差（過擬合） |
| 低 | 低 | 良好 |
| 高 | 低 | 數據洩漏或其他問題 |

**解決方法**：
- 高偏差：增加模型複雜度、增加特徵、減少正則化
- 高方差：增加資料、減少特徵、增加正則化、集成方法

---

## 7. 學習曲線 | Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    estimator=RandomForestClassifier(),
    X=X, y=y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, 'o-', label='Training')
plt.plot(train_sizes, val_mean, 's-', label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Learning Curve')
plt.grid()
plt.show()
```

**判斷**：
- 兩條曲線都低且接近 → 高偏差 → 增加模型複雜度
- 訓練高、驗證低，差距大 → 高方差 → 增加資料或正則化
- 兩條曲線逐漸接近 → 增加資料可能有益

---

## 8. 類別不平衡處理 | Imbalanced Classes

| 策略 | 描述 |
|------|------|
| **SMOTE** | 合成少數類樣本，過採樣 |
| **ADASYN** | 自適應合成，關注難分類樣本 |
| **Random Undersampling** | 隨機刪除多數類 |
| **Class Weight** | 損失函數加權少數類 |
| **Threshold Tuning** | 調整分類閾值 |
| **Focal Loss** | 關注難分類樣本 |

```python
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Class Weight
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))
model = RandomForestClassifier(class_weight=class_weight_dict)
```

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│  完整的評估流程                                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 切分資料                                                  │
│     └── Train / Validation / Test (60/20/20)                 │
│     └── 或 K-Fold Cross-Validation                           │
│                                                              │
│  2. 選擇指標                                                  │
│     ├── 分類：Accuracy, F1, AUC, PR-AUC                      │
│     ├── 回歸：MAE, RMSE, R²                                   │
│     └── 聚類：Silhouette, ARI, NMI                            │
│                                                              │
│  3. 基線模型                                                  │
│     ├── Random / Majority Class                              │
│     └── 簡單線性模型                                          │
│                                                              │
│  4. 訓練 + 交叉驗證                                           │
│     └── Grid / Random / Bayesian 搜參                        │
│                                                              │
│  5. 偏差-方差診斷                                             │
│     └── 學習曲線 + 驗證曲線                                   │
│                                                              │
│  6. 最終評估（僅一次）                                        │
│     └── 在 Test Set 上報告最終分數                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 參考文獻 | References

1. Powers, D. M. (2011). Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation.
2. Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *JMLR*.
3. Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD*.

---

*© 2026 chanxaviersy*
