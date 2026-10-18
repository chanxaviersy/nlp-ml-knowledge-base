# 📊 Statistics for ML | 機器學習統計學

> 摘要：本章介紹機器學習所需的統計學基礎，包括描述性統計、推論統計、假設檢定、貝氏推論和回歸分析。這些是理解模型行為和實證評估的關鍵工具。  
> Abstract: This chapter introduces statistical foundations for machine learning, covering descriptive statistics, inferential statistics, hypothesis testing, Bayesian inference, and regression analysis.

---

## 1. 描述性統計 | Descriptive Statistics

### 1.1 集中趨勢 | Measures of Central Tendency

| 指標 | 公式 | 適用場景 |
|------|------|----------|
| **均值 (Mean)** | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ | 對稱分佈、無極端值 |
| **中位數 (Median)** | 排序後中間值 | 偏態分佈、有異常值 |
| **眾數 (Mode)** | 出現次數最多的值 | 類別資料、檢測多峰分佈 |

```python
import numpy as np
from scipy import stats

data = np.array([1, 2, 2, 3, 4, 4, 4, 5, 6, 100])  # 含離群值

print(f"Mean:   {np.mean(data):.2f}")      # 12.9 (受離群值影響)
print(f"Median: {np.median(data):.2f}")    # 4.0  (穩健)
print(f"Mode:   {stats.mode(data).mode}")  # 4
```

### 1.2 離散程度 | Measures of Dispersion

$$\text{Variance} = s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

$$\text{Std Dev} = s = \sqrt{s^2}$$

```python
data = np.random.randn(1000) * 5 + 10
print(f"Variance: {np.var(data, ddof=1):.4f}")  # 樣本方差
print(f"Std Dev:  {np.std(data, ddof=1):.4f}")
print(f"IQR:      {stats.iqr(data):.4f}")        # 四分位距
```

### 1.3 偏度與峰度 | Skewness & Kurtosis

- **偏度 (Skewness)**：分佈對稱性。正偏 = 右尾長，負偏 = 左尾長
- **峰度 (Kurtosis)**：尾部厚度。高峰度 = 重尾、低峰度 = 輕尾

---

## 2. 機率分佈 | Probability Distributions

### 2.1 離散分佈

| 分佈 | PMF | 期望 | 方差 | 應用 |
|------|-----|------|------|------|
| Bernoulli | $P(X=1)=p$ | $p$ | $p(1-p)$ | 二元事件 |
| Binomial | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | 試驗成功次數 |
| Poisson | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | 稀有事件計數 |
| Categorical | $p_i$ | - | - | 多類別分佈 |
| Geometric | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ | 首次成功試驗次數 |

### 2.2 連續分佈

| 分佈 | PDF | 期望 | 方差 | 應用 |
|------|-----|------|------|------|
| Uniform | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | 隨機生成 |
| Normal | $\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | 中心極限定理 |
| Exponential | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | 等待時間 |
| Beta | - | $\frac{\alpha}{\alpha+\beta}$ | - | 機率的機率 |
| Gamma | - | $k\theta$ | $k\theta^2$ | 形狀參數分佈 |

### 2.3 中心極限定理 | Central Limit Theorem (CLT)

> 從任何分佈中抽取 $n$ 個獨立同分佈樣本，當 $n$ 足夠大時，樣本均值的分佈趨近於常態分佈。

$$\bar{X}_n \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$$

這是**為什麼常態分佈如此重要**的根本原因。

```python
import matplotlib.pyplot as plt
import numpy as np

def clt_demo(distribution="exponential", n_samples=30, n_experiments=10000):
    """展示中心極限定理"""
    
    # 從非正態分佈採樣
    if distribution == "exponential":
        population = np.random.exponential(scale=2.0, size=100000)
    elif distribution == "uniform":
        population = np.random.uniform(0, 1, size=100000)
    
    # 重複計算樣本均值
    sample_means = []
    for _ in range(n_experiments):
        sample = np.random.choice(population, size=n_samples)
        sample_means.append(np.mean(sample))
    
    sample_means = np.array(sample_means)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左圖：原始分佈
    axes[0].hist(population, bins=50, density=True, alpha=0.7, color='skyblue')
    axes[0].set_title(f"Original Distribution: {distribution}", fontsize=13)
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    
    # 右圖：樣本均值的分佈
    axes[1].hist(sample_means, bins=50, density=True, alpha=0.7, color='salmon')
    axes[1].axvline(np.mean(sample_means), color='red', linestyle='--', label='Mean')
    axes[1].set_title(f"Sampling Distribution of Mean (n={n_samples})", fontsize=13)
    axes[1].set_xlabel("Sample Mean")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Population mean: {np.mean(population):.4f}")
    print(f"Distribution of sample means — Mean: {np.mean(sample_means):.4f}, "
          f"Std: {np.std(sample_means):.4f}")

clt_demo("exponential", n_samples=30)
```

---

## 3. 假設檢定 | Hypothesis Testing

### 3.1 基本概念

| 概念 | 描述 |
|------|------|
| **零假設 $H_0$** | 待檢驗的預設假設（通常：無效應、無差異） |
| **對立假設 $H_1$** | 與 $H_0$ 對立的假設 |
| **顯著性水準 $\alpha$** | 拒絕 $H_0$ 的最大容許錯誤率（通常 0.05） |
| **p 值** | 在 $H_0$ 為真時，觀察到當前結果或更極端的機率 |
| **Type I Error** | 拒絕了正確的 $H_0$（假陽性），概率 = $\alpha$ |
| **Type II Error** | 未拒絕錯誤的 $H_0$（假陰性），概率 = $\beta$ |
| **統計功效 (Power)** | $1 - \beta$，正確拒絕錯誤 $H_0$ 的能力 |

### 3.2 假設檢定流程

```
┌──────────────────────────────────────────────────────────┐
│  假設檢定流程                                              │
├──────────────────────────────────────────────────────────┤
│  1. 定義假設：H₀, H₁                                      │
│  2. 選擇檢定方法（t 檢定、Z 檢定、χ²、ANOVA...）          │
│  3. 計算檢定統計量                                         │
│  4. 計算 p 值                                              │
│  5. 比較 p 值與 α，做出決策                               │
│  6. 計算效應量 (effect size)                              │
└──────────────────────────────────────────────────────────┘
```

### 3.3 常見檢定方法 | Common Tests

| 檢定 | 適用場景 | 假設 |
|------|----------|------|
| **單樣本 t 檢定** | 樣本均值是否等於已知值 | 常態分佈 |
| **雙樣本 t 檢定** | 兩組均值是否相等 | 常態 + 等方差 |
| **Welch's t 檢定** | 兩組均值是否相等（不等方差） | 常態 |
| **配對 t 檢定** | 配對數據的差異 | 差值常態 |
| **卡方檢定** | 類別變數獨立性、適合度 | 期望頻率 ≥ 5 |
| **ANOVA** | 多組均值比較 | 常態 + 等方差 |
| **Mann-Whitney U** | 兩組中位數比較（非參數） | 不要求常態 |

```python
import numpy as np
from scipy import stats

# 雙樣本 t 檢定：新舊模型性能是否顯著差異
np.random.seed(42)
model_a_scores = np.random.normal(0.85, 0.05, 100)  # 舊模型
model_b_scores = np.random.normal(0.88, 0.05, 100)  # 新模型

t_stat, p_value = stats.ttest_ind(model_a_scores, model_b_scores)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.4f}")

if p_value < 0.05:
    print("→ 在 α=0.05 水準下，拒絕 H₀：兩組均值有顯著差異")
else:
    print("→ 沒有足夠證據拒絕 H₀")

# 效應量 (Cohen's d)
mean_diff = np.mean(model_b_scores) - np.mean(model_a_scores)
pooled_std = np.sqrt((np.std(model_a_scores)**2 + np.std(model_b_scores)**2) / 2)
cohens_d = mean_diff / pooled_std
print(f"Cohen's d: {cohens_d:.4f} (小: 0.2, 中: 0.5, 大: 0.8)")
```

### 3.4 多重比較校正 | Multiple Comparison Correction

當進行多次假設檢定時，家族錯誤率 (Family-Wise Error Rate) 會膨脹：

$$\text{FWER} = 1 - (1-\alpha)^m$$

**Bonferroni 校正**：$\alpha_{adjusted} = \alpha / m$

```python
from statsmodels.stats.multitest import multipletests

# 假設 100 個特徵的 t 檢定
p_values = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.15, 0.49])

# Bonferroni
rejected_bonf, p_bonf, _, _ = multipletests(p_values, method='bonferroni')
# Benjamini-Hochberg (FDR 控制)
rejected_bh, p_bh, _, _ = multipletests(p_values, method='fdr_bh')

for i, p in enumerate(p_values):
    print(f"Original p={p:.3f} | Bonferroni p={p_bonf[i]:.3f} "
          f"(reject={rejected_bonf[i]}) | BH p={p_bh[i]:.3f} (reject={rejected_bh[i]})")
```

---

## 4. 貝氏推論 | Bayesian Inference

### 4.1 貝氏定理

$$P(\theta | D) = \frac{P(D | \theta) \cdot P(\theta)}{P(D)}$$

- $P(\theta)$：**先驗分佈 (Prior)** — 對參數的先驗信念
- $P(D | \theta)$：**似然 (Likelihood)** — 數據在參數下的可能性
- $P(\theta | D)$：**後驗分佈 (Posterior)** — 觀察數據後的更新信念
- $P(D)$：**證據 (Evidence)** — 邊際似然

### 4.2 共軛先驗 | Conjugate Priors

| 似然 | 先驗 | 後驗 |
|------|------|------|
| Bernoulli | Beta | Beta |
| Binomial | Beta | Beta |
| Poisson | Gamma | Gamma |
| Normal (μ 未知) | Normal | Normal |
| Normal (σ² 未知) | Inverse-Gamma | Inverse-Gamma |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def beta_binomial_demo(n_trials=100, n_successes=60):
    """Beta-Binomial 共軛先驗示例"""
    
    # 三種先驗信念
    priors = {
        "Uniform α=1, β=1":       stats.beta(1, 1),
        "Weakly informative α=2, β=2":  stats.beta(2, 2),
        "Strongly informative α=20, β=10":  stats.beta(20, 10),  # 預期高成功率
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 1, 1000)
    
    for label, prior in priors.items():
        # Beta-Binomial 共軛：後驗 = Beta(α + successes, β + failures)
        alpha_post = prior.args[0] + n_successes
        beta_post = prior.args[1] + (n_trials - n_successes)
        posterior = stats.beta(alpha_post, beta_post)
        
        ax.plot(x, posterior.pdf(x), label=f"Posterior: {label}", linewidth=2)
    
    ax.axvline(n_successes / n_trials, color='red', linestyle='--', 
               label=f'MLE = {n_successes/n_trials}')
    ax.set_xlabel("θ (success probability)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Posterior after {n_successes}/{n_trials} successes", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.show()

beta_binomial_demo()
```

### 4.3 MCMC 採樣

對於沒有解析後驗的複雜模型，使用 MCMC：

```python
# 使用 PyMC 進行貝氏推論
import pymc as pm
import numpy as np

np.random.seed(42)
observed_data = np.random.normal(loc=5, scale=2, size=200)

with pm.Model() as bayesian_model:
    # 先驗
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    
    # 似然
    likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=observed_data)
    
    # MCMC 採樣
    trace = pm.sample(2000, return_inferencedata=True, progressbar=False)

print(trace.posterior["mu"].mean(), trace.posterior["sigma"].mean())
```

---

## 5. 回歸分析 | Regression Analysis

### 5.1 線性回歸 | Linear Regression

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 I)$$

**最小二乘解**：

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

**正則化變體**：

| 方法 | 目標函數 | 特性 |
|------|----------|------|
| OLS | $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$ | 無約束 |
| Ridge | $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|_2^2$ | L2 正則化 |
| Lasso | $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|_1$ | L1 正則化，產生稀疏解 |
| ElasticNet | 兩者加權組合 | 平衡 L1/L2 |

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42)
X = np.random.randn(100, 5)
true_beta = np.array([1, 0.5, 0, 0, -0.3])
y = X @ true_beta + 0.5 * np.random.randn(100)

models = {
    "OLS":        LinearRegression(),
    "Ridge":      Ridge(alpha=1.0),
    "Lasso":      Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
}

for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    print(f"{name:12s} | R²: {r2_score(y, y_pred):.4f} | "
          f"Coefs: {np.round(model.coef_, 3)}")
```

### 5.2 模型評估指標

| 指標 | 公式 | 特性 |
|------|------|------|
| **MSE** | $\frac{1}{n}\sum (y - \hat{y})^2$ | 受離群值影響大 |
| **RMSE** | $\sqrt{\text{MSE}}$ | 與 y 同尺度 |
| **MAE** | $\frac{1}{n}\sum \|y - \hat{y}\|$ | 對離群值穩健 |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | 解釋方差比例 |
| **Adjusted R²** | $1 - \frac{(1-R^2)(n-1)}{n-p-1}$ | 對變數數量懲罰 |

---

## 6. 抽樣方法 | Sampling Methods

### 6.1 簡單隨機抽樣 | Simple Random Sampling

每個樣本被選中概率相同。

### 6.2 分層抽樣 | Stratified Sampling

按重要變量分層，保證每層都有樣本：

```python
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

df = pd.DataFrame({
    "feature": np.random.randn(1000),
    "class":   np.random.choice([0, 1], 1000, p=[0.3, 0.7])
})

# 分層抽樣保持類別比例
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(df, df["class"]):
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    print(f"Train class dist: {train['class'].value_counts(normalize=True).values}")
    print(f"Test  class dist: {test['class'].value_counts(normalize=True).values}")
```

### 6.3 Bootstrap | 自助法

從樣本中有放回地抽樣，估計統計量的不確定性：

```python
def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, ci=0.95):
    """計算統計量的 Bootstrap 置信區間"""
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_stats, 100 * alpha)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha))
    
    return lower, upper, np.mean(bootstrap_stats)

data = np.random.normal(100, 15, 50)
lower, upper, mean = bootstrap_ci(data)
print(f"95% CI: [{lower:.2f}, {upper:.2f}], Mean: {mean:.2f}")
```

---

## 總結 | Summary

| 概念 | 應用於 ML |
|------|----------|
| 描述性統計 | 探索性資料分析 (EDA)、資料理解 |
| 機率分佈 | 模型輸出校準、不確定性估計 |
| 假設檢定 | A/B 測試、模型性能差異檢驗 |
| 貝氏推論 | 先驗知識整合、超參數優化 |
| 線性回歸 | 基線模型、特徵重要性分析 |
| Bootstrap | 模型穩健性、置信區間估計 |

---

## 參考文獻 | References

1. James, G., et al. (2013). *An Introduction to Statistical Learning*. Springer.
2. Gelman, A., et al. (2013). *Bayesian Data Analysis*. Chapman & Hall.
3. Casella, G., & Berger, R. L. (2002). *Statistical Inference*. Duxbury.

---

*© 2026 chanxaviersy*
