# Scripts 工具集合 | Utility Scripts

通用工具模組，配合各 Lab 使用。

## 📁 文件

| 文件 | 功能 |
|------|------|
| `utils.py` | 基礎工具（評估、可視化、隨機種子） |
| `text_preprocessing.py` | 中文/英文文本預處理管道 |
| `trainer.py` | PyTorch 通用訓練器 |
| `metrics.py` | NLP/ML 評估指標 |

## 🚀 使用

```python
# 基礎工具
from scripts.utils import set_seed, plot_confusion_matrix, evaluate_classification

# 預處理
from scripts.text_preprocessing import preprocess_pipeline, tokenize_chinese

# 訓練器
from scripts.trainer import Trainer, get_optimizer, get_scheduler

# 指標
from scripts.metrics import precision_recall_f1, bleu_score, rouge_l, ner_metrics
```

## 📦 安裝

```bash
pip install torch transformers jieba nltk scikit-learn matplotlib seaborn
```

---

*© 2026 chanxaviersy*
