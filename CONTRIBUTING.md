# 貢獻指南 | Contributing Guide

歡迎來到 NLP & ML 知識庫的貢獻流程！  
Welcome to the contribution guide for the NLP & ML Knowledge Base!

## 如何貢獻 | How to Contribute

### 1. Fork & Clone

```bash
# Fork on GitHub first, then:
git clone https://github.com/YOUR_USERNAME/nlp-ml-knowledge-base.git
cd nlp-ml-knowledge-base
git remote add upstream https://github.com/chanxaviersy/nlp-ml-knowledge-base.git
```

### 2. 創建分支 | Create a Branch

```bash
git checkout -b feature/add-transformer-tutorial-zh
# or
git checkout -b fix/correct-math-notation
```

### 3. 撰寫內容 | Write Content

- 所有文檔使用 **Markdown** (.md) 格式
- 中英文雙語：先用繁體中文撰寫，再附英文版本
- 數學公式使用 LaTeX 語法：`$...$` 行內，`$$...$$` 區塊
- 代碼區塊標明語言：` ```python `, ` ```bash `, ` ```yaml `
- 每個文檔頂部包含中英文摘要

### 4. 文檔格式規範 | Document Format

```markdown
# 標題 | Title

> 摘要：本文介紹...  
> Abstract: This article introduces...

## 第一節 | Section 1

內容...

## 參考文獻 | References
```

### 5. 提交 & 推送 | Commit & Push

```bash
git add .
git commit -m "feat: add transformer tutorial in Chinese and English"
git push origin feature/your-feature-name
```

### 6. 提交 Pull Request

在 GitHub 上發起 PR，描述你的改動內容。

## 文檔結構 | Document Structure

```
docs/
├── 01-foundations/       # 基礎知識
├── 02-machine-learning/  # 機器學習
├── 03-deep-learning/     # 深度學習
├── 04-nlp-fundamentals/  # NLP 基礎
├── 05-transformers/      # Transformer
├── 06-applications/       # 應用場景
├── 07-advanced-topics/    # 前沿專題
└── 08-labs/              # 實驗室實踐
```

## 代碼規範 | Code Standards

- 遵循 **PEP 8** (Python)
- 所有代碼需包含說明註釋
- 實驗室代碼需有 `requirements.txt` 依賴清單
- 優先使用 `typing` 進行類型註釋

---

*一起構建最好的 NLP & ML 學習資源！*  
*Let's build the best NLP & ML learning resource together!*
