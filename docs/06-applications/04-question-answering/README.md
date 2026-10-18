# ❓ 問答系統 | Question Answering (QA)

> 摘要：問答系統從文本或知識庫中檢索/生成答案。本章介紹抽取式 QA（SQuAD）、生成式 QA、多跳 QA 以及基於 RAG 和 LLM 的現代問答系統。  
> Abstract: QA systems retrieve or generate answers from text or knowledge bases. This chapter covers extractive QA, generative QA, multi-hop QA, and modern RAG/LLM-based systems.

---

## 1. QA 任務分類 | Task Taxonomy

```
問答系統 (QA)
├── 按數據來源
│   ├── 文本 QA (Text-based)
│   │   ├── 抽取式：從給定段落提取答案 span
│   │   └── 生成式：生成自由文本
│   ├── 知識庫 QA (KBQA)
│   │   └── 將問題轉化為 SPARQL 查詢
│   ├── 表格 QA (TableQA)
│   │   └── 從表格檢索答案
│   └── 多模態 QA
│       └── 結合圖像、視頻
│
├── 按對話性
│   ├── 單輪 QA：獨立問題
│   └── 多輪 QA (ConvQA)：依賴上下文
│
├── 按推理複雜度
│   ├── 單跳 QA：一步推理
│   └── 多跳 QA (Multi-hop)：多步推理
│
└── 按答案類型
    ├── 區間答案 (Span)：連續文本片段
    ├── 抽象答案 (Abstract)：自由生成
    ├── 是/否 (Yes/No)
    └── 多選 (Multiple Choice)
```

---

## 2. 抽取式 QA | Extractive QA

### 2.1 SQuAD 風格的任務

**輸入**：
- 問題 (Question)
- 段落 (Context/Passage)

**輸出**：
- 答案在段落中的**起始位置**和**結束位置**

**範例**：
```
Context: "張三是台積電的工程師，在新竹工作。"
Question: "張三在哪裡工作？"
Answer: "新竹" (start=10, end=12)
```

### 2.2 BERT 抽取式 QA

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", 
                       truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 預測 start 和 end 位置
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    # 取 top-k start 和 end，選擇最合理的組合
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)
    
    # 確保 start <= end
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    
    tokens = inputs["input_ids"][0][start_idx:end_idx+1]
    answer = tokenizer.decode(tokens, skip_special_tokens=True)
    return answer, start_idx.item(), end_idx.item()

context = "台積電是台灣最大的半導體公司，總部位於新竹科學園區。"
question = "台積電的總部在哪裡？"
answer, s, e = answer_question(question, context)
print(f"答案: {answer} (位置: {s}-{e})")
```

### 2.3 處理長文本

```python
def answer_long_context(question, context, max_length=512, stride=128):
    """處理長於 max_length 的上下文"""
    inputs = tokenizer(
        question, context, 
        return_tensors="pt",
        truncation="only_second",  # 只截斷 context
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding=True,
    )
    
    best_answer = ""
    best_score = float("-inf")
    
    for i in range(inputs["input_ids"].shape[0]):
        input_ids = inputs["input_ids"][i:i+1]
        with torch.no_grad():
            outputs = model(input_ids)
        
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        
        # 取 top 候選
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)
        score = start_logits[start_idx] + end_logits[end_idx]
        
        if score > best_score and end_idx >= start_idx:
            tokens = input_ids[0][start_idx:end_idx+1]
            best_answer = tokenizer.decode(tokens, skip_special_tokens=True)
            best_score = score
    
    return best_answer
```

---

## 3. 生成式 QA | Generative QA

### 3.1 T5 for QA

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def answer_with_t5(question, context):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        inputs["input_ids"],
        max_length=64,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

answer = answer_with_t5("台積電的總部在哪裡？", context)
print(answer)
```

### 3.2 端到端問答 LLM

```python
from openai import OpenAI

client = OpenAI()

def answer_with_llm(question, context=None, model="gpt-4o"):
    if context:
        prompt = f"""基於以下上下文回答問題。如果上下文中沒有答案，請說「無法回答」。

上下文：{context}

問題：{question}

回答："""
    else:
        prompt = question
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content
```

---

## 4. 多跳問答 | Multi-Hop QA

需要多步推理才能得出答案：

```
問題：台積電創辦人是哪個公司的董事長？

推理步驟：
1. 台積電的創辦人是誰？→ 張忠謀
2. 張忠謀是哪個公司的董事長？→ 台積電（循環！）

或者：台積電的創辦人 → 張忠謀 → 德儀董事長
```

**經典數據集**：HotpotQA、2WikiMultihopQA

```python
# 多跳 QA 的實現：分步推理
def multi_hop_qa(question, retriever, llm, max_hops=3):
    context = []
    reasoning_steps = []
    
    for hop in range(max_hops):
        # 1. 檢索相關文檔
        query = question if hop == 0 else reasoning_steps[-1]
        docs = retriever.retrieve(query, top_k=3)
        context.extend(docs)
        
        # 2. 推理下一步
        prompt = f"""問題：{question}
已收集的事實：{reasoning_steps}
新檢索到的事實：{[d.text for d in docs]}

下一步推理："""
        next_step = llm.invoke(prompt)
        reasoning_steps.append(next_step)
        
        # 3. 檢查是否已有答案
        if "答案是" in next_step or "答案：" in next_step:
            return next_step
    
    # 最終生成
    final_prompt = f"問題：{question}\n推理步驟：{reasoning_steps}\n最終答案："
    return llm.invoke(final_prompt)
```

---

## 5. 對話式 QA | Conversational QA

需要記住對話歷史：

```python
class ConversationalQA:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []
    
    def ask(self, question, context=None):
        # 構建帶歷史的問題
        history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.history])
        full_question = f"對話歷史：\n{history_str}\n\n新問題：{question}"
        
        # 檢索並回答
        if context is None:
            context = "\n".join([d.text for d in self.retriever.retrieve(question, top_k=3)])
        
        answer = answer_with_llm(full_question, context)
        
        # 更新歷史
        self.history.append((question, answer))
        return answer

qa = ConversationalQA(retriever, llm)
print(qa.ask("台積電是做什麼的？"))
print(qa.ask("它的主要客戶有哪些？"))  # 依賴上文
```

---

## 6. 知識庫 QA | Knowledge Base QA

從結構化知識庫（如 Wikidata）查詢：

```python
# 將自然語言問題轉化為 SPARQL
def question_to_sparql(question, llm):
    prompt = f"""將以下自然語言問題轉化為 SPARQL 查詢。

問題：{question}

SPARQL："""
    return llm.invoke(prompt)

# 範例
question = "台積電的創辦人是誰？"
sparql = """
SELECT ?person WHERE {
  ?company rdfs:label "台積電"@zh .
  ?company wdt:P112 ?person .
}
"""
```

---

## 7. 評估 | Evaluation

### 7.1 抽取式 QA

**Exact Match (EM)**：預測答案與標準答案**完全相同**（常做標準化）

**F1 Score**：token 級別的 F1

```python
def normalize_answer(s):
    """標準化答案"""
    import re
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = ' '.join(s.split())
    return s

def f1_score(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))
```

### 7.2 生成式 QA

| 指標 | 描述 |
|------|------|
| **BLEU/ROUGE** | n-gram 重疊 |
| **BERTScore** | 語義相似 |
| **BLEURT** | 學習評分 |
| **Human Eval** | 人工評估 |
| **FactCC** | 事實一致性 |
| **QAGS** | 基於 QA 的一致性 |

### 7.3 基準

| 基準 | 描述 | 評估 |
|------|------|------|
| **SQuAD 1.1/2.0** | Wikipedia 抽取式 QA | EM, F1 |
| **Natural Questions** | Google 搜索真實問題 | EM, F1 |
| **TriviaQA** | 常識問答 | EM, F1 |
| **HotpotQA** | 多跳 QA | EM, F1, Supp. EM/F1 |
| **DROP** | 離散推理 | EM, F1 |
| **MS MARCO** | 真實搜索問題 | Rouge-L |
| **MMLU** | 多學科選擇題 | Accuracy |

---

## 8. 現代 QA 系統 | Modern QA Systems

### 8.1 RAG-QA 系統

```python
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# 構建 RAG-QA
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vectorstore = FAISS.load_local("knowledge_base", embeddings, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
)

result = qa_chain.invoke({"query": "台積電的創辦人是誰？"})
print("答案:", result["result"])
print("來源:")
for doc in result["source_documents"]:
    print(f"  - {doc.page_content[:100]}... (來源: {doc.metadata.get('source')})")
```

### 8.2 開放域 QA

無需提供 context，模型結合內部知識回答：

```python
def open_domain_qa(question, model="gpt-4o"):
    """開放域問答 - 模型利用自身知識"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一個準確的問答助手。對於不確定的問題，請說明你不確定。"},
            {"role": "user", "content": question}
        ],
        temperature=0,
    )
    return response.choices[0].message.content

print(open_domain_qa("台積電的創辦人是誰？"))
```

**注意**：開放域 QA 容易產生幻覺，重要場景應結合 RAG。

---

## 9. 應用案例 | Use Cases

| 領域 | 應用 |
|------|------|
| **客戶服務** | FAQ 自動答覆、客服機器人 |
| **搜索引擎** | 直接答案卡片 |
| **教育** | 智能輔導、知識問答 |
| **醫療** | 醫學文獻問答、症狀查詢 |
| **法律** | 法律條文查詢、案例檢索 |
| **金融** | 財報分析、市場資訊問答 |
| **企業內部** | 員工手冊問答、知識管理 |

---

## 總結 | Summary

| 方法 | 優點 | 缺點 | 適用 |
|------|------|------|------|
| **BERT 抽取式** | 準確、快速 | 限於原文 | 已知文檔集 |
| **T5/BART** | 自由生成 | 可能不準確 | 開放問答 |
| **LLM 開放域** | 強大、零樣本 | 幻覺、成本 | 通用場景 |
| **RAG** | 事實準確、可引用 | 需維護 KB | 企業知識問答 |
| **多跳 QA** | 複雜推理 | 難實現 | 學術研究 |

**最佳實踐**：
1. 高風險場景（醫療、法律）→ **RAG + 抽取式 + 引用**
2. 開放域閒聊 → **LLM + RAG**
3. 結構化查詢 → **Text-to-SQL/SparQL**
4. 多輪對話 → **ConvQA + 歷史管理**

---

## 參考文獻 | References

1. Rajpurkar, P., et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. *EMNLP*.
2. Yang, Z., et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. *EMNLP*.
3. Kwiatkowski, T., et al. (2019). Natural Questions: A Benchmark for Question Answering Research. *TACL*.
4. Chen, D., et al. (2017). DrQA: Reading Wikipedia to Answer Open-Domain Questions. *ACL*.
5. Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP*.

---

*© 2026 chanxaviersy*
