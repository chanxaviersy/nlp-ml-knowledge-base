# ✍️ 文本生成 | Text Generation

> 摘要：文本生成是 LLM 的核心能力，涵蓋從對話、創意寫作到結構化內容生成。本章介紹自回歸生成、解碼策略、條件控制以及評估方法。  
> Abstract: Text generation is the core capability of LLMs, covering dialogue, creative writing, and structured content generation. This chapter covers autoregressive generation, decoding strategies, controlled generation, and evaluation.

---

## 1. 文本生成任務分類 | Task Taxonomy

```
文本生成 (Text Generation)
├── 開放式生成
│   ├── 創意寫作（故事、詩歌、劇本）
│   ├── 對話（聊天、客服）
│   └── 自由問答
│
├── 結構化生成
│   ├── 摘要、報告
│   ├── 翻譯
│   ├── 數據到文本 (Data-to-Text)
│   └── 結構化輸出 (JSON, XML, SQL)
│
├── 條件生成
│   ├── 指令遵循 (Instruction Following)
│   ├── 風格轉換 (Style Transfer)
│   └── 屬性控制 (Attribute Control)
│
└── 特殊任務
    ├── 代碼生成
    ├── 數學推理
    └── 多模態生成
```

---

## 2. 自回歸生成 | Autoregressive Generation

### 2.1 基本原理

$$P(y_1, y_2, ..., y_n | x) = \prod_{t=1}^{n} P(y_t | y_{<t}, x)$$

每個 token 基於已生成的 token 預測下一個 token。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate_text("Once upon a time"))
```

---

## 3. 解碼策略 | Decoding Strategies

### 3.1 Greedy Search

每步選概率最高的 token：

```python
outputs = model.generate(
    **inputs,
    max_length=100,
    do_sample=False,  # Greedy
    num_beams=1,
)
```

**缺點**：容易陷入重複（"the the the..."）

### 3.2 Beam Search

保留 top-k 條路徑：

```python
outputs = model.generate(
    **inputs,
    max_length=100,
    num_beams=5,  # 5 條 beam
    no_repeat_ngram_size=2,  # 避免重複
    early_stopping=True,
)
```

**優點**：通常比 Greedy 質量高  
**缺點**：容易產生通用、安全的回覆，缺少驚喜

### 3.3 Sampling

從概率分佈中隨機採樣：

```python
outputs = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
```

**Temperature** 控制分佈尖銳度：
- $T < 1$：分佈更尖銳，更確定
- $T > 1$：分佈更平滑，更隨機
- $T \to 0$：等同 greedy
- $T \to \infty$：均勻隨機

```python
def apply_temperature(logits, temperature=1.0):
    """調整溫度"""
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=-1)
```

**Top-K Sampling**：只從概率最高的 K 個 token 中採樣

**Top-p (Nucleus) Sampling**：從概率累積達到 p 的最小集合中採樣

```python
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """過濾 logits"""
    # Top-K
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    # Top-P
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    
    return logits
```

### 3.4 對比搜索 (Contrastive Search)

```python
outputs = model.generate(
    **inputs,
    max_length=100,
    penalty_alpha=0.6,  # 對比搜索懲罰
    top_k=4,
)
```

避免重複的同時保持高質量。

### 3.5 策略對比

| 策略 | 質量 | 多樣性 | 速度 | 適用 |
|------|------|--------|------|------|
| **Greedy** | 中 | 低 | 快 | 簡單任務 |
| **Beam** | 高 | 低 | 中 | 翻譯、摘要 |
| **Sampling (T=1)** | 中 | 高 | 中 | 創意寫作 |
| **Top-K** | 中 | 中 | 中 | 通用 |
| **Top-p** | 中高 | 中高 | 中 | 通用 |
| **Contrastive** | 高 | 中 | 慢 | 高質量生成 |

---

## 4. 條件生成 | Controlled Generation

### 4.1 Prompt Engineering

```python
# 角色 + 格式 + 範例
prompt = """你是一個專業的文案寫手。請為以下產品撰寫 100 字的行銷文案，採用年輕活潑的語氣。

產品：智慧型空氣清淨機
特點：AI 自動感測、靜音設計、CADR 400

文案："""
```

### 4.2 結構化輸出 (JSON)

```python
from pydantic import BaseModel
import json

class ProductInfo(BaseModel):
    name: str
    price: float
    features: list[str]

# 方法 1：Function Calling
def extract_product_info(text, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Extract product information as JSON."},
            {"role": "user", "content": text}
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "extract_product",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number"},
                        "features": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        }],
        tool_choice={"type": "function", "function": {"name": "extract_product"}}
    )
    return json.loads(response.choices[0].message.tool_calls[0].function.arguments)
```

### 4.3 風格控制

```python
def generate_with_style(text, target_style, model="gpt-4o"):
    prompt = f"""將以下文本改寫為 {target_style} 風格，保留原意。

原文：{text}

{target_style} 風格："""
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content

# 風格轉換
original = "The product has good performance and the price is reasonable."
formal = generate_with_style(original, "正式學術")
casual = generate_with_style(original, "口語輕鬆")
poetic = generate_with_style(original, "詩意")
```

### 4.4 屬性控制 (Classifier Guidance)

訓練一個分類器來引導生成：

```python
class ControlledGenerator:
    """使用分類器引導生成"""
    
    def __init__(self, model, classifier, weight=2.0):
        self.model = model
        self.classifier = classifier
        self.weight = weight
    
    def generate(self, prompt, target_class):
        # 1. 正常生成下一個 token
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model(input_ids)
        logits = outputs.logits[0, -1]
        
        # 2. 分類器對每個候選 token 的引導
        # 假設我們能計算每個 token 對分類結果的影響
        # 簡化：直接對最終輸出做調整
        # 實際實現需要基於梯度的引導
        
        # ... 完整實現較複雜
        return generated_text
```

---

## 5. 推理型生成 | Reasoning Generation

### 5.1 Chain-of-Thought (CoT)

```python
COT_PROMPT = """問題：{question}

讓我們一步步思考。

步驟 1："""

def generate_with_cot(question, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"{COT_PROMPT.format(question=question)}\n\n答案："
        }],
        temperature=0,
    )
    return response.choices[0].message.content
```

### 5.2 Self-Consistency

生成多個推理路徑，投票決定：

```python
def self_consistency(question, n_samples=5, model="gpt-4o"):
    answers = []
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"問題：{question}\n讓我們一步步思考。"}],
            temperature=0.7,
        )
        answer = extract_final_answer(response.choices[0].message.content)
        answers.append(answer)
    
    # 投票
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common
```

### 5.3 Tree of Thoughts (ToT)

```python
def tree_of_thoughts(problem, n_branches=3, depth=3, model="gpt-4o"):
    """樹狀搜索推理"""
    
    def expand(node, current_depth):
        if current_depth == depth:
            return node
        
        # 生成多個下一步
        branches = []
        for _ in range(n_branches):
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user", 
                    "content": f"問題：{problem}\n當前思路：{node}\n下一步："
                }],
                temperature=0.7,
            )
            next_step = response.choices[0].message.content
            branches.append(node + "\n" + next_step)
        
        # 評估每個分支
        best_branch = max(branches, key=evaluate_thought)
        return expand(best_branch, current_depth + 1)
    
    return expand("", 0)
```

---

## 6. 評估 | Evaluation

### 6.1 自動指標

| 指標 | 衡量 |
|------|------|
| **BLEU/ROUGE** | 與參考答案的 n-gram 重疊 |
| **BERTScore** | 語義相似度 |
| **BLEURT** | 學習式評分 |
| **BARTScore** | 基於 BART 的生成評分 |
| **Perplexity** | 語言模型困惑度 |

### 6.2 LLM-as-Judge

```python
def llm_evaluate(generated, reference, criteria="quality", model="gpt-4o"):
    """使用 LLM 評估生成質量"""
    prompt = f"""請評估以下生成文本的質量。

評估維度：{criteria}
參考答案：{reference}
生成文本：{generated}

評分（1-10）：
理由："""
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content
```

### 6.3 人工評估維度

| 維度 | 評估內容 |
|------|----------|
| **流暢性 (Fluency)** | 語言是否自然通順 |
| **連貫性 (Coherence)** | 邏輯是否連貫 |
| **相關性 (Relevance)** | 是否切題 |
| **事實性 (Factuality)** | 是否準確無誤 |
| **創造性 (Creativity)** | 新穎程度 |
| **有用性 (Helpfulness)** | 對用戶的幫助 |
| **安全性 (Safety)** | 無有害內容 |

### 6.4 基準測試

| 基準 | 評估內容 |
|------|----------|
| **MMLU** | 多任務語言理解 |
| **HumanEval** | 代碼生成 |
| **GSM8K** | 數學推理 |
| **TruthfulQA** | 真實性 |
| **MT-Bench** | 多輪對話 |
| **AlpacaEval** | 指令遵循 |
| **Chatbot Arena** | 人類偏好對戰 |

---

## 7. 應用場景 | Applications

| 應用 | 描述 |
|------|------|
| **內容創作** | 文章、故事、詩歌 |
| **代碼生成** | Copilot, Cursor |
| **行銷文案** | 廣告、社交媒體 |
| **客服對話** | 自動回覆、智能助手 |
| **教育** | 解題、解釋 |
| **遊戲** | NPC 對話、劇情 |
| **研究** | 文獻綜述、數據分析 |
| **設計** | UI 文案、品牌口號 |

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│  文本生成最佳實踐                                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 選擇合適的解碼策略                                        │
│     ├── 翻譯/摘要 → Beam Search                             │
│     ├── 創意寫作 → Top-p + 較高溫度                          │
│     └── 問答/事實 → Greedy 或低溫度                          │
│                                                              │
│  2. 提示工程                                                  │
│     ├── 明確角色和任務                                        │
│     ├── 給予格式和範例                                        │
│     ├── 思維鏈 (CoT) 提升推理                                │
│     └── 結構化輸出 (JSON, XML)                               │
│                                                              │
│  3. 控制生成                                                  │
│     ├── 溫度控制隨機性                                        │
│     ├── 函數調用確保格式                                      │
│     └── RAG 提供事實基礎                                      │
│                                                              │
│  4. 評估生成質量                                              │
│     ├── 自動指標 + LLM-as-Judge                              │
│     ├── 人工評估                                              │
│     └── A/B 測試對比                                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 參考文獻 | References

1. Holtzman, A., et al. (2020). The Curious Case of Neural Text Degeneration. *ICLR* (Top-p Sampling).
2. Welleck, S., et al. (2020). Neural Text Generation with Unlikelihood Training. *ICLR*.
3. Wei, J., et al. (2022). Chain-of-Thought Prompting. *NeurIPS*.
4. Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR*.
5. Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *NeurIPS*.

---

*© 2026 chanxaviersy*
