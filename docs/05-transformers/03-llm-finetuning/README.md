# 🎯 大語言模型微調 | LLM Fine-tuning

> 摘要：完整微調一個大語言模型成本高昂，PEFT（Parameter-Efficient Fine-Tuning）方法通過只更新少量參數來實現高效適配。本章介紹 LoRA、QLoRA、Prefix Tuning 等主流方法及 RLHF、對齊技術。  
> Abstract: Full fine-tuning of large language models is prohibitively expensive. PEFT methods adapt LLMs by updating only a small subset of parameters. This chapter covers LoRA, QLoRA, Prefix Tuning, and alignment techniques like RLHF.

---

## 1. 為什麼需要 PEFT？| Why PEFT?

**全量微調的成本**（以 LLaMA-7B 為例）：

| 項目 | 大小 |
|------|------|
| 參數量 | 7B (28GB, FP32) |
| 訓練記憶體（Adam 優化器） | ~4× 參數量 = 112GB |
| 訓練 GPU | 至少 4× A100 (80GB) |
| 訓練時間 | 數天到數週 |

**PEFT 的目標**：訓練 < 1% 的參數，達到接近全量微調的效果。

| 方法 | 訓練參數 | 推理延遲增加 | 適用場景 |
|------|----------|------------|----------|
| **LoRA** | 0.1-1% | 無（可合併） | 通用首選 |
| **QLoRA** | 0.1-1% | 無 | 4-bit 量化大模型 |
| **Adapter** | 1-5% | 略增 | 多任務快速切換 |
| **Prefix Tuning** | 0.1% | 略增 | 生成任務 |
| **Prompt Tuning** | 0.01% | 無 | 極低資源 |

---

## 2. LoRA (Low-Rank Adaptation)

### 2.1 核心思想

**關鍵發現**：模型微調時權重變化矩陣 $\Delta W$ 是「低秩」的。

$$\Delta W = B A, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k)$$

$$W' = W + \Delta W = W + BA$$

```
原權重:  W ∈ ℝ^(d × k)  (d × k 參數)
低秩分解: BA, B ∈ ℝ^(d × r), A ∈ ℝ^(r × k)  (r(d + k) 參數)
當 r << min(d, k) 時, 參數大幅減少
```

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """LoRA 包裝的線性層"""
    
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.05):
        super().__init__()
        # 凍結的原始權重
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        # LoRA 參數（可訓練）
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
    
    def forward(self, x):
        # 原始路徑（凍結）
        original = torch.nn.functional.linear(x, self.weight)
        # LoRA 路徑
        lora = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return original + lora * self.scaling
```

### 2.2 超參數選擇

| 超參數 | 影響 | 推薦值 |
|--------|------|--------|
| **r (秩)** | 控制參數量 | 4-64，常用 8-16 |
| **α (縮放)** | LoRA 影響強度 | 通常 2× r |
| **target_modules** | 應用於哪些層 | q_proj, v_proj（最少）<br>q, k, v, o（更好）<br>q, k, v, o, gate, up, down（最全）|
| **dropout** | 正則化 | 0.05-0.1 |

### 2.3 完整微調示例

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

# 1. 載入模型
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,  # QLoRA 量化
)
tokenizer.pad_token = tokenizer.eos_token

# 2. 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                  # 秩
    lora_alpha=32,         # 縮放
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 輸出範例：trainable params: 13,631,488 || all params: 6,738,415,616 || trainable%: 0.2023

# 3. 準備數據
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""

# 4. 訓練
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=512,
    formatting_func=format_prompt,
)

trainer.train()

# 5. 保存 LoRA 權重（只有幾 MB）
model.save_pretrained("./lora_adapter")
```

### 2.4 合併 LoRA 權重

訓練完成後，可以將 LoRA 權重合併回原始模型：

```python
# 合併並保存完整模型
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")

# 推理時不需要額外開銷
from transformers import pipeline
generator = pipeline("text-generation", model="./merged_model", tokenizer=tokenizer)
```

---

## 3. QLoRA (Quantized LoRA)

QLoRA 在 LoRA 基礎上加入 4-bit 量化：

| 創新 | 描述 |
|------|------|
| **4-bit NormalFloat (NF4)** | 適合常態分佈的 4-bit 數據類型 |
| **雙重量化 (Double Quantization)** | 對量化常數再量化 |
| **Paged Optimizers** | 用 NVIDIA 統一記憶體防止 OOM |
| **LoRA Adapter** | 在量化模型上微調 LoRA |

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 量化
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,    # 雙重量化
    llm_int8_enable_fp32_cpu_offload=True,  # CPU offload
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

**單 GPU 訓練 LLaMA-65B**：QLoRA 只需要約 48GB VRAM（A100）。

---

## 4. 對齊技術：RLHF 與 DPO

### 4.1 為什麼需要對齊？

預訓練的 LM 只是「會接話」，但未必：
- 遵循指令 (Instruction Following)
- 拒絕有害請求
- 提供有用、無害、誠實的回答 (HHH)

### 4.2 RLHF (Reinforcement Learning from Human Feedback)

**三步驟流程**：

```
┌────────────────────────────────────────────────────────────┐
│  Step 1: 監督微調 (SFT)                                    │
│  用人工編寫的高質量 (指令, 回覆) 對微調 LLM                 │
└─────────────────┬──────────────────────────────────────────┘
                  ↓
┌────────────────────────────────────────────────────────────┐
│  Step 2: 訓練獎勵模型 (Reward Model)                       │
│  對同一 prompt 採樣多個回覆，人工排序                       │
│  訓練 RM 學習 "好回答" 的特徵                              │
└─────────────────┬──────────────────────────────────────────┘
                  ↓
┌────────────────────────────────────────────────────────────┐
│  Step 3: PPO 強化學習                                      │
│  用 RM 作為獎勵函數，用 PPO 優化 LM 的策略                  │
└────────────────────────────────────────────────────────────┘
```

### 4.3 DPO (Direct Preference Optimization)

DPO 直接從偏好數據學習，省去 RM 和 PPO：

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]$$

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,
    learning_rate=1e-6,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,           # 參考模型
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()
```

**RLHF vs DPO 對比**：

| 特性 | RLHF | DPO |
|------|------|-----|
| 階段數 | 3 階段（SFT → RM → PPO） | 2 階段（SFT → DPO） |
| 獎勵模型 | 必需 | 不需要 |
| 訓練穩定性 | 較不穩定 | 穩定 |
| 效果 | 高 | 略低 |
| 實現複雜度 | 高 | 低 |

### 4.4 其他對齊方法

| 方法 | 核心思想 |
|------|----------|
| **ORPO** | 結合 SFT 和 DPO，無需參考模型 |
| **KTO** | 從單個反饋（好/壞）學習，無需成對 |
| **IPO** | DPO 的正則化版本 |
| **SimPO** | 簡化版 DPO，使用長度歸一化 |

---

## 5. 指令微調 (Instruction Tuning)

### 5.1 數據集格式

```json
{
  "instruction": "將以下英文翻譯成中文",
  "input": "Hello, how are you?",
  "output": "你好，你好嗎？"
}
```

### 5.2 高質量數據集

| 數據集 | 規模 | 語言 |
|--------|------|------|
| Alpaca | 52K | EN |
| Dolly 15K | 15K | EN |
| ShareGPT | 90K | 中英文 |
| BELLE | 2M+ | 中文 |
| WizardLM | 250K | EN |
| UltraChat | 1.5M | 中英文 |
| OpenOrca | 4M+ | EN |

### 5.3 中文指令微調

```python
from datasets import load_dataset

# 載入中文指令數據
dataset = load_dataset("Chinese-Vicuna/guanaco_belle_merge", split="train")
print(dataset[0])

# 或使用 BELLE 數據集
belle = load_dataset("BelleGroup/train_3.5M_CN", split="train[:1000]")
```

---

## 6. Prompt Engineering vs Fine-tuning

| 方法 | 成本 | 效果 | 適用場景 |
|------|------|------|----------|
| **Prompt Engineering** | 極低 | 中 | 通用任務、快速原型 |
| **Few-shot Prompting** | 低 | 中高 | 簡單特定任務 |
| **RAG** | 中 | 高 | 知識密集型任務 |
| **LoRA 微調** | 中高 | 高 | 領域適配、複雜格式 |
| **全量微調** | 極高 | 最高 | 大規模定制 |

**推薦策略**：
1. 從 Prompt Engineering 開始
2. 效果不夠 → 加 RAG
3. 仍不夠 → LoRA 微調
4. 仍不夠 → 全量微調（考慮預算）

---

## 7. 評估微調效果 | Evaluation

### 7.1 自動評估

```python
from lm_eval import simple_evaluate

# 在標準基準上評估
results = simple_evaluate(
    model=model,
    tasks=["mmlu", "hellaswag", "arc_easy"],
    batch_size=8,
)
print(results["results"])
```

### 7.2 人工評估

| 維度 | 評估內容 |
|------|----------|
| **有用性 (Helpfulness)** | 是否有效解決問題 |
| **無害性 (Harmlessness)** | 是否包含有害內容 |
| **誠實性 (Honesty)** | 是否準確、無編造 |
| **指令遵循 (Instruction Following)** | 是否按要求格式/風格回答 |

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│  LLM 適配方法選擇流程                                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  預訓練 LLM (LLaMA, Qwen, Mistral)                          │
│       │                                                      │
│       ├── 少量資料 + 通用任務 → Few-shot Prompt              │
│       ├── 領域知識 → RAG + Prompt                            │
│       ├── 特定格式/風格 → LoRA (8-32 rank)                   │
│       ├── 極大模型 (70B+) → QLoRA (4-bit)                    │
│       ├── 對齊人類偏好 → DPO/ORPO                          │
│       └── 大規模定制 → 全量微調 (SFT)                        │
│                                                              │
│  關鍵技巧：                                                  │
│  1. 高質量 > 大量：1K 高質數據 > 100K 噪音數據              │
│  2. 學習率要小：1e-4 ~ 5e-5                                 │
│  3. 評估先行：先建立評估集再訓練                             │
│  4. 小步快跑：先 LoRA r=8 實驗，再擴大                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 參考文獻 | References

1. Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.
2. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS*.
3. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS* (InstructGPT).
4. Rafailov, R., et al. (2023). Direct Preference Optimization. *NeurIPS*.
5. Hong, J., et al. (2024). ORPO: Monolithic Preference Optimization without Reference Model.

---

*© 2026 chanxaviersy*
