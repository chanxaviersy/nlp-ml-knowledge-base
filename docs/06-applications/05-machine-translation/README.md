# 🌐 機器翻譯 | Machine Translation (MT)

> 摘要：機器翻譯將一種語言的文本自動轉換為另一種語言。本章介紹 NMT 架構、Transformer MT、低資源翻譯、多語言模型以及現代 LLM 在翻譯中的應用。  
> Abstract: Machine translation automatically converts text between languages. This chapter covers NMT architectures, Transformer MT, low-resource translation, multilingual models, and modern LLM-based translation.

---

## 1. MT 發展史 | History

| 時期 | 方法 | 代表系統 |
|------|------|----------|
| 1950s-1980s | 規則方法（Rule-based） | Systran, METEO |
| 1990s-2010s | 統計方法（SMT） | Moses, Google Translate (初期) |
| 2014-2017 | 神經機器翻譯（RNN-based） | Google NMT (2016) |
| 2017+ | Transformer MT | Google, DeepL, GPT |
| 2023+ | LLM Prompting | GPT-4, Claude, Gemini |

---

## 2. 統計機器翻譯 (SMT)

### 2.1 噪聲信道模型

$$\hat{e} = \arg\max_e P(e|f) = \arg\max_e P(f|e) P(e)$$

- $P(e)$：語言模型（流暢度）
- $P(f|e)$：翻譯模型（忠實度）

### 2.2 詞對齊

```python
# IBM Model 1 的簡化實現
def ibm_model1(corpus, num_iterations=10):
    """IBM Model 1: 對齊機率"""
    # 初始化均勻分佈
    translation_prob = defaultdict(lambda: 1.0 / len(vocabulary_f))
    
    for iteration in range(num_iterations):
        # E-step: 計算期望對齊
        counts = defaultdict(float)
        total_f = defaultdict(float)
        
        for e_sent, f_sent in corpus:
            for e in e_sent:
                for f in f_sent:
                    counts[(f, e)] += translation_prob[(f, e)] / sum(
                        translation_prob[(f, e2)] for e2 in e_sent
                    )
                    total_f[f] += translation_prob[(f, e)] / sum(
                        translation_prob[(f, e2)] for e2 in e_sent
                    )
        
        # M-step: 更新翻譯機率
        for f, e in counts:
            translation_prob[(f, e)] = counts[(f, e)] / total_f[f]
    
    return translation_prob
```

---

## 3. 神經機器翻譯 (NMT)

### 3.1 Seq2Seq 架構

```
源語言序列 (Source)       目標語言序列 (Target)
   x₁ x₂ x₃ x₄    →    y₁ y₂ y₃
       ↓                    ↑
   [編碼器 Encoder]    [解碼器 Decoder]
       ↓                    ↑
     c (上下文向量)
```

### 3.2 Bahdanau Attention (2014)

```python
class Seq2SeqWithAttention(nn.Module):
    """Bahdanau 注意力 Seq2Seq"""
    
    def __init__(self, src_vocab, tgt_vocab, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.encoder = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(embed_dim + 2*hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 3, 1)
        self.src_embed = nn.Embedding(src_vocab, embed_dim)
        self.tgt_embed = nn.Embedding(tgt_vocab, embed_dim)
        self.output = nn.Linear(hidden_dim, tgt_vocab)
    
    def forward(self, src, tgt):
        # 編碼
        src_emb = self.src_embed(src)
        enc_out, (h, c) = self.encoder(src_emb)  # (batch, src_len, 2*hidden)
        
        # 解碼（簡化）
        decoder_hidden = (h[:1], c[:1])  # 取一個方向
        outputs = []
        
        for t in range(tgt.size(1)):
            tgt_emb_t = self.tgt_embed(tgt[:, t:t+1])
            
            # 計算 attention
            combined = torch.cat([decoder_hidden[0].squeeze(0).unsqueeze(1).expand_as(enc_out), 
                                  enc_out], dim=-1)
            attn_scores = self.attention(combined).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_out)  # (batch, 1, 2*hidden)
            
            # 解碼
            dec_input = torch.cat([tgt_emb_t, context], dim=-1)
            dec_out, decoder_hidden = self.decoder(dec_input, decoder_hidden)
            outputs.append(self.output(dec_out))
        
        return torch.cat(outputs, dim=1)
```

### 3.3 Transformer MT

```python
from transformers import MarianMTModel, MarianTokenizer

# 預訓練翻譯模型
model_name = "Helsinki-NLP/opus-mt-zh-en"  # 中 → 英
# 或 "Helsinki-NLP/opus-mt-en-zh"          # 英 → 中
# 或 "Helsinki-NLP/opus-mt-zh-tw"          # 簡中 → 繁中

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text, model, tokenizer, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, num_beams=4, max_length=max_length, early_stopping=True)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

chinese_text = "機器翻譯是自然語言處理的重要應用。"
english = translate(chinese_text, model, tokenizer)
print(f"中文: {chinese_text}")
print(f"英文: {english}")
```

---

## 4. 多語言翻譯 | Multilingual Translation

### 4.1 M2M-100 (Meta)

100 語言之間直接翻譯，無需以英語為橋樑：

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

tokenizer.src_lang = "zh"
encoded = tokenizer("機器翻譯讓世界更近。", return_tensors="pt")
generated_tokens = model.generate(
    **encoded,
    forced_bos_token_id=tokenizer.get_lang_id("ja"),  # 日文
)
print(tokenizer.decode(generated_tokens[0], skip_special_tokens=True))
```

### 4.2 NLLB (No Language Left Behind, Meta)

支持 200 種語言，包括低資源語言：

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# 設置源語言和目標語言（FLORES-200 代碼）
tokenizer.src_lang = "zho_Hans"  # 簡體中文
inputs = tokenizer("台灣是個美麗的島嶼。", return_tensors="pt")

# 強制翻譯為繁體中文
forced_bos = tokenizer.convert_tokens_to_ids("zho_Hant")
outputs = model.generate(**inputs, forced_bos_token_id=forced_bos)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 5. 大語言模型翻譯 | LLM-based Translation

### 5.1 GPT-4 翻譯

```python
from openai import OpenAI

client = OpenAI()

def llm_translate(text, target_lang="English", model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "system",
            "content": f"You are a professional translator. Translate the following text into {target_lang}. Only output the translation."
        }, {
            "role": "user",
            "content": text
        }],
        temperature=0,  # 翻譯需要確定性
    )
    return response.choices[0].message.content.strip()

# 中 → 英
chinese = "人工智慧正在改變世界，帶來前所未有的機遇。"
english = llm_translate(chinese, "English")
print(english)
```

### 5.2 Few-shot 翻譯

```python
FEW_SHOT_TRANSLATION = """將以下中文翻譯為英文。

範例 1：
中文：今天天氣真好。
英文：The weather is really nice today.

範例 2：
中文：我喜歡學習新的語言。
英文：I enjoy learning new languages.

現在翻譯：
中文：{text}
英文："""

def few_shot_translate(text, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": FEW_SHOT_TRANSLATION.format(text=text)}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
```

### 5.3 術語控制翻譯

```python
def controlled_translate(text, glossary, target_lang="English"):
    """使用術語表控制翻譯"""
    glossary_str = "\n".join([f'"{k}" → "{v}"' for k, v in glossary.items()])
    
    prompt = f"""使用以下術語表翻譯文本。術語表中的翻譯必須嚴格遵守。

術語表：
{glossary_str}

文本：{text}

{target_lang}："""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# 範例
glossary = {"Transformer": "Transformer", "注意力機制": "attention mechanism"}
text = "Transformer 的核心是注意力機制。"
print(controlled_translate(text, glossary))
```

---

## 6. 低資源翻譯 | Low-Resource Translation

### 6.1 挑戰

平行語料稀缺（< 10K 句對）時傳統 NMT 表現差。

### 6.2 解決方案

| 方法 | 描述 |
|------|------|
| **Back-translation** | 用目標→源模型生成偽平行語料 |
| **Transfer Learning** | 從高資源語言遷移 |
| **Multilingual Models** | 多語言聯合訓練 |
| **Zero-shot** | 模型從未見過的語言對直接翻譯 |
| **Pivot Translation** | 經過中間語言（如英語）翻譯 |
| **Unsupervised MT** | 僅有單語資料 |

```python
# Back-translation
def back_translate(model, monolingual_target_texts, src_tokenizer, tgt_tokenizer):
    """將目標語言單語資料翻譯回源語言，生成偽平行對"""
    pseudo_pairs = []
    for text in monolingual_target_texts:
        # 目標 → 源
        inputs = tgt_tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            src_text = model.generate(**inputs, max_length=128)
        src_text = src_tokenizer.decode(src_text[0], skip_special_tokens=True)
        pseudo_pairs.append((src_text, text))
    return pseudo_pairs
```

---

## 7. 評估指標 | Evaluation

### 7.1 BLEU Score

**n-gram 精度的幾何平均**：

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

reference = [["the", "cat", "is", "on", "the", "mat"]]
candidate = ["the", "cat", "is", "on", "mat"]

# 單句 BLEU
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(f"BLEU: {score:.4f}")
```

### 7.2 chrF

字符級 F1，對形態豐富的語言更友好：

```python
from sacrebleu.metrics import CHRF

chrf = CHRF()
score = chrf.corpus_score(["the cat is on mat"], [["the cat is on the mat"]])
print(f"chrF: {score.score:.2f}")
```

### 7.3 COMET

基於神經網路的現代評估指標：

```python
# pip install unbabel-comet
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

data = [
    {"src": "機器翻譯是人工智慧的應用。", 
     "mt": "Machine translation is an application of AI.", 
     "ref": "Machine translation is an application of artificial intelligence."}
]
scores = model.predict(data, batch_size=8, gpus=0)
print(f"COMET: {scores.scores[0]:.4f}")
```

### 7.4 指標對比

| 指標 | 範圍 | 優點 | 缺點 |
|------|------|------|------|
| **BLEU** | [0, 1] | 經典、廣泛使用 | 不考慮同義詞 |
| **chrF** | [0, 100] | 字符級 | 仍舊 n-gram |
| **TER** | [0, ∞] | 編輯距離 | 直觀但粗 |
| **COMET** | [-1, 1] | 與人類評分高度相關 | 需預訓練模型 |
| **BLEURT** | [0, 1] | 學習式指標 | 訓練成本高 |

---

## 8. 翻譯記憶與術語庫 | Translation Memory & Terminology

```python
# 簡單的翻譯記憶系統
class TranslationMemory:
    def __init__(self):
        self.memory = {}  # (src_text_hash): (translation, metadata)
    
    def add(self, source, target, domain="general"):
        self.memory[source] = {"target": target, "domain": domain}
    
    def lookup(self, source, threshold=0.8):
        """模糊匹配"""
        from difflib import SequenceMatcher
        for src, entry in self.memory.items():
            similarity = SequenceMatcher(None, source, src).ratio()
            if similarity >= threshold:
                return entry["target"], similarity
        return None, 0
    
    def apply_glossary(self, source, glossary):
        """術語替換"""
        translated = source
        for src_term, tgt_term in glossary.items():
            translated = translated.replace(src_term, tgt_term)
        return translated

tm = TranslationMemory()
tm.add("人工智慧", "Artificial Intelligence")
tm.add("機器學習", "Machine Learning")

result, score = tm.lookup("人工智慧")
print(f"匹配度 {score:.2f}: {result}")
```

---

## 總結 | Summary

| 方法 | 優點 | 缺點 | 適用 |
|------|------|------|------|
| **規則 MT** | 可控 | 開發成本高 | 專門領域 |
| **統計 MT** | 可解釋 | 需要大量平行語料 | 已被淘汰 |
| **NMT (Transformer)** | 質量高 | 需要平行語料 | 主流 |
| **Multilingual NMT** | 多語言共享 | 容量限制 | 多語言場景 |
| **LLM Translation** | 零樣本強 | 成本高、不穩定 | 小批量、長尾語言 |
| **RAG + MT** | 術語可控 | 複雜 | 專業領域 |

**最佳實踐**：
1. **通用翻譯**：NLLB-200 或商用 API (DeepL, Google)
2. **高質量專門領域**：微調 NMT + 術語庫
3. **罕見語言**：LLM Few-shot
4. **生產環境**：商用 API + 自定義後處理

---

## 參考文獻 | References

1. Bahdanau, D., et al. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*.
2. Vaswani, A., et al. (2017). Attention Is All You Need (Transformer). *NeurIPS*.
3. Johnson, M., et al. (2017). Google's Multilingual Neural Machine Translation System. *EMNLP*.
4. Fan, A., et al. (2021). Beyond English-Centric Multilingual Machine Translation. *JMLR* (M2M-100).
5. Costa-jussà, M. R., et al. (2022). No Language Left Behind (NLLB). *arXiv*.
6. Papineni, K., et al. (2002). BLEU. *ACL*.

---

*© 2026 chanxaviersy*
