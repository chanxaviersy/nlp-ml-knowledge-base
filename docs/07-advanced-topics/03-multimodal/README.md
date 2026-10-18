# 🖼️ 多模態學習 | Multimodal Learning

> 摘要：多模態學習結合文本、圖像、音頻、視頻等多種感官信息，使 AI 能像人類一樣綜合理解世界。本章介紹視覺-語言模型（CLIP、BLIP）、圖文生成（DALL-E、Stable Diffusion）以及多模態大模型（GPT-4V、Gemini）。  
> Abstract: Multimodal learning combines text, image, audio, and video to enable AI to understand the world holistically. This chapter covers vision-language models, image generation, and multimodal LLMs.

---

## 1. 多模態概覽 | Overview

### 1.1 模態與任務

```
┌──────────────────────────────────────────────────────────────────┐
│                      Multimodal Landscape                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input Modalities        Tasks              Output Modalities   │
│   ────────────────        ─────              ─────────────────   │
│                                                                  │
│   📝 Text           ─→    VQA          ─→   📝 Text              │
│   🖼️ Image          ─→    Captioning   ─→   📝 Text              │
│   🖼️ Image          ─→    Generation   ─→   🖼️ Image             │
│   📝 Text           ─→    Generation   ─→   🖼️ Image             │
│   🎬 Video          ─→    Captioning   ─→   📝 Text              │
│   🎵 Audio          ─→    ASR          ─→   📝 Text              │
│   📝+🖼️            ─→    Reasoning    ─→   📝 Text              │
│                                                                  │
│   Vision-Language Models (VLMs)                                │
│   ├── CLIP / ALIGN          — 對比學習                          │
│   ├── BLIP / BLIP-2         — 引導式預訓練                      │
│   ├── LLaVA / MiniGPT-4     — 多模態 LLM                       │
│   ├── GPT-4V / Gemini       — 商業旗艦                          │
│   └── Qwen-VL / InternVL   — 開源 SOTA                          │
│                                                                  │
│   Generative Models                                           │
│   ├── Stable Diffusion      — Latent Diffusion                  │
│   ├── DALL-E 3 / Imagen     — 商業 SOTA                        │
│   ├── Midjourney            — 商業藝術                        │
│   └── Sora / Veo            — 視頻生成                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 為什麼需要多模態？

- **真實世界是多模態的**：人類通過視覺、聽覺、語言、觸覺理解世界
- **互補性**：不同模態提供互補的信息
- **數據豐富性**：圖文對、音視頻等多模態數據大量存在
- **下游應用**：自動駕駛、機器人、醫療影像、視頻理解

---

## 2. 視覺-語言預訓練 (VLP)

### 2.1 CLIP (Contrastive Language-Image Pre-training)

**核心思想**：在共享嵌入空間中對齊圖像和文本。

```
                圖像編碼器
                (ViT/ResNet)
                    ↓
            ┌───────────────────┐
   🖼️  ──→  │  Image Embedding   │  ──→  圖像向量 (512d)
            └───────────────────┘
                    ↑
            對比學習
                    ↑
            ┌───────────────────┐
   📝  ──→  │  Text Embedding    │  ──→  文本向量 (512d)
            └───────────────────┘
                文本編碼器
                (Transformer)
```

**訓練目標**：對角線為正樣本，其他為負樣本的對比損失：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    """CLIP 的對比學習損失"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
    
    def forward(self, image_features, text_features):
        """
        image_features: (batch, dim)  圖像特徵
        text_features:  (batch, dim)  文本特徵
        """
        # 餘弦相似度
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        logits_per_image = self.logit_scale.exp() * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        # 標籤：對角線為正樣本
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2


# 使用 HuggingFace CLIP
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 零樣本圖像分類
from PIL import Image
image = Image.open("cat.jpg")
texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob.item():.4f}")
```

### 2.2 BLIP-2：引導式 Q-Former

BLIP-2 通過輕量級的 Q-Former 連接凍結的圖像編碼器和 LLM：

```
Image Encoder (frozen)  →  Q-Former  →  LLM (frozen)
   (EVA-CLIP)            (可訓練)     (LLaMA / OPT)
        ↓
   Image Features
   257 tokens
```

**優勢**：只訓練 Q-Former 參數（約 188M），節省大量資源。

### 2.3 LLaVA：視覺指令微調

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 準備輸入
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg"
image = Image.open(requests.get(url, stream=True).raw)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "這張圖片裡有什麼？請用繁體中文回答。"},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(image, prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

---

## 3. 圖像生成 | Image Generation

### 3.1 Diffusion Models 原理

**前向過程 (Forward)**：逐步加噪聲

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

**反向過程 (Reverse)**：學習去噪聲

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))$$

**訓練目標**：預測噪聲

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 \right]$$

### 3.2 Stable Diffusion

```python
from diffusers import StableDiffusionPipeline
import torch

# 載入模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    device_map="auto",
)

# 文本生成圖像
prompt = "A serene mountain landscape at sunset, with a lake reflecting the orange sky, digital art, high quality"
image = pipe(
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
).images[0]

image.save("generated.png")
```

### 3.3 ControlNet：可控生成

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np

# 載入 ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny"
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)

# 從邊緣圖控制生成
canny_image = cv2.Canny(cv2.imread("input.jpg"), 100, 200)
canny_image = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)

image = pipe(
    "a beautiful house, sunny day, high quality",
    image=Image.fromarray(canny_image),
).images[0]
```

### 3.4 SDXL 與最新進展

| 模型 | 發布 | 創新 |
|------|------|------|
| **DALL-E 2** | 2022 | CLIP + Diffusion |
| **Stable Diffusion 1.5** | 2022 | Latent Diffusion，開源 |
| **SDXL** | 2023 | 1024×1024，雙 UNet |
| **SD 3** | 2024 | DiT (Diffusion Transformer) |
| **FLUX** | 2024 | 開源 SOTA |
| **DALL-E 3** | 2023 | 與 ChatGPT 整合 |

---

## 4. 視頻生成 | Video Generation

### 4.1 Sora (OpenAI, 2024)

**核心創新**：
- **Spacetime Patches**：將視頻視為時空 patch 序列
- **Diffusion Transformer (DiT)**：用 Transformer 替代 UNet
- **原生生成**：可生成 60 秒 1080p 視頻
- **物理一致性**：理解重力、流體等物理規律

### 4.2 開源替代

```python
# 使用 CogVideoX
from diffusers import CogVideoXPipeline
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.float16)
video = pipe(prompt="A cat walking in the garden").frames[0]
```

---

## 5. 音頻與語音 | Audio & Speech

### 5.1 任務分類

| 任務 | 輸入 | 輸出 | 代表模型 |
|------|------|------|----------|
| **ASR (語音識別)** | 音頻 | 文本 | Whisper, Paraformer |
| **TTS (語音合成)** | 文本 | 音頻 | Tacotron, VALL-E, CosyVoice |
| **音樂生成** | 文本/音頻 | 音樂 | MusicGen, Suno |
| **聲紋識別** | 音頻 | 說話人 ID | SpeakerNet |
| **語音翻譯** | 音頻 | 另一語言 | SeamlessM4T |

### 5.2 Whisper (OpenAI)

```python
import whisper

model = whisper.load_model("large-v3")
result = model.transcribe("audio.mp3", language="zh")
print(result["text"])
```

### 5.3 音樂生成

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile

processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")

inputs = processor(
    text=["A cheerful jazz piano piece with light drums"],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

# 保存音頻
sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("musicgen_output.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
```

---

## 6. 多模態大語言模型 (MLLM)

### 6.1 架構

```
┌──────────────────────────────────────────────────────────────┐
│            Multimodal LLM 架構                                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Image ──→ Vision Encoder ──→ Projector ──→ LLM ──→ Text  │
│                  │                    │        │              │
│              (CLIP/ViT)            (MLP/     (LLaMA/       │
│                                    Q-Former)  Qwen)          │
│                                                              │
│   Audio ──→ Audio Encoder ──→ Projector ──→ LLM ──→ Text   │
│   Video ──→ Video Encoder ──→ Projector ──→ LLM ──→ Text   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 主流開源 MLLM

| 模型 | 視覺編碼器 | LLM | 參數 | 特點 |
|------|------------|-----|------|------|
| **LLaVA-1.5** | CLIP ViT-L | Vicuna | 13B | 經典開源 |
| **Qwen-VL** | ViT | Qwen | 9.6B | 中文友好 |
| **InternVL** | InternViT | InternLM | 6B-26B | 高分辨率 |
| **MiniCPM-V** | SigLIP | MiniCPM | 8B | 高效 |
| **LLaVA-NeXT** | ViT | Mistral/LLaMA | 7B-34B | 高分辨率增強 |
| **CogVLM** | EVA-CLIP | CogLLM | 17B | 視覺推理強 |

### 6.3 構建自定義 MLLM

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor

class MultimodalLLM(nn.Module):
    """簡化的多模態 LLM"""
    
    def __init__(self, llm_name="meta-llama/Llama-2-7b-chat-hf", 
                 vision_name="openai/clip-vit-large-patch14"):
        super().__init__()
        
        # 視覺編碼器
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_name)
        
        # LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        
        # 視覺到語言的投影層
        vision_dim = self.vision_encoder.config.hidden_size  # 1024
        llm_dim = self.llm.config.hidden_size                # 4096
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        
        # 凍結視覺編碼器和 LLM
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad = False
    
    def forward(self, images, input_ids, attention_mask=None, labels=None):
        # 1. 編碼圖像
        with torch.no_grad():
            vision_outputs = self.vision_encoder(images)
            image_features = vision_outputs.last_hidden_state  # (B, 257, 1024)
        
        # 2. 投影到 LLM 維度
        image_embeds = self.projector(image_features.to(self.llm.dtype))  # (B, 257, 4096)
        
        # 3. 編碼文本
        text_embeds = self.llm.model.embed_tokens(input_ids)
        
        # 4. 拼接
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # 5. LLM 前向
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs
```

---

## 7. 多模態評估基準 | Benchmarks

| 基準 | 內容 | 評估能力 |
|------|------|----------|
| **MMBench** | 多選題 | 綜合理解 |
| **MMMU** | 大學級多模態問題 | 學科推理 |
| **MMStar** | 高質量視覺評估 | 視覺感知 |
| **VQA v2** | 視覺問答 | 視覺理解 |
| **COCO Caption** | 圖像描述 | 描述生成 |
| **ChartQA** | 圖表問答 | 結構化視覺 |
| **MathVista** | 視覺數學推理 | 數學能力 |
| **DocVQA** | 文檔問答 | 文字識別 |

---

## 8. 應用場景 | Applications

| 領域 | 應用 |
|------|------|
| **醫療** | CT/MRI 分析、輔助診斷 |
| **自動駕駛** | 場景理解、行人檢測 |
| **教育** | 解題、概念可視化 |
| **電商** | 視覺搜尋、商品生成 |
| **娛樂** | 遊戲資產、影視特效 |
| **機器人** | 操作視覺反饋 |
| **無障礙** | 視障輔助、實時描述 |
| **內容創作** | 設計、行銷、寫作 |

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│  多模態學習發展路線                                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  單模態預訓練 (2017-)                                         │
│  ├── BERT, GPT — 文本                                         │
│  ├── ResNet, ViT — 圖像                                       │
│  └── WaveNet — 音頻                                           │
│                                                              │
│  雙模態對齊 (2021-)                                            │
│  ├── CLIP — 圖文對齊                                          │
│  ├── DALL-E — 文本→圖像                                       │
│  └── Whisper — 音頻→文本                                      │
│                                                              │
│  多模態 LLM (2023-)                                            │
│  ├── LLaVA, MiniGPT-4 — 開源首波                              │
│  ├── GPT-4V, Gemini — 商業旗艦                                │
│  └── Qwen-VL, InternVL — 開源 SOTA                            │
│                                                              │
│  統一多模態 (2024+)                                            │
│  ├── GPT-4o — 原生多模態                                      │
│  ├── Sora — 視頻生成                                          │
│  └── Gemini 1.5 — 長上下文多模態                              │
│                                                              │
│  未來趨勢：                                                   │
│  1. 統一架構（One Model For All）                              │
│  2. 任意模態之間轉換                                          │
│  3. 具身智能 (Embodied AI)                                     │
│  4. 物理世界模型 (World Models)                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 參考文獻 | References

1. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML* (CLIP).
2. Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR* (Stable Diffusion).
3. Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. *ICML*.
4. Liu, H., et al. (2023). Visual Instruction Tuning (LLaVA). *NeurIPS*.
5. OpenAI. (2024). Sora: Creating video from text.
6. Brown, T., et al. (2020). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision.

---

*© 2026 chanxaviersy*
