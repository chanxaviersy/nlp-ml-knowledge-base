# 🎤 語音識別與語音處理 | Speech Recognition & Processing

> 摘要：語音處理將聲學信號轉換為文字或意義。本章介紹語音基礎、特徵提取 (MFCC)、語音識別 (ASR)、語音合成 (TTS)、語音翻譯以及 Whisper 等現代模型。  
> Abstract: Speech processing converts acoustic signals to text or meaning. This chapter covers speech fundamentals, feature extraction (MFCC), ASR, TTS, speech translation, and modern models like Whisper.

---

## 1. 語音信號基礎 | Speech Signal Fundamentals

### 1.1 物理基礎

- **採樣率 (Sample Rate)**：每秒採樣次數，常用 16 kHz、44.1 kHz、48 kHz
- **奈奎斯特定理**：採樣率必須 ≥ 2× 信號最高頻率
- **量化位元 (Bit Depth)**：每個樣本的精度，常用 16-bit

### 1.2 語音學特徵

| 屬性 | 描述 | 範圍 |
|------|------|------|
| **音高 (Pitch)** | 基頻 F0 | 80-400 Hz |
| **音色 (Timbre)** | 頻譜包絡 | 由聲道形狀決定 |
| **音量 (Loudness)** | 能量 | dB SPL |
| **持續時間 (Duration)** | 語音時長 | ms |

### 1.3 語音學單元

- **音素 (Phoneme)**：最小的語音單位（語言學）
- **音子 (Phone)**：語音的具體實現
- **字位 (Grapheme)**：書寫字符
- **詞 (Word)**：語義單位
- **音節 (Syllable)**：發音單位

---

## 2. 特徵提取 | Feature Extraction

### 2.1 預處理

```python
import numpy as np
import scipy.signal as signal

def preemphasis(signal, coefficient=0.97):
    """預加重：高頻增強"""
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])

def frame_signal(signal, frame_size=25, frame_shift=10, sample_rate=16000):
    """分幀"""
    frame_len = int(frame_size * sample_rate / 1000)  # 樣本數
    frame_step = int(frame_shift * sample_rate / 1000)
    
    num_frames = (len(signal) - frame_len) // frame_step + 1
    frames = np.zeros((num_frames, frame_len))
    for i in range(num_frames):
        frames[i] = signal[i * frame_step: i * frame_step + frame_len]
    
    return frames

def apply_hamming_window(frames):
    """應用漢明窗"""
    window = np.hamming(frames.shape[1])
    return frames * window
```

### 2.2 MFCC (Mel-Frequency Cepstral Coefficients)

MFCC 是語音識別中最經典的特徵，模擬人耳的聽覺感知：

```
語音信號 → 預加重 → 分幀 → 加窗 → FFT → Mel 濾波器組 → 對數 → DCT → MFCC
```

```python
import scipy.fftpack

def mel_filterbank(num_filters=26, fft_size=512, sample_rate=16000, 
                   low_freq=0, high_freq=None):
    """構建 Mel 濾波器組"""
    if high_freq is None:
        high_freq = sample_rate / 2
    
    # Hz 轉 Mel
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)
    
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    
    # FFT 頻率 bin
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    
    # 構建濾波器
    filters = np.zeros((num_filters, fft_size // 2 + 1))
    for i in range(num_filters):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        
        for j in range(left, center):
            filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            filters[i, j] = (right - j) / (right - center)
    
    return filters


def extract_mfcc(audio, sample_rate=16000, num_ceps=13, num_filters=26,
                 frame_size=25, frame_shift=10, fft_size=512):
    """提取 MFCC 特徵"""
    # 1. 預加重
    audio = preemphasis(audio)
    
    # 2. 分幀加窗
    frames = frame_signal(audio, frame_size, frame_shift, sample_rate)
    frames = apply_hamming_window(frames)
    
    # 3. FFT
    mag_frames = np.abs(np.fft.rfft(frames, n=fft_size)) ** 2
    
    # 4. Mel 濾波器組
    mel_filters = mel_filterbank(num_filters, fft_size, sample_rate)
    mel_energy = np.dot(mag_frames, mel_filters.T)
    mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
    
    # 5. 對數
    log_mel = np.log(mel_energy)
    
    # 6. DCT
    mfcc = scipy.fftpack.dct(log_mel, type=2, norm='ortho')[:, :num_ceps]
    
    # 7. 動態特徵（可選）
    delta = np.diff(mfcc, axis=0)
    delta_delta = np.diff(delta, axis=0)
    
    # 拼接
    features = np.concatenate([mfcc, delta, delta_delta], axis=1)
    
    return features
```

### 2.3 Filterbank (Fbank)

與 MFCC 類似但跳過 DCT 步驟，保留更多語音信息：

```python
def extract_fbank(audio, sample_rate=16000, num_filters=80, ...):
    """提取 Fbank 特徵"""
    # 同 MFCC 步驟 1-5
    return log_mel  # 不做 DCT
```

### 2.4 現代特徵

| 特徵 | 描述 | 用途 |
|------|------|------|
| **MFCC** | 13 維倒譜係數 | 傳統 ASR (GMM-HMM) |
| **Fbank** | 對數 Mel 濾波器能量 | 深度學習 ASR |
| **Whisper Features** | log-Mel spectrogram | Whisper |
| **Wav2Vec 2.0** | 自監督學習特徵 | 多語言 ASR |
| **HuBERT** | 離散單元 | 多用途 |

---

## 3. 語音識別 ASR | Automatic Speech Recognition

### 3.1 任務定義

將語音信號序列 $(x_1, x_2, ..., x_T)$ 映射到文字序列 $(w_1, w_2, ..., w_N)$：

$$\hat{W} = \arg\max_W P(W | X) = \arg\max_W P(X | W) P(W)$$

### 3.2 傳統 ASR：GMM-HMM

```
GMM (聲學模型) + HMM (時序模型) + N-gram LM
```

```python
# GMM-HMM 的簡化架構示意
class GMMHMM_ASR:
    def __init__(self):
        self.gmm = GaussianMixture(n_components=100)
        self.hmm = HMM(n_states=3)  # 每個音素 3 個狀態
        self.lm = NGramLanguageModel(n=3)
    
    def recognize(self, audio_features):
        # 1. GMM 計算每幀對每個狀態的概率
        # 2. HMM Viterbi 解碼得到狀態序列
        # 3. 狀態映射到音素
        # 4. 語言模型 + 解碼得到詞序列
        return text
```

### 3.3 深度學習 ASR

**CTC (Connectionist Temporal Classification)**：

解決輸入長度 $T$ 與輸出長度 $N$ 不一致的問題。

```python
import torch
import torch.nn as nn

class CTC_ASR(nn.Module):
    """基於 CTC 的語音識別模型"""
    
    def __init__(self, input_dim, vocab_size, hidden_dim=512, n_layers=3):
        super().__init__()
        # 卷積特徵提取
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        # BiLSTM 編碼器
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, n_layers, 
                               bidirectional=True, batch_first=True,
                               dropout=0.3)
        # 線性投影到詞彙表
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        # CTC 損失
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    
    def forward(self, audio, audio_lengths, text=None, text_lengths=None):
        """
        audio: (batch, time, feat_dim)
        """
        # 特徵提取
        x = self.conv(audio.transpose(1, 2)).transpose(1, 2)
        
        # BiLSTM 編碼
        x, _ = self.encoder(x)
        
        # 投影
        logits = self.fc(x)  # (batch, time, vocab_size)
        
        if text is not None:
            # 訓練模式：計算 CTC 損失
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, C)
            loss = self.ctc_loss(log_probs, text, audio_lengths, text_lengths)
            return loss
        
        return logits
```

### 3.4 端到端 Transformer ASR

```python
class TransformerASR(nn.Module):
    """Transformer 端到端 ASR"""
    
    def __init__(self, input_dim, vocab_size, d_model=512, nhead=8, n_layers=12):
        super().__init__()
        # CNN 降採樣
        self.conv = nn.Conv1d(input_dim, d_model, kernel_size=3, stride=2, padding=1)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # 解碼器
        self.decoder = nn.Linear(d_model, vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=0)
    
    def forward(self, audio, text=None):
        # CNN 降採樣
        x = self.conv(audio.transpose(1, 2)).transpose(1, 2)  # (B, T/2, D)
        
        # Transformer 編碼
        x = self.transformer(x)
        
        # 輸出
        logits = self.decoder(x)
        
        if text is not None:
            return self.ctc_loss(logits.log_softmax(-1).transpose(0, 1), 
                                 text, ...)
        return logits
```

### 3.5 Whisper (OpenAI)

**多語言、多任務**語音處理模型：

```python
import whisper

# 載入模型（tiny, base, small, medium, large）
model = whisper.load_model("base")

# 1. 語音識別
result = model.transcribe("audio.mp3", language="zh")
print(result["text"])
print(result["language"])  # 自動檢測的語言

# 2. 翻譯為英文
result = model.transcribe("audio_chinese.mp3", task="translate")
print(result["text"])  # 英文翻譯

# 3. 長音頻處理（自動切分）
result = model.transcribe("long_audio.mp3", 
                           language="en",
                           initial_prompt="This is a technical presentation about AI.")
```

### 3.6 主流 ASR 系統

| 系統 | 發布 | 特點 |
|------|------|------|
| **Whisper** | OpenAI 2022 | 多語言、零樣本強 |
| **Wav2Vec 2.0** | Meta 2020 | 自監督預訓練 |
| **HuBERT** | Meta 2021 | 離散單元 |
| **Conformer** | Google 2020 | CNN + Transformer |
| **Paraformer** | Alibaba 2022 | 非自回歸、並行 |
| **USM** | Google 2023 | 統一語音模型 |
| **SeamlessM4T** | Meta 2023 | 多語言翻譯 |
| **Canary** | NVIDIA 2024 | 高精度多語言 |

---

## 4. 語音合成 TTS | Text-to-Speech

### 4.1 任務定義

從文字生成自然語音：

$$Y = f(W, \text{speaker}, \text{style})$$

### 4.2 傳統 TTS 流水線

```
文本 → 文本分析 → 韻律預測 → 聲學特徵 → 聲碼器 → 音頻
         ↓          ↓           ↓          ↓
       分詞/韻律  音高/時長    Mel 譜    Waveform
```

### 4.3 現代 TTS 系統

| 系統 | 特點 |
|------|------|
| **Tacotron 2** | 端到端 Mel 譜預測 |
| **WaveNet** | 自回歸波形生成 |
| **HiFi-GAN** | 高效並行聲碼器 |
| **VITS** | 端到端 VAE |
| **VALL-E** | 3 秒樣本克隆聲音 |
| **Bark** | 多語言、情緒、音效 |
| **CosyVoice** | 中文友好、零樣本克隆 |
| **ChatTTS** | 對話場景 |
| **Tortoise TTS** | 高質量 |

```python
# 使用 HuggingFace 套件
from TTS.api import TTS

# 載入模型
tts = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST").to("cuda")

# 生成語音
tts.tts_to_file(
    text="你好，這是中文語音合成測試。",
    file_path="output.wav",
    speaker_wav="reference.wav",  # 參考音色（聲音克隆）
    language="zh"
)
```

### 4.4 VALL-E：神經編解碼器語音克隆

```python
# 概念實現
class VALLE(nn.Module):
    """VALL-E 風格的語音克隆模型"""
    
    def __init__(self, codec_model, lm_model):
        super().__init__()
        # 神經音頻編解碼器（EnCodec）
        self.codec = codec_model
        # 大語言模型作為骨幹
        self.lm = lm_model
        # 音頻 token 嵌入
        self.audio_embed = nn.Embedding(codec_vocab_size, hidden_size)
    
    def clone_voice(self, text, reference_audio, target_text):
        """克隆聲音"""
        # 1. 編碼參考音頻為離散 token
        ref_tokens = self.codec.encode(reference_audio)
        
        # 2. LLM 生成目標音頻 token（基於文本 + 參考）
        target_tokens = self.lm.generate(
            prompt=ref_tokens,
            condition=target_text,
        )
        
        # 3. 解碼 token 為音頻
        audio = self.codec.decode(target_tokens)
        return audio
```

---

## 5. 語音翻譯 | Speech Translation

直接從語音翻譯到目標語言文字：

```python
# SeamlessM4T 多語言翻譯
from transformers import AutoProcessor, SeamlessM4TModel

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-large")
model = SeamlessM4TModel.from_pretrained("facebook/seamless-m4t-large")

# 從音頻翻譯為文字
audio_inputs = processor(audios=audio_array, return_tensors="pt")
output_tokens = model.generate(**audio_inputs, tgt_lang="cmn_Hant")  # 繁體中文
text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
print(text)
```

**ST 三種範式**：

| 範式 | 描述 | 優缺點 |
|------|------|--------|
| **級聯 (Cascade)** | ASR → MT | 模塊化、靈活 |
| **端到端 (E2E)** | Speech → Text | 低延遲、簡潔 |
| **統一模型** | 多任務聯合 | 通用但複雜 |

---

## 6. 語音評估 | Evaluation

### 6.1 ASR 評估

**WER (Word Error Rate)**：

$$\text{WER} = \frac{S + D + I}{N}$$

其中 $S$ 替換、$D$ 刪除、$I$ 插入，$N$ 標準答案詞數。

```python
def calculate_wer(reference, hypothesis):
    """計算 WER"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # 編輯距離
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[n][m] / n

# 中文 WER（按字符計算）
def calculate_cer(reference, hypothesis):
    """字符錯誤率 CER"""
    return calculate_wer(list(reference), list(hypothesis))
```

**中文場景**：
- **WER**（詞級別）：需要先分詞
- **CER**（字符級別）：直接統計字符錯誤

### 6.2 TTS 評估

| 指標 | 衡量 |
|------|------|
| **MOS (Mean Opinion Score)** | 人類主觀評分（1-5） |
| **CMOS** | 比較兩個系統 |
| **WER of ASR** | TTS→ASR 一致性 |
| **Speaker Similarity** | 音色相似度 |
| **UTMOS** | 自動 MOS 預測 |

### 6.3 基準

| 基準 | 內容 |
|------|------|
| **LibriSpeech** | 1000 小時英文有聲書 |
| **Common Voice** | Mozilla 多語言眾包 |
| **AISHELL** | 中文 178 小時 |
| **WenetSpeech** | 中文 10000 小時 |
| **GigaSpeech** | 大規模多領域 |
| **TED-LIUM** | TED 演講 |

---

## 7. 完整 ASR Pipeline

```python
class ASRPipeline:
    """完整 ASR Pipeline"""
    
    def __init__(self, model_name="openai/whisper-base"):
        from transformers import pipeline
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=model_name,
        )
    
    def transcribe(self, audio_path, language=None):
        result = self.asr(
            audio_path,
            return_timestamps=True,
            generate_kwargs={"language": language} if language else {},
        )
        return result
    
    def transcribe_long(self, audio_path, chunk_length_s=30):
        """長音頻處理"""
        result = self.asr(
            audio_path,
            chunk_length_s=chunk_length_s,
            return_timestamps=True,
        )
        return result


# 使用
asr = ASRPipeline("openai/whisper-base")
result = asr.transcribe("chinese_audio.mp3", language="zh")
for chunk in result["chunks"]:
    print(f"[{chunk['timestamp'][0]:.1f}s-{chunk['timestamp'][1]:.1f}s] {chunk['text']}")
```

---

## 8. 應用場景 | Applications

| 領域 | 應用 |
|------|------|
| **語音助手** | Siri, Alexa, Google Assistant |
| **客服** | 自動應答、呼叫中心 |
| **會議** | 實時轉錄、會議紀要 |
| **教育** | 語言學習、發音評估 |
| **無障礙** | 聽障輔助、實時字幕 |
| **醫療** | 病歷語音輸入、診斷記錄 |
| **娛樂** | 視頻字幕、有聲書 |
| **汽車** | 車載語音控制 |

---

## 總結 | Summary

```
┌──────────────────────────────────────────────────────────────┐
│  語音處理發展路線                                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  傳統方法 (1990s-2010s)                                       │
│  ├── GMM-HMM                                                │
│  ├── Kaldi 工具包                                            │
│  └── 規則 + 統計混合                                         │
│                                                              │
│  深度學習 (2015-2020)                                          │
│  ├── DeepSpeech (CNN+RNN+CTC)                               │
│  ├── Listen Attend Spell (LAS)                               │
│  ├── Jasper (CNN)                                           │
│  └── Tacotron 2 (TTS)                                       │
│                                                              │
│  自監督預訓練 (2020+)                                          │
│  ├── Wav2Vec 2.0                                            │
│  ├── HuBERT                                                 │
│  ├── WavLM                                                  │
│  └── BEST-RQ                                               │
│                                                              │
│  統一多任務模型 (2022+)                                        │
│  ├── Whisper (OpenAI)                                       │
│  ├── SeamlessM4T (Meta)                                     │
│  ├── USM (Google)                                           │
│  └── VALL-E / Bark (TTS)                                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 參考文獻 | References

1. Yu, D., & Deng, L. (2016). *Automatic Speech Recognition: A Deep Learning Approach*. Springer.
2. Hinton, G., et al. (2012). Deep Neural Networks for Acoustic Modeling in Speech Recognition. *IEEE Signal Processing Magazine*.
3. Baevski, A., et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *NeurIPS*.
4. Radford, A., et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision (Whisper). *ICML*.
5. Wang, C., et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E). *arXiv*.

---

## ⚡ 5 分鐘快速實戰 | Quick Practice

### 1. Whisper 語音識別 (1 行)

```python
import whisper
model = whisper.load_model("base")
result = model.transcribe("audio.mp3", language="zh")
print(result["text"])
```

### 2. HuggingFace 語音翻譯

```python
from transformers import pipeline

# 多語言 ASR
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
text = asr("audio.mp3", return_timestamps=True)["text"]

# 翻譯為英文
translator = pipeline("translation", model="facebook/m2m100_418M")
english = translator(text, src_lang="zh", tgt_lang="en")[0]["translation_text"]
```

### 3. 文本轉語音 (TTS)

```python
# 方法 1：HuggingFace
from transformers import pipeline
tts = pipeline("text-to-speech", model="microsoft/speecht5_tts")
speech = tts("Hello, this is a test.")
# 播放：speech["audio"]

# 方法 2：Coqui TTS
# from TTS.api import TTS
# tts = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST")
# tts.tts_to_file(text="你好", file_path="out.wav")
```

### 4. 音頻特徵提取

```python
import librosa
import numpy as np

# 載入音頻
y, sr = librosa.load("audio.wav", sr=16000)

# MFCC 特徵
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Mel 頻譜
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
log_mel = librosa.power_to_db(mel)

# 零交叉率（語音活動檢測）
zcr = librosa.feature.zero_crossing_rate(y)
```

---

## 🧪 練習題 | Exercises

| # | 難度 | 題目 |
|---|------|------|
| 1 | ⭐ | 用 Whisper 識別 5 段不同語言的音頻 |
| 2 | ⭐⭐ | 比較 Whisper base/small/medium 在 10 段中文音頻的 WER |
| 3 | ⭐⭐ | 用 TTS 合成 5 段不同情緒的語音 |
| 4 | ⭐⭐⭐ | 微調 Whisper 在自定義語音數據（如專業術語）|
| 5 | ⭐⭐⭐ | 構建語音對話系統：ASR → LLM → TTS |
| 6 | ⭐⭐⭐⭐ | 聲音克隆：3 秒樣本克隆任意聲音 |

---

## 🎧 開源工具

| 工具 | 用途 |
|------|------|
| **Whisper** | OpenAI 多語言 ASR |
| **ESPnet** | 端到端語音處理 |
| **Kaldi** | 經典 ASR 工具 |
| **Coqui TTS** | 高質量 TTS |
| **librosa** | 音頻分析 |
| **pyannote.audio** | 說話人分離 |

---

*© 2026 chanxaviersy*
