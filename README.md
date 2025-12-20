# Multimodal Chart-To-Text Generator for Reports (ChartNarrator)

This repo is our end-to-end pipeline to **fine-tune InternVL2-1B on ChartQA** so the model gets better at reading charts and answering chart-based questions (and ultimately generating clean chart-to-text explanations you can reuse in reports).

It’s built to run on **Colab Pro (A100 GPU)**, LoRA + 4-bit quantization instead of full fine-tuning.

---

## What this project does (in plain words)

Charts are everywhere (dashboards, papers, reports), but extracting meaning from them is still annoying:
- you need OCR-ish understanding (axis labels, legends, numbers),
- plus reasoning (comparisons, trends, max/min, proportions),
- plus clean language generation.

So we take a strong multimodal base model (**InternVL2-1B**) and fine-tune it specifically on **ChartQA** to improve chart understanding and chart-to-text responses.

---

## Model we fine-tune: InternVL2-1B (quick but detailed)

**InternVL2-1B** is a small-but-strong multimodal model made of:

### 1) InternViT-300M (Vision Encoder) — **448×448 input**
- This is the vision backbone that converts a chart image into visual tokens.
- **448×448** is important because charts have small text + thin lines; higher resolution preserves details better than 224×224.
- The model can optionally split images into tiles (patch groups), but for Colab stability we keep:
  - `image_size = 448`
  - `max_num = 1` (single tile / single image chunk)

### 2) MLP Projector (Vision → Language bridge)
- The vision encoder outputs visual features.
- The MLP projector maps those features into the same embedding space that the language model can “understand”.
- This is the bridge that lets the LLM attend to image content.

### 3) Qwen2-0.5B-Instruct (Language Model)
- This is the text brain: it reads the prompt + image tokens and generates the answer.
- Since it’s instruction-tuned, it follows a chat-style format nicely (user/assistant messages).

**Why we chose this model**
- It’s **small enough** to fine-tune on a free T4 using PEFT.
- It’s already strong on chart/document understanding (solid ChartQA baseline).
- It supports a clean multimodal chat interface (`<image>` token in prompt).

---

## Dataset: ChartQA (HuggingFaceM4/ChartQA)

ChartQA is a chart question answering dataset with:
- chart images (bar, line, pie, etc.)
- questions about the chart
- answers (sometimes numeric, sometimes text)
- metadata like whether the question/answer is human-authored or machine-generated

We convert it into a chat-style JSONL format like:

```json
{
  "images": ["path/to/chart.png"],
  "messages": [
    {"role": "user", "content": "<image>\n<Question here>"},
    {"role": "assistant", "content": "<Answer here>"}
  ]
}
```

Repo Structure

```

Multimodal-Chart-To-Text/
├── ChartNarrator/
│   ├── data/                 # (optional) extra local data pointers / scripts
│   ├── outputs/              # training outputs, checkpoints, logs
│   ├── src/                  # any helper scripts (optional)
│   ├── README.md             # (this file if you move it inside)
│   └── requirements.txt
├── intern_chart_qa.ipynb     # main notebook (can also move inside ChartNarrator/)
└── .gitignore
```

Environment setup
Recommended (Colab / Linux)

Inside Colab:

```
pip -q install -U "transformers>=4.43" accelerate bitsandbytes peft datasets pandas tqdm matplotlib albumentations ms-swift sentencepiece

```

Pillow issue (common)

If you see conflicts like:

gradio requires pillow < 12.0 but you have pillow 12.0.0

Fix it by pinning Pillow:
```
pip -q install "pillow<12.0"

```

---

End-to-end pipeline (what the notebook does)
Step 1 — Download ChartQA and save locally

We pull ChartQA from HuggingFace and store:

- images in chartqa_local/images/...
- metadata in chartqa_local/metadata/train.csv, val.csv
- This makes the pipeline repeatable and avoids re-downloading every run.

---

Step 2 — Preprocessing: make JSONL in chat format

We convert CSV rows into messages:

User prompt: "<image>\n{query}"

Assistant: {label}

This is the key thing: we train the model exactly in the format we will use for inference.

---

Step 3 — Augmentation

Charts vary a lot in brightness, compression, quality, and color.
So we apply image-level augmentation using Albumentations to improve robustness.

Augmentations used (from our pipeline):

Resize with LongestMaxSize(max_size=448)

Strong-ish color/contrast variations (OneOf):

- RandomBrightnessContrast
- ColorJitter
- CLAHE
- RGBShift
- ToGray

Small blur sometimes: GaussianBlur(p=0.20)

Compression artifacts: ImageCompression(quality_range=(70, 100), p=0.30)

---

Step 4 — Baseline evaluation (before fine-tuning)

We evaluate the base InternVL2-1B on validation JSONL to get a baseline.

Metric style:

- Exact match accuracy after text normalization:

  - lowercase
  - strip special characters
  - collapse whitespace

This baseline is important so we can clearly say:

“Fine-tuning improved accuracy from X → Y”

---

### Fine-tuning strategy (PEFT + 4-bit) — explained clearly

We use LoRA (Low-Rank Adaptation) + 4-bit quantization to fit training on a T4.

### Why LoRA?

Instead of updating all weights (too heavy), LoRA:

- freezes the base model
- learns small trainable low-rank matrices
- gives most of the benefit with a tiny fraction of parameters

### Why 4-bit quantization?

We load most weights in 4-bit (bitsandbytes) to reduce VRAM usage.
This is basically the “QLoRA-style” approach:

- base weights in 4-bit
- LoRA weights trainable (small)
- compute in bf16 for stability

---

### Training command (SWIFT SFT)

We fine-tune with ms-swift using the swift sft command:

Key training settings we use:

- train_type: lora
- quant_method: bnb, quant_bits: 4
- torch_dtype: bfloat16
- batch size: 1 (per device), plus gradient accumulation
- gradient_accumulation_steps: 8
- learning_rate: 2e-4
- epochs: 9
- max_length: 8192
- model kwargs: {"max_num": 1, "image_size": 448}

Output goes into something like:
```
outputs/internvl2-1b-chartqa-qlora-aug-full-clean/
```

---

### After training: load the LoRA adapter

We do **adapter loading** instead of merging weights (keeps the base model untouched):

- Load base InternVL2-1B
- Attach the LoRA adapter from output dir
- Run inference normally

That’s why this approach is neat:

- base model stays the same
- adapter is portable (small)
- easy to share and re-use

---

### Evaluation (after fine-tuning)

We run the same evaluation loop again on the validation set:
- same normalization
- same exact match scoring
- log predictions for error analysis

Suggested things we look at in analysis:

- numeric mistakes (rounding, formatting)
- OCR-type failures (axis labels, legends)
- multi-step reasoning failures (compare two bars, trend over time)

---
### Results

<img width="1146" height="294" alt="image" src="https://github.com/user-attachments/assets/261c5196-497b-4be3-a1c7-f3bb7a7edd76" />

**LLM Training and Sample input**
1. <img width="1152" height="988" alt="image" src="https://github.com/user-attachments/assets/d53cc93a-8c4b-4535-9f70-0a80b02513fc" />

2. <img width="864" height="1078" alt="image" src="https://github.com/user-attachments/assets/ecb65c5b-19d6-4eed-89a0-c349e0b486f5" />

### Some predictions by Vision Language Model

1. <img width="1298" height="1052" alt="image" src="https://github.com/user-attachments/assets/d1aedbbc-43eb-4b07-8bdb-00588d1b95fc" />

### Validation Results

1. <img width="1956" height="454" alt="image" src="https://github.com/user-attachments/assets/fe98c34c-2e61-4c6e-8181-bf3fd3d2c2f8" />

### Few wrong prediction by model

1. <img width="1114" height="570" alt="image" src="https://github.com/user-attachments/assets/2abb11d3-0f5c-4784-9085-4ddcead2dc2a" />

---

How to run (recommended order)
Option A — Colab (simplest)

Open intern_chart_qa.ipynb

Run cells top-to-bottom:
- install
- download ChartQA
- preprocess + augment
- baseline eval
- fine-tune with swift sft
- load adapter + final eval

Option B — Local / GPU machine

Same notebook works, but make sure:
- CUDA is available
- ms-swift installed
- enough disk space for ChartQA images




---



