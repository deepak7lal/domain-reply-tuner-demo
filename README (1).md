---
title: Domain Reply Tuner
emoji: 💬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
---

# Domain Reply Tuner — Chat Demo

A Gradio chat interface for a Mistral-7B model fine-tuned with QLoRA on
customer-support replies. Code lives on GitHub; the live demo runs on
Hugging Face Spaces using **ZeroGPU** — a free, on-demand GPU tier for
Gradio Spaces (Docker-based SDKs, which Streamlit now requires, no longer
have a free tier).

**Live demo:** https://huggingface.co/spaces/kali-89/domain-reply-tuner-demo

## Architecture

```
GitHub repo (this code)  ──mirrors to──>  Hugging Face Space (runs it, public URL)
                                                    │
                                                    ▼
                                    loads: base Mistral-7B (from HF Hub)
                                         + your LoRA adapter (from HF Hub)
                                    inference runs on ZeroGPU (free, on-demand)
```

## Setup steps

### 1. Push your trained adapter to Hugging Face Hub

Upload the adapter files (`adapter_config.json`, `adapter_model.safetensors`,
tokenizer files) to a model repo, e.g. via `huggingface_hub`'s
`HfApi().upload_folder(...)` or the web uploader.

### 2. Update this app to point at your adapter

In `app.py`:

```python
ADAPTER_REPO = "your-username/your-adapter-repo"
```

### 3. Push this code to GitHub

```bash
git init
git add app.py requirements.txt README.md
git commit -m "Domain reply tuner Gradio demo"
git remote add origin https://github.com/YOUR_USERNAME/domain-reply-tuner-demo.git
git push -u origin main
```

### 4. Deploy to Hugging Face Spaces

1. Go to https://huggingface.co/new-space
2. Choose **Gradio** as the SDK
3. Space hardware: **ZeroGPU (Free)**
4. Push your code to the Space's git remote, or upload the files directly
5. In **Settings → Variables and secrets**, add a secret named `HF_TOKEN`
   with a Hugging Face access token (needed since the base model,
   Mistral-7B-Instruct, is gated) — make sure the same HF account has
   accepted the model's license first

## Key implementation notes

- **Lazy model loading**: on ZeroGPU, a physical GPU is only attached
  while a function decorated with `@spaces.GPU` is running. Loading the
  model at import time causes a `RuntimeError: No CUDA GPUs are available`.
  This app loads the base model + adapter lazily, on the first call to
  `generate_reply`, which is itself `@spaces.GPU`-decorated.
- **Pinned dependency versions**: `gradio==5.9.1` and `spaces>=0.30.3` are
  pinned explicitly in `requirements.txt`. Looser version ranges can
  resolve to mismatched `gradio`/`spaces` combinations that break Space
  startup with launch errors.
- First reply after a cold start takes 1-2 minutes (downloading +
  quantizing the 7B model). Subsequent replies are much faster since the
  model is cached in memory for the life of the Space's session.
- ZeroGPU has a daily usage quota per user; HF PRO raises the allowance
  if you outgrow the free tier.
