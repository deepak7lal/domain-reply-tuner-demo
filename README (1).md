---
title: Domain Reply Tuner
emoji: 💬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# Domain Reply Tuner — Chat Demo

A Gradio chat interface for a Mistral-7B model fine-tuned with QLoRA on
customer-support replies. Code lives on GitHub; the live demo runs on
Hugging Face Spaces using **ZeroGPU** — currently the only free way to get
GPU access on Spaces (Docker-based SDKs, which Streamlit now requires,
no longer have a free tier).

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

In your Kaggle notebook, after training completes, run `push_to_hub.py`
(copy its contents into a new cell) — this uploads your small (~27MB)
adapter to a public HF model repo. Update `HF_USERNAME` and `REPO_NAME`
in that script first.

### 2. Update this app to point at your adapter

In `app.py`, change:

```python
ADAPTER_REPO = "your-username/mistral-support-tuner"
```

to match the repo you pushed to.

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
2. Name it, choose **Gradio** as the SDK
3. Space hardware: choose **ZeroGPU (Free)** — this gives on-demand GPU
   access at no cost, which the free CPU basic tier can't provide for a
   7B model
4. Once created, push your code to the Space's git remote, or use
   **Files → Add file → Upload files** to add `app.py`,
   `requirements.txt`, and `README.md` directly
5. In **Settings → Variables and secrets**, add a secret named `HF_TOKEN`
   with a Hugging Face access token (needed since Mistral-7B-Instruct is
   a gated model) — make sure the same HF account has accepted the
   model's license on its model page first
6. The Space will build and give you a public URL like:
   `https://huggingface.co/spaces/YOUR_USERNAME/domain-reply-tuner-demo`

That URL is what you link from your resume/portfolio/GitHub README —
anyone can open it and chat with your fine-tuned model directly.

## Notes

- Functions that need the GPU are wrapped with `@spaces.GPU`, which is
  required for ZeroGPU Spaces — GPU access is only granted inside
  decorated function calls, not to the app as a whole.
- First load takes a minute or two (downloading + quantizing the 7B
  model). ZeroGPU has a daily quota per user; each reply consumes a small
  slice of it.
- If you outgrow ZeroGPU's free quota, HF PRO ($9/month) meaningfully
  raises the daily allowance.
