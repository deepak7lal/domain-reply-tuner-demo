# Domain Reply Tuner — Chat Demo

A Streamlit chat interface for a Mistral-7B model fine-tuned with QLoRA on
customer-support replies. Code lives on GitHub; the live demo runs on
Hugging Face Spaces (needed for the RAM/GPU a 7B model requires — most
free Streamlit-only hosts can't run this).

## Architecture

```
GitHub repo (this code)  ──mirrors to──>  Hugging Face Space (runs it, public URL)
                                                    │
                                                    ▼
                                    loads: base Mistral-7B (from HF Hub)
                                         + your LoRA adapter (from HF Hub)
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
git commit -m "Domain reply tuner Streamlit demo"
git remote add origin https://github.com/YOUR_USERNAME/domain-reply-tuner-demo.git
git push -u origin main
```

### 4. Deploy to Hugging Face Spaces

1. Go to https://huggingface.co/new-space
2. Name it, choose **Streamlit** as the SDK
3. Choose hardware:
   - **CPU basic (free)**: works, but very slow (~1-2 min per reply) since
     a 7B model on CPU is heavy even quantized.
   - **T4 small (paid, ~$0.60/hr while running)**: fast, recommended if
     you want a snappy demo for interviews — you only pay while the Space
     is actively running.
4. Once created, go to the Space's **Settings → Repository** and link it
   to sync from your GitHub repo, OR just upload `app.py` and
   `requirements.txt` directly through the Space's web file editor.
5. The Space will build and give you a public URL like:
   `https://huggingface.co/spaces/YOUR_USERNAME/domain-reply-tuner-demo`

That URL is what you link from your resume/portfolio/GitHub README —
anyone can open it and chat with your fine-tuned model directly.

## Notes

- First load takes 1-2 minutes (downloading + quantizing the 7B model).
  Subsequent messages in the same session are fast since the model stays
  cached in memory (`@st.cache_resource`).
- If you used a CPU-tier Space, expect each reply to take noticeably
  longer than a GPU tier. Mention this expectation to anyone testing it,
  or upgrade to a GPU tier temporarily when demoing live.
