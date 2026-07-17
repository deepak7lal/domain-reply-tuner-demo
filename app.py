"""
Domain Reply Tuner — Streamlit Demo
=====================================
Loads Mistral-7B-Instruct in 4-bit + your trained LoRA adapter, and gives
a chat-style interface to test it on any customer-support question.

Deploy this on Hugging Face Spaces (Streamlit SDK) for a public, shareable
demo -- Spaces gives enough RAM/disk for a 7B model, unlike most free
Streamlit-only hosts.
"""

import os
import streamlit as st
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------------------------
# Configuration -- change ADAPTER_REPO to your Hugging Face Hub repo once
# you've pushed the adapter (see push_to_hub.py / the README for steps).
# ---------------------------------------------------------------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_REPO = "kali-89/mistral-support-tuner"

# Log in to Hugging Face Hub using a token stored as a Space secret.
# Required because Mistral-7B-Instruct is a gated model -- without this,
# from_pretrained() will fail with a 401/403 even if quantization works.
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

st.set_page_config(page_title="Domain Reply Tuner", page_icon="💬")
st.title("💬 Domain Reply Tuner")
st.caption(
    "Mistral-7B-Instruct fine-tuned with QLoRA on customer-support replies. "
    "Type a question below to see the fine-tuned model respond."
)

HAS_GPU = torch.cuda.is_available()

if not HAS_GPU:
    st.error(
        "No GPU detected on this Space. 4-bit quantized inference for a "
        "7B model requires a CUDA GPU -- it will not run on CPU basic "
        "hardware. Go to Space Settings -> Hardware and switch to a "
        "T4-small (or better) GPU tier, then restart the Space."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading model (first run only, ~1-2 min)...")
def load_model():
    """
    Cached so the model only loads once per app session, not on every
    interaction. Streamlit reruns the whole script on each user action,
    so without this decorator we'd reload a 7B model every time someone
    typed a message.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
    model.eval()

    return model, tokenizer


def generate_reply(model, tokenizer, prompt, max_new_tokens=200):
    text = f"<s>[INST] {prompt} [/INST]"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text.split("[/INST]")[-1].strip()


# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.spinner("Loading model..."):
    model, tokenizer = load_model()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask a customer-support style question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Generating reply..."):
            reply = generate_reply(model, tokenizer, user_input)
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

st.divider()
st.caption(
    "Built with QLoRA (4-bit quantization + LoRA adapters) on the "
    "Bitext customer-support dataset. "
    "[View training details on GitHub](https://github.com/deepak7lal7)"
)
