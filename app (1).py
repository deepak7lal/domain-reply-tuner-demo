"""
Domain Reply Tuner — Gradio Demo (ZeroGPU)
=============================================
Loads Mistral-7B-Instruct in 4-bit + your trained LoRA adapter, and gives
a chat-style interface to test it on any customer-support question.

Runs on Hugging Face Spaces using the Gradio SDK + ZeroGPU hardware --
this is currently the only free way to get GPU access on Spaces, since
Docker-based SDKs (which Streamlit now requires) no longer have a free tier.
"""

import os
import spaces
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr

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

# ---------------------------------------------------------------------------
# Load model at import time (standard ZeroGPU pattern). The physical GPU is
# only attached when a function decorated with @spaces.GPU actually runs,
# but ZeroGPU's driver lets you build the model with device_map="auto" /
# .to("cuda") here regardless.
# ---------------------------------------------------------------------------
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


@spaces.GPU(duration=60)
def generate_reply(message, history):
    """
    ZeroGPU only grants GPU access inside functions decorated with
    @spaces.GPU. `duration` is the max seconds this call is allowed to run
    on the GPU -- keep it tight so quota is used efficiently.
    """
    prompt = f"<s>[INST] {message} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text.split("[/INST]")[-1].strip()


demo = gr.ChatInterface(
    fn=generate_reply,
    type="messages",
    title="💬 Domain Reply Tuner",
    description=(
        "Mistral-7B-Instruct fine-tuned with QLoRA on customer-support "
        "replies. Type a question below to see the fine-tuned model respond. "
        "Built with 4-bit quantization + LoRA adapters on the Bitext "
        "customer-support dataset."
    ),
    examples=[
        "My order hasn't arrived yet, what should I do?",
        "How do I request a refund?",
        "I want to change my shipping address after placing an order.",
    ],
)

if __name__ == "__main__":
    demo.launch()
