"""
Domain Reply Tuner — Gradio Demo (ZeroGPU)
=============================================
Loads Mistral-7B-Instruct in 4-bit + your trained LoRA adapter, and gives
a chat-style interface to test it on any customer-support question.

Runs on Hugging Face Spaces using the Gradio SDK + ZeroGPU hardware --
this is currently the only free way to get GPU access on Spaces, since
Docker-based SDKs (which Streamlit now requires) no longer have a free tier.

IMPORTANT: on ZeroGPU, a physical GPU is only attached while a function
decorated with @spaces.GPU is running. Any code that touches CUDA at
import time (e.g. loading a 4-bit quantized model or an adapter onto
"cuda") will fail with "No CUDA GPUs are available". To avoid this, all
model loading happens lazily, the first time generate_reply() is called
inside the @spaces.GPU-decorated function -- not at module import time.
"""

import os
import spaces
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_REPO = "kali-89/mistral-support-tuner"

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# Prevents Gradio's internal self-check (verifying the server can reach
# itself) from being routed through a proxy, which is the actual cause of
# "When localhost is not accessible" on Gradio 4.x -- not SSR (that's a
# 5.x-only feature and doesn't apply to the version pinned here).
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

# Loaded lazily on first request, inside the GPU-decorated function.
_model = None
_tokenizer = None


def _load_model():
    """Loads base model + adapter. Only ever called from inside a
    @spaces.GPU-decorated function, so a physical GPU is guaranteed
    to be attached by the time this runs."""
    global _model, _tokenizer

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

    _model = model
    _tokenizer = tokenizer
    return _model, _tokenizer


@spaces.GPU(duration=120)
def generate_reply(message, history):
    """
    duration=120 gives extra headroom for the FIRST call, since it also
    has to load + quantize the 7B model (a minute or more) on top of
    generating a reply. Subsequent calls reuse the cached _model/_tokenizer
    and are much faster.
    """
    global _model, _tokenizer
    if _model is None:
        _load_model()

    prompt = f"<s>[INST] {message} [/INST]"
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=_tokenizer.eos_token_id,
        )

    full_text = _tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text.split("[/INST]")[-1].strip()


demo = gr.ChatInterface(
    fn=generate_reply,
    type="messages",
    title="💬 Domain Reply Tuner",
    description=(
        "Mistral-7B-Instruct fine-tuned with QLoRA on customer-support "
        "replies. Type a question below to see the fine-tuned model respond. "
        "The first reply may take 1-2 minutes while the model loads -- "
        "subsequent replies will be much faster. "
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
