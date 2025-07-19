from transformers import pipeline
import torch
import re

# Use a small, fast model (e.g., Zephyr-7B or TinyLlama)
model_name = "HuggingFaceH4/zephyr-7b-beta"  # Free, Apache 2.0 license

# Load quantized model for low RAM (works on CPU/GPU)
llm = pipeline(
    "text-generation",
    model=model_name,
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16  # Saves memory
)

def parse_receipt_with_llm(text: str) -> dict:
    prompt = f"""
    Extract vendor, date, and amount as JSON from this receipt:
    {text}
    Return ONLY valid JSON like: {{"vendor": "Amazon", "date": "2024-05-01", "amount": 29.99}}
    """
    response = llm(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    # Extract JSON from the response (handles markdown-style wrapping)
    json_str = re.search(r"\{.*\}", response[0]["generated_text"]).group()
    return json.loads(json_str)

def categorize_vendor(vendor: str) -> str:
    prompt = f"""
    Classify this vendor into: Groceries, Utilities, Entertainment, Other.
    Vendor: {vendor}
    Respond ONLY with the category name.
    """
    response = llm(prompt, max_new_tokens=10)
    return response[0]["generated_text"].strip() 
client = llm