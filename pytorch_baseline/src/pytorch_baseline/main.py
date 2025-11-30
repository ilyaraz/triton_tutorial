import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_device():
    """
    Prefer Apple GPU (mps) on Apple Silicon, then CUDA, otherwise CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    model_id = "google/gemma-3-1b-it"  # 1B instruct, text-only
    device = get_device()
    print(f"Using device: {device}")

    # Dtype: use float16 on GPU/MPS, float32 on CPU
    if device.type in ("mps", "cuda"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Attention implementation:
    #  - "sdpa" is best on CUDA
    #  - "eager" is usually safer on MPS / CPU
    attn_impl = "eager" if device.type == "mps" else "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation=attn_impl,
    )
    model.to(device)
    model.eval()

    # Chat-style messages for the instruct model
    messages = [
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Write code that uses PyTorch and Transformers to do LLM inference using Gemma 3 model, please!"}
                ],
            },
        ]
    ]

    # Use the tokenizer's chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move inputs to the device (keep integer dtypes as-is!)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128000,
            do_sample=False,
        )

    output_text = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0]

    print("=== Model output ===")
    print(output_text)


if __name__ == "__main__":
    main()
