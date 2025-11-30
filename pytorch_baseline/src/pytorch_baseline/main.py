import torch
import torch.nn.functional as F
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
                    {"type": "text", "text": "You are a helpful assistant."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Write code that uses PyTorch and Transformers to do LLM inference using Gemma 3 model, please!"},
                ],
            },
        ],
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

    # Manual autoregressive sampling loop that mimics `generate()`
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    static_model_kwargs = {
        k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")
    }
    eos_token_ids = model.generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = tokenizer.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = []
    elif isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    max_new_tokens = 128000

    generated_tokens = None
    cur_input_ids = input_ids
    cur_attention_mask = attention_mask
    past_key_values = None

    with torch.inference_mode():
        for step in range(max_new_tokens):
            model_kwargs = dict(static_model_kwargs)
            if cur_attention_mask is not None:
                model_kwargs["attention_mask"] = cur_attention_mask

            outputs = model(
                input_ids=cur_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                **model_kwargs,
            )
            past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :].float()
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)

            if generated_tokens is None:
                generated_tokens = next_token
            else:
                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            # Decode and show running generation (prompt + new tokens)
            current_output_ids = torch.cat([input_ids, generated_tokens], dim=-1)
            current_text = tokenizer.batch_decode(
                current_output_ids,
                skip_special_tokens=True,
            )[0]
            print(f"[step {step + 1}] {current_text}")

            if eos_token_ids and all(
                token.item() in eos_token_ids for token in next_token.view(-1)
            ):
                break

            cur_input_ids = next_token
            if cur_attention_mask is not None:
                cur_attention_mask = torch.cat(
                    [cur_attention_mask, torch.ones_like(next_token)],
                    dim=-1,
                )

    if generated_tokens is not None:
        output_ids = torch.cat([input_ids, generated_tokens], dim=-1)
    else:
        output_ids = input_ids

    output_text = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0]

    print("=== Model output ===")
    print(output_text)


if __name__ == "__main__":
    main()
