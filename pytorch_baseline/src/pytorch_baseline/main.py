import torch
import torch.nn.functional as F
from transformers import Gemma3TextConfig, Gemma3ForCausalLM, GemmaTokenizerFast
import sys


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    model_id = "google/gemma-3-1b-it"  # 1B instruct, text-only
    device = get_device()
    print(f"Using device: {device}")

    if device.type in ("mps", "cuda"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = GemmaTokenizerFast.from_pretrained(
        "google/gemma-3-1b-it",
        padding_side="left",
        trust_remote_code=False,
    )

    attn_impl = "eager" if device.type == "mps" else "sdpa"

    config = Gemma3TextConfig.from_pretrained("google/gemma-3-1b-it")
    model = Gemma3ForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        config=config,
        dtype=dtype,
        attn_implementation=attn_impl,
    )
    model.to(device)
    model.eval()

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
                    #{"type": "text", "text": "What's up dog!"},
                ],
            },
        ],
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    eos_token_ids = model.generation_config.eos_token_id
    max_new_tokens = 128000

    cur_input_ids = input_ids
    cur_attention_mask_length = attention_mask.shape[1]
    past_key_values = None

    with torch.inference_mode():
        for step in range(max_new_tokens):
            outputs = model(
                input_ids=cur_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                attention_mask=torch.ones(1, cur_attention_mask_length, dtype=torch.long),
            )
            past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :].float()
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)

            if next_token.item() in eos_token_ids:
                break

            decoded_token = tokenizer.decode(next_token.squeeze(0).tolist())
            sys.stdout.write(decoded_token)
            sys.stdout.flush()

            cur_input_ids = next_token
            cur_attention_mask_length += 1


if __name__ == "__main__":
    main()
