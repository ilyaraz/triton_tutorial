import torch
import torch.nn.functional as F
from transformers import Gemma3TextConfig, Gemma3ForCausalLM, GemmaTokenizerFast
import sys


def main():
    device = torch.device("mps") 

    model_id = "google/gemma-3-1b-it"
    tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
    model = Gemma3ForCausalLM.from_pretrained(
        model_id,
        config=Gemma3TextConfig.from_pretrained(model_id),
        dtype=torch.float16,
        attn_implementation="eager",
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

    cur_input_ids = inputs["input_ids"].to(device)
    cur_attention_mask_length = inputs["attention_mask"].shape[1]
    past_key_values = None
    eos_token_ids = model.generation_config.eos_token_id

    max_new_tokens = 32000
    with torch.inference_mode():
        for step in range(max_new_tokens):
            outputs = model(
                input_ids=cur_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                attention_mask=torch.ones(1, cur_attention_mask_length, dtype=torch.long),
            )

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
            past_key_values = outputs.past_key_values


if __name__ == "__main__":
    main()
