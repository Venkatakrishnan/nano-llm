import argparse
from pathlib import Path

import torch

from export_hf import export_hf
from model import MinimalLLM
from tokenizer import HuggingFaceBpeTokenizer, SimpleTokenizer


SAMPLE_TEXT = (
    "Human: What is a minimal language model?\n"
    "Assistant: A minimal language model trained from scratch using PyTorch tensors, "
    "attention mechanisms, and rotary position embeddings.\n\n"
) * 50


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the educational nano-llm model.")
    parser.add_argument("--text-file", help="UTF-8 training text; defaults to a tiny built-in sample.")
    parser.add_argument("--output", default="minimal_llm.pt", help="Checkpoint file to write.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "xpu"), default="auto")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument(
        "--tokenizer-format",
        choices=("bpe", "gpt2"),
        default="bpe",
        help="Use GPT-2-compatible byte-level BPE for llama.cpp GGUF export.",
    )
    parser.add_argument(
        "--tokenizer-file",
        help="Load this existing tokenizer JSON instead of training a new tokenizer.",
    )
    parser.add_argument(
        "--export-hf",
        help="Also export a Llama-compatible Hugging Face model directory.",
    )
    parser.add_argument("--norm", choices=("rmsnorm", "layernorm"), default="rmsnorm")
    parser.add_argument("--mlp", choices=("swiglu", "mlp"), default="swiglu")
    parser.add_argument("--no-rope", action="store_true", help="Disable rotary position embeddings.")
    parser.add_argument("--generation-tokens", type=int, default=100)
    parser.add_argument(
        "--no-mask-human",
        action="store_true",
        help="Disable -100 loss masking on Human: prompt tokens (train on full text stream).",
    )
    return parser.parse_args(argv)


def available_devices():
    mps = getattr(torch.backends, "mps", None)
    xpu = getattr(torch, "xpu", None)
    return {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": mps is not None and mps.is_available(),
        "xpu": xpu is not None and xpu.is_available(),
    }


def select_device(requested):
    available = available_devices()
    if requested == "auto":
        for name in ("cuda", "xpu", "mps", "cpu"):
            if available[name]:
                return torch.device(name)
    elif available[requested]:
        return torch.device(requested)
    else:
        raise RuntimeError(
            f"PyTorch device '{requested}' is not available in this installation. "
            "Use --device cpu or install a PyTorch build that supports the device."
        )


def _encode_raw(tokenizer, segment):
    """Encode a text segment without inserting extra BOS tokens mid-dialogue."""
    if isinstance(tokenizer, HuggingFaceBpeTokenizer):
        return tokenizer.tokenizer.encode(segment, add_special_tokens=False).ids
    return tokenizer.encode(segment)


def encode_chat_examples(text, tokenizer, block_size, mask_human=True):
    """
    Splits dialogue blocks on blank lines, masks Human: targets with -100, and
    pads each context window to block_size + 1. Long blocks are cropped around
    their first supervised target so a short prompt cannot mask the whole window.
    """
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    all_inputs, all_targets = [], []

    for block in blocks:
        # Skip standalone markdown section headers that have no dialogue turns
        if "Assistant:" not in block:
            continue

        lines = block.splitlines(keepends=True)
        input_ids = [tokenizer.bos_id] if isinstance(tokenizer, HuggingFaceBpeTokenizer) else []
        target_ids = [-100] * len(input_ids)

        is_assistant = False
        for line in lines:
            if line.startswith("Human:"):
                is_assistant = False
            elif line.startswith("Assistant:"):
                is_assistant = True

            ids = _encode_raw(tokenizer, line)
            input_ids.extend(ids)

            if is_assistant or not mask_human:
                target_ids.extend(ids)
            else:
                target_ids.extend([-100] * len(ids))

        # Append EOS at the end of the dialogue so the model learns when to stop
        if isinstance(tokenizer, HuggingFaceBpeTokenizer):
            input_ids.append(tokenizer.eos_id)
            target_ids.append(tokenizer.eos_id)

        max_length = block_size + 1
        valid_target_positions = [
            index
            for index, target in enumerate(target_ids[1:], start=1)
            if target != -100
        ]
        if not valid_target_positions:
            raise ValueError("Dialogue block has no supervised target tokens.")

        if len(input_ids) > max_length:
            first_target = valid_target_positions[0]
            start = max(0, first_target - block_size // 2)
            start = min(start, len(input_ids) - max_length)
            end = start + max_length
            input_ids = input_ids[start:end]
            target_ids = target_ids[start:end]
        else:
            pad_len = max_length - len(input_ids)
            pad_id = tokenizer.eos_id if isinstance(tokenizer, HuggingFaceBpeTokenizer) else 0
            input_ids.extend([pad_id] * pad_len)
            target_ids.extend([-100] * pad_len)

        all_inputs.append(input_ids[:-1])
        all_targets.append(target_ids[1:])

    if not all_inputs:
        raise ValueError("No valid 'Human:' / 'Assistant:' dialogue blocks found in training text.")

    return torch.tensor(all_inputs, dtype=torch.long), torch.tensor(all_targets, dtype=torch.long)


def get_batch(inputs_all, targets_all, batch_size, device):
    indices = torch.randint(len(inputs_all), (batch_size,))
    return inputs_all[indices].to(device), targets_all[indices].to(device)


def main(argv=None):
    args = parse_args(argv)
    if args.steps <= 0 or args.eval_interval <= 0 or args.batch_size <= 0:
        raise ValueError("steps, eval-interval, and batch-size must be positive.")
    if args.block_size <= 0 or args.generation_tokens < 0:
        raise ValueError("block-size must be positive and generation-tokens cannot be negative.")
    if args.hidden_dim <= 0 or args.heads <= 0 or args.hidden_dim % args.heads:
        raise ValueError("hidden-dim must be positive and divisible by a positive heads value.")
    if not args.no_rope and (args.hidden_dim // args.heads) % 2:
        raise ValueError("RoPE requires an even attention head dimension.")
    if args.export_hf and args.tokenizer_format != "gpt2":
        raise ValueError("--export-hf requires --tokenizer-format gpt2.")
    if args.export_hf and (args.no_rope or args.norm != "rmsnorm" or args.mlp != "swiglu"):
        raise ValueError("--export-hf requires RoPE, RMSNorm, and SwiGLU.")
    if args.export_hf and (args.hidden_dim // args.heads) % 2:
        raise ValueError("Llama RoPE export requires an even attention head dimension.")

    device = select_device(args.device)
    print(f"Using {device}.")

    text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else SAMPLE_TEXT
    if args.tokenizer_format == "gpt2":
        tokenizer = HuggingFaceBpeTokenizer()
    else:
        tokenizer = SimpleTokenizer()
    if args.tokenizer_file:
        tokenizer.load(args.tokenizer_file)
    else:
        tokenizer.train(text, vocab_size=args.vocab_size)

    inputs_all, targets_all = encode_chat_examples(
        text,
        tokenizer,
        args.block_size,
        mask_human=not args.no_mask_human,
    )
    split_at = int(0.9 * len(inputs_all))
    train_x, train_y = inputs_all[:split_at], targets_all[:split_at]
    val_x, val_y = inputs_all[split_at:], targets_all[split_at:]
    if len(train_x) == 0 or len(val_x) == 0:
        raise ValueError("Not enough dialogue blocks for a 90/10 train/validation split.")

    print(
        f"Loaded {len(inputs_all)} dialogue blocks "
        f"({len(train_x)} train / {len(val_x)} validation); "
        f"assistant-only loss masking: {not args.no_mask_human}."
    )

    model = MinimalLLM(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        use_rope=not args.no_rope,
        norm_type=args.norm,
        mlp_type=args.mlp,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model: {parameter_count / 1e6:.2f}M parameters; vocabulary: {tokenizer.vocab_size}.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    def estimate_losses():
        model.eval()
        losses = {}
        with torch.no_grad():
            for name, (split_x, split_y) in (("train", (train_x, train_y)), ("validation", (val_x, val_y))):
                total_loss = 0.0
                total_tokens = 0
                for start in range(0, len(split_x), args.batch_size):
                    inputs = split_x[start : start + args.batch_size].to(device)
                    targets = split_y[start : start + args.batch_size].to(device)
                    _, batch_loss = model(inputs, targets)
                    token_count = (targets != -100).sum().item()
                    total_loss += batch_loss.item() * token_count
                    total_tokens += token_count
                if total_tokens == 0:
                    raise ValueError(f"{name} split has no supervised target tokens.")
                losses[name] = total_loss / total_tokens
        model.train()
        return losses

    model.train()
    for step in range(args.steps):
        inputs, targets = get_batch(train_x, train_y, args.batch_size, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        completed_steps = step + 1
        if completed_steps % args.eval_interval == 0 or completed_steps == args.steps:
            losses = estimate_losses()
            print(
                f"Step {completed_steps:4d} | "
                f"train {losses['train']:.4f} | validation {losses['validation']:.4f}"
            )
    prompt_text = "Human: What is a budget for?\nAssistant: "
    context = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
    eos_id = tokenizer.eos_id if isinstance(tokenizer, HuggingFaceBpeTokenizer) else None
    model.eval()
    with torch.no_grad():
        for _ in range(args.generation_tokens):
            logits, _ = model(context[:, -args.block_size :])
            probabilities = torch.softmax(logits[:, -1, :] / 0.3, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            if eos_id is not None and next_token.item() == eos_id:
                break
            context = torch.cat((context, next_token), dim=1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_path.with_suffix(output_path.suffix + ".tokenizer.json")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.layers,
                "num_heads": args.heads,
                "block_size": args.block_size,
                "use_rope": not args.no_rope,
                "norm_type": args.norm,
                "mlp_type": args.mlp,
                "tokenizer_format": args.tokenizer_format,
                "bos_token_id": tokenizer.bos_id if isinstance(tokenizer, HuggingFaceBpeTokenizer) else None,
                "eos_token_id": tokenizer.eos_id if isinstance(tokenizer, HuggingFaceBpeTokenizer) else None,
            },
        },
        output_path,
    )
    tokenizer.save(tokenizer_path)
    print(f"\nGenerated sample:\n{tokenizer.decode(context[0].tolist())}")
    print(f"Saved checkpoint to {output_path} and tokenizer to {tokenizer_path}.")
    if args.export_hf:
        export_dir = export_hf(output_path, tokenizer_path, args.export_hf)
        print(f"Exported Hugging Face Llama model to {export_dir}.")


if __name__ == "__main__":
    main()
