import argparse
from pathlib import Path

import torch

from export_hf import export_hf
from model import MinimalLLM
from tokenizer import HuggingFaceBpeTokenizer, SimpleTokenizer


SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "A minimal language model trained from scratch using PyTorch tensors, "
    "attention mechanisms, and rotary position embeddings.\n"
) * 100


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the educational nano-llm model.")
    parser.add_argument("--text-file", help="UTF-8 training text; defaults to a tiny built-in sample.")
    parser.add_argument("--output", default="minimal_llm.pt", help="Checkpoint file to write.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "xpu"), default="auto")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument(
        "--tokenizer-format",
        choices=("bpe", "gpt2"),
        default="bpe",
        help="Use GPT-2-compatible byte-level BPE for llama.cpp GGUF export.",
    )
    parser.add_argument(
        "--export-hf",
        help="Also export a Llama-compatible Hugging Face model directory.",
    )
    parser.add_argument("--norm", choices=("rmsnorm", "layernorm"), default="rmsnorm")
    parser.add_argument("--mlp", choices=("swiglu", "mlp"), default="swiglu")
    parser.add_argument("--no-rope", action="store_true", help="Disable rotary position embeddings.")
    parser.add_argument("--generation-tokens", type=int, default=100)
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


def get_batch(data, batch_size, block_size, device):
    starts = torch.randint(len(data) - block_size, (batch_size,))
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return inputs.to(device), targets.to(device)


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
        tokenizer.train(text, vocab_size=args.vocab_size)
    else:
        tokenizer = SimpleTokenizer()
        tokenizer.train(text, vocab_size=args.vocab_size)

    token_ids = tokenizer.encode(text)
    if isinstance(tokenizer, HuggingFaceBpeTokenizer):
        token_ids.append(tokenizer.eos_id)
    data = torch.tensor(token_ids, dtype=torch.long)
    split_at = int(0.9 * len(data))
    train_data, val_data = data[:split_at], data[split_at:]
    if len(train_data) <= args.block_size or len(val_data) <= args.block_size:
        raise ValueError("Training text is too short for the selected block-size.")

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
            for name, split in (("train", train_data), ("validation", val_data)):
                batch_losses = [
                    model(*get_batch(split, args.batch_size, args.block_size, device))[1].item()
                    for _ in range(5)
                ]
                losses[name] = sum(batch_losses) / len(batch_losses)
        model.train()
        return losses

    model.train()
    for step in range(args.steps):
        inputs, targets = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps - 1:
            losses = estimate_losses()
            print(
                f"Step {step + 1:4d} | "
                f"train {losses['train']:.4f} | validation {losses['validation']:.4f}"
            )

    context = torch.tensor([tokenizer.encode("The")], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(args.generation_tokens):
            logits, _ = model(context[:, -args.block_size :])
            probabilities = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
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
