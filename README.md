# nano-llm 🚀

A minimal, from-scratch implementation of a decoder-only Large Language Model (LLM) built in PyTorch. Designed for clarity, readability, and educational depth—stripping away complex distributed frameworks to focus purely on core tensor mechanics, attention mathematics, and training loops.

## 🧠 Architectural Overview

`nano-llm` follows the modern GPT/Llama decoder-only architecture:

1. **Tokenization & Embeddings**: Maps raw text inputs into discrete tokens via BPE, then projects them into continuous vector space.
2. **Positional Encodings**: Supports both **NoPE** (No Positional Encoding) and **RoPE** (Rotary Position Embeddings) to handle sequence order.
3. **Transformer Blocks**: Stacked layers featuring:
   * Multi-Head Self-Attention with Causal Masking
   * RMSNorm / LayerNorm
   * Feed-Forward Networks (MLP / SwiGLU)
4. **Language Modeling Head**: Linear projection layer mapping hidden states back to vocabulary logits for next-token prediction.

---

## 📂 Repository Structure

```text
nano-llm/
├── model.py        # Core Transformer architecture, Attention, and RoPE/NoPE modules
├── train.py        # Training loop, data loading pipeline, and AdamW optimization
├── tokenizer.py    # Token processing utilities
└── README.md       # Project documentation
