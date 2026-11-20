# nano-llm

A small, educational decoder-only language model implemented in PyTorch. It is
designed to make the main pieces of a transformer readable rather than to
produce a useful general-purpose language model.

## Architecture

- **Tokenizer:** a from-scratch byte-level BPE tokenizer. It learns merges from
  the training text and can save/load its vocabulary.
- **Position information:** rotary position embeddings (RoPE), or no positional
  encoding with `--no-rope`.
- **Attention:** multi-head self-attention with a causal mask.
- **Normalization:** RMSNorm by default, selectable LayerNorm with `--norm layernorm`.
- **Feed-forward:** SwiGLU by default, selectable GELU MLP with `--mlp mlp`.
- **Language-model head:** next-token logits with tied input-embedding/output
  weights.

The training script uses a tiny repeated sample by default. Supply a UTF-8 text
file with `--text-file` for your own experiment; this is not a pretrained model
or a production training pipeline.

## Run on Windows 11

Use Python 3.10 or later. In PowerShell, from the repository folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe train.py --device cpu --steps 100 --batch-size 8 --hidden-dim 128 --layers 2
```

The example deliberately uses the CPU PyTorch build. The training script can
also select CUDA, MPS, or XPU if the installed PyTorch build reports that device
as available. `--device auto` selects an available backend, otherwise CPU.
For example, `python train.py --device xpu` requests PyTorch's XPU backend; an
Intel graphics driver or a Vulkan installation alone does not provide that
backend.

Training writes `minimal_llm.pt` and `minimal_llm.pt.tokenizer.json` in the
current folder. Pass `--help` to see options for the text file, model
architecture, training duration, and output path.

### Suggested code-reading order

- [`model.py`](./model.py): token embeddings, causal self-attention, transformer
  blocks, normalization, feed-forward layers, and next-token logits.
- [`tokenizer.py`](./tokenizer.py): the educational byte-level BPE tokenizer
  and the optional Hugging Face GPT-2-style tokenizer.
- [`train.py`](./train.py): command-line arguments, dialogue encoding and loss
  masks, batches, optimizer updates, train/validation evaluation, generation,
  and checkpoint saving.
- [`export_hf.py`](./export_hf.py): maps this project's Llama-shaped weights
  into the Hugging Face Llama file layout for the optional GGUF route.
- [`test_model.py`](./test_model.py): small tests of tokenizer, masking,
  architecture, training, and export behavior.

For a line-by-line walkthrough, follow `main()` in `train.py` into
`encode_chat_examples()`, then `get_batch()`, `MinimalLLM.forward()` in
`model.py`, the optimizer update, `estimate_losses()`, and finally generation
and saving. This follows the tensors from text to predicted next-token IDs.

## Training concepts and code walkthrough

This project initializes a small model with random weights and trains it on
text. It does not start from a pretrained assistant. The training flow in
`train.py` is:

1. Read the UTF-8 text and train a tokenizer, or load one with `--tokenizer-file`.
   The tokenizer turns text into integer token IDs.
2. Split the blank-line-separated dialogue blocks into the first 90% for
   training and the last 10% for validation. Shuffle the blocks before running
   if their current order is grouped by topic.
3. Encode each block into an input sequence and a one-token-shifted target
   sequence. The model uses the input to predict the next token.
4. For each training step, sample `--batch-size` dialogue blocks, run the model,
   calculate cross-entropy loss, backpropagate gradients, and update the
   parameters once with AdamW.
5. At each evaluation interval, calculate loss across the full training split
   and full held-out validation split without updating weights. The script saves
   the final checkpoint; it does not save or resume the best-validation
   checkpoint.
6. Generate a sample from the fixed prompt `Human: What is a budget for?`
   followed by `Assistant:`, then save the model state and tokenizer.

**A step is one optimizer update, not one pass through the dataset (an epoch).**
Each step samples dialogue blocks randomly, so examples can be sampled more than
once or skipped during a run. `--batch-size` is the number of blocks in an
update; `--block-size` limits the token context used from each block. A printed
`Step N` counts N completed updates. `--eval-interval 200` evaluates at steps
200, 400, and so on, plus the final step.

**Training loss** measures next-token prediction on the training blocks.
**Validation loss** measures the same objective on held-out blocks that do not
update the model. Lower values are better. If training loss keeps falling while
validation loss rises across evaluations, the model may be overfitting. A
validation minimum identifies the best observed step, but this script currently
only saves the final step's weights and does not save optimizer state for
resuming.

### Assistant-only loss masking

Chat examples are separated by blank lines and use `Human:` and `Assistant:`
labels. By default, `train.py` leaves Human tokens in the input context but
masks their target labels with `-100`; PyTorch ignores those labels in the
cross-entropy loss. The loss is therefore applied to Assistant replies, not the
Human prompt. `--no-mask-human` disables this masking and trains on both sides
of each dialogue. This is a small supervised fine-tuning-style objective, but
the model still starts with random weights; it is not PEFT/LoRA fine-tuning.

The shuffled finance examples intentionally include some explanatory text in
the `Human:` context. That text is visible to the model as prompt context, while
the default loss mask still applies only to Assistant targets. The training
code treats unprefixed continuation lines as part of the preceding role. Long
blocks are cropped around the first supervised target to keep assistant tokens
inside the context window.

The generated sample uses temperature 0.3 and stops at EOS when the GPT-2
tokenizer is used. It is a stochastic continuation, not a measure of chat
accuracy. This small, from-scratch model and educational corpus are not enough
to create a reliable general-purpose assistant.

## Educational personal-finance examples

The `examples` folder contains original role-labeled dialogue examples, a
shuffled and augmented Q&A corpus, a plain-language lesson handout, and a
script for regenerating the scenario-based examples. The `personal_finance_chat`
corpus is about 180 KB. Regenerate its added practice dialogues with:

```powershell
.\.venv\Scripts\python.exe .\examples\expand_personal_finance_chat.py
```

Train from the chat corpus with assistant-only loss masking:

```powershell
.\.venv\Scripts\python.exe train.py `
  --text-file ".\examples\shuffled_qa.txt" `
  --tokenizer-format gpt2 `
  --output ".\finance-masked.pt" `
  --device cpu `
  --steps 600 `
  --eval-interval 100 `
  --batch-size 16 `
  --block-size 256
```

The first run learns and saves a GPT-2-style tokenizer beside the checkpoint.
For later runs, reuse that tokenizer with
`--tokenizer-file ".\finance-masked.pt.tokenizer.json"`; this keeps token IDs
stable but still initializes new model weights. `--vocab-size` controls the
target size only when creating a tokenizer; the actual learned size may be
smaller, and loading a tokenizer file uses its existing vocabulary instead.

For the full optional Hugging Face-to-GGUF-to-Ollama workflow, run
`.\run-masked.ps1` from PowerShell. It uses `.venv\Scripts\python.exe` if
present, otherwise Python on `PATH`; install PyTorch, `tokenizers`, and the
llama.cpp converter dependencies in that same Python environment. It also
expects the llama.cpp checkout at `..\llama.cpp` and Ollama installed. On the
first run it learns a GPT-2-style tokenizer and saves it beside
`finance-masked.pt`; later runs reuse it so token IDs stay fixed. If llama.cpp
does not recognize the new tokenizer fingerprint, follow the vocabulary-only
lookup below before rerunning the script. It exports F16 GGUF and creates or
updates the ignored local `Modelfile`.

The finance material is general education, not individualized financial,
investment, tax, or legal advice. The assistant-only masking is a teaching
example, not a guarantee of safe or correct financial answers.

## Export a Llama-compatible GGUF for Ollama

The default byte-level BPE tokenizer is specific to this project and cannot be
used by the standard llama.cpp converter. To export a model, install the
optional Hugging Face `tokenizers` package and **retrain** using its
GPT-2-compatible byte-level BPE tokenizer and the compatible architecture:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -r requirements-gguf.txt
.\.venv\Scripts\python.exe train.py `
  --text-file ".\examples\shuffled_qa.txt" `
  --tokenizer-format gpt2 `
  --tokenizer-file ".\finance-masked.pt.tokenizer.json" `
  --output ".\finance-masked-export.pt" `
  --export-hf ".\finance-masked-hf" `
  --device cpu `
  --steps 600 `
  --eval-interval 100 `
  --batch-size 4 `
  --block-size 256 `
  --hidden-dim 192 `
  --layers 4 `
  --heads 4
```

`--export-hf` maps the trained weights into the supported Llama model layout.
It requires RoPE, RMSNorm, SwiGLU, tied embeddings, and the compatible GPT-2
byte-level BPE tokenizer. The export directory contains the model config,
mapped weights, and Hugging Face `tokenizer.json` needed by llama.cpp's BPE
vocabulary importer; SentencePiece is not required. Existing checkpoints
trained with the default tokenizer cannot be converted by this exporter;
retraining creates a new, different model.

To avoid a new llama.cpp fingerprint every time you train on different text,
reuse a saved GPT-2 tokenizer instead of learning a new vocabulary. For
example, preserve the tokenizer JSON from a successful run and pass it with
`--tokenizer-file`:

```powershell
.\.venv\Scripts\python.exe train.py `
  --text-file ".\examples\personal_finance_chat.txt" `
  --tokenizer-format gpt2 `
  --tokenizer-file ".\finance-masked.pt.tokenizer.json" `
  --output ".\finance-next.pt" `
  --export-hf ".\finance-next-hf" `
  --device cpu `
  --steps 1000
```

The vocabulary size and token IDs then stay fixed even when the training text
changes, so the existing llama.cpp fingerprint rule continues to apply.
`--tokenizer-file` reuses only the tokenizer; it does not resume model weights
from an earlier checkpoint. Keep a backup of the tokenizer you want to reuse.

With llama.cpp's Python conversion dependencies installed, convert the
`finance-masked-hf` export. The converter needs a trained GPT-2-style tokenizer;
it does not require the Vulkan backend. A quantizer executable is optional.
llama.cpp recognizes some BPE pre-tokenizers by a fingerprint of their token
IDs, so a newly trained vocabulary may not match its built-in list. To inspect
the fingerprint for the current export without converting model weights, run
this from the repository root:

```powershell
.\.venv\Scripts\python.exe ..\llama.cpp\convert_hf_to_gguf.py --vocab-only --outfile "$env:TEMP\finance-tokenizer-check.gguf" .\finance-masked-hf
```

If the tokenizer is not recognized, the command exits with an error but prints
`chkhsh: <fingerprint>` in the warning. That is expected for this lookup. In
`llama.cpp\conversion\base.py`, find `get_vocab_base_pre`, and add a rule after
`res = None`, replacing the placeholder with the fingerprint printed for your
current export. This is a local workaround in the separate llama.cpp checkout,
not a change to this repository:

```python
if chkhsh == "PASTE_CURRENT_FINGERPRINT_HERE":
    res = "gpt-2"
```

This identifies the project's GPT-2-style byte-level pre-tokenizer; it doesn't
change the model or vocabulary. Keep reusing the tokenizer saved beside the
checkpoint to avoid generating a different fingerprint on later runs. After
adding the rule, rerun the `--vocab-only` command; once it succeeds, run the
full conversion. The temporary vocabulary-only output can be removed afterward.

```powershell
.\.venv\Scripts\python.exe ..\llama.cpp\convert_hf_to_gguf.py .\finance-masked-hf --outfile .\finance-masked-f16.gguf --outtype f16
```

If you have built llama.cpp's quantizer, you can optionally make a smaller
quantized file:

```powershell
..\llama.cpp\build\bin\Release\llama-quantize.exe .\finance-masked-f16.gguf .\finance-masked-Q4_K_M.gguf Q4_K_M
```

For quantized import, create a file named `Modelfile` containing:

```text
FROM ./finance-masked-Q4_K_M.gguf
```

Or use the F16 file directly by changing the `FROM` line to
`FROM ./finance-masked-f16.gguf`. Then import and run with Ollama:

```powershell
ollama create nano-llm -f .\Modelfile
ollama run nano-llm
```

## Vulkan and training hardware

PyTorch training does not use llama.cpp's Vulkan backend. A converted GGUF can
be run by your Vulkan-enabled llama.cpp/Ollama installation; inference
acceleration depends on that build and available GPU memory. For example, from
a llama.cpp checkout with the Vulkan build already configured:

```powershell
.\build\bin\Release\llama-cli.exe -m C:\models\model.gguf -ngl 99 -p "Write a short greeting."
```

Replace the model path with a GGUF model you already have; lower `-ngl` if the
GPU runs out of memory.
