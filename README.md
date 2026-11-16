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

## Export a Llama-compatible GGUF for Ollama

The default byte-level BPE tokenizer is specific to this project and cannot be
used by the standard llama.cpp converter. To export a model, install the
optional Hugging Face `tokenizers` package and **retrain** using its
GPT-2-compatible byte-level BPE tokenizer and the compatible architecture:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -r requirements-gguf.txt
.\.venv\Scripts\python.exe train.py `
  --text-file ".\data\wikipedia.txt" `
  --tokenizer-format gpt2 `
  --export-hf ".\llama-hf" `
  --device cpu `
  --steps 1000 `
  --eval-interval 100 `
  --batch-size 4 `
  --block-size 256 `
  --hidden-dim 128 `
  --layers 2 `
  --heads 4 `
  --vocab-size 512
```

`--export-hf` maps the trained weights into the supported Llama model layout.
It requires RoPE, RMSNorm, SwiGLU, tied embeddings, and the compatible GPT-2
byte-level BPE tokenizer. The export directory contains the model config,
mapped weights, and Hugging Face `tokenizer.json` needed by llama.cpp's BPE
vocabulary importer; SentencePiece is not required. Existing checkpoints
trained with the default tokenizer cannot be converted by this exporter;
retraining creates a new, different model.

With llama.cpp's conversion dependencies installed and its Vulkan build
available, convert and quantize the exported model:

```powershell
python .\llama.cpp\convert_hf_to_gguf.py .\llama-hf --outfile .\nano-llm-f16.gguf --outtype f16
.\llama.cpp\build\bin\Release\llama-quantize.exe .\nano-llm-f16.gguf .\nano-llm-Q4_K_M.gguf Q4_K_M
```

Create a file named `Modelfile` containing:

```text
FROM ./nano-llm-Q4_K_M.gguf
```

Then import and run it with Ollama:

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
