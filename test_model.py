import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from export_hf import export_hf, map_to_llama
from model import MinimalLLM
from tokenizer import HuggingFaceBpeTokenizer, SimpleTokenizer
from train import encode_chat_examples, main


class SimpleTokenizerTests(unittest.TestCase):
    def test_bpe_round_trip_and_serialization(self):
        text = "Hello, byte-level BPE! Καλημέρα 🌍 " * 4
        tokenizer = SimpleTokenizer()
        tokenizer.train(text, vocab_size=300)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            loaded = SimpleTokenizer()
            loaded.load(path)
            self.assertEqual(loaded.encode(text), tokenizer.encode(text))
            self.assertEqual(loaded.decode(loaded.encode(text)), text)


class MinimalLLMTests(unittest.TestCase):
    def test_chat_encoder_masks_user_text_and_keeps_targets_in_short_windows(self):
        tokenizer = SimpleTokenizer()
        text = (
            "Human: hidden prompt text that is longer than the context window.\n"
            "Assistant: visible reply text should remain trainable.\n\n"
        )
        tokenizer.train(text, vocab_size=256)

        inputs, targets = encode_chat_examples(text, tokenizer, block_size=48)
        supervised_text = tokenizer.decode(targets[0][targets[0] != -100].tolist())
        self.assertIn("Assistant:", supervised_text)
        self.assertIn("visible", supervised_text)
        self.assertNotIn("hidden prompt", supervised_text)

        model = MinimalLLM(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
        )
        _, loss = model(inputs, targets)
        self.assertTrue(torch.isfinite(loss).item())

    def test_forward_supports_documented_architecture_options(self):
        tokens = torch.randint(0, 300, (2, 6))
        targets = torch.randint(0, 300, (2, 6))

        for use_rope in (False, True):
            for norm_type in ("rmsnorm", "layernorm"):
                for mlp_type in ("mlp", "swiglu"):
                    model = MinimalLLM(
                        vocab_size=300,
                        hidden_dim=16,
                        num_layers=1,
                        num_heads=4,
                        use_rope=use_rope,
                        norm_type=norm_type,
                        mlp_type=mlp_type,
                    )
                    logits, loss = model(tokens, targets)
                    self.assertEqual(logits.shape, (2, 6, 300))
                    self.assertTrue(torch.isfinite(loss).item())

    def test_llama_weight_mapping_matches_model_parameters(self):
        model = MinimalLLM(
            vocab_size=300,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            use_rope=True,
            norm_type="rmsnorm",
            mlp_type="swiglu",
        )
        mapped, config = map_to_llama(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": 300,
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 4,
                    "block_size": 8,
                    "use_rope": True,
                    "norm_type": "rmsnorm",
                    "mlp_type": "swiglu",
                    "tokenizer_format": "gpt2",
                    "bos_token_id": 0,
                    "eos_token_id": 1,
                },
            }
        )
        self.assertEqual(config["model_type"], "llama")
        self.assertTrue(
            torch.equal(mapped["model.layers.0.self_attn.o_proj.weight"], model.layers[0].attn.out_proj.weight)
        )
        self.assertTrue(
            torch.equal(mapped["model.layers.0.mlp.gate_proj.weight"], model.layers[0].mlp.gate_proj.weight)
        )

    def test_training_entrypoint_generates_and_saves_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--device",
                        "cpu",
                        "--steps",
                        "1",
                        "--eval-interval",
                        "1",
                        "--batch-size",
                        "2",
                        "--block-size",
                        "4",
                        "--hidden-dim",
                        "16",
                        "--layers",
                        "1",
                        "--heads",
                        "4",
                        "--vocab-size",
                        "256",
                        "--generation-tokens",
                        "2",
                        "--output",
                        str(checkpoint),
                    ]
                )
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(Path(str(checkpoint) + ".tokenizer.json").is_file())


@unittest.skipUnless(importlib.util.find_spec("tokenizers"), "The tokenizers package is optional.")
class LlamaExportTests(unittest.TestCase):
    def test_gpt2_bpe_round_trip_and_serialization(self):
        text = ("A small Llama-compatible tokenizer test. Καλημέρα 🌍\n") * 8
        tokenizer = HuggingFaceBpeTokenizer()
        tokenizer.train(text, vocab_size=300)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            loaded = HuggingFaceBpeTokenizer()
            loaded.load(path)
            self.assertEqual(loaded.encode(text), tokenizer.encode(text))
            self.assertEqual(loaded.decode(loaded.encode(text)), text)

    def test_export_maps_checkpoint_to_llama_huggingface_layout(self):
        text = ("A tiny model mapping test with byte fallback. " * 20)
        tokenizer = HuggingFaceBpeTokenizer()
        tokenizer.train(text, vocab_size=300)

        model = MinimalLLM(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            use_rope=True,
            norm_type="rmsnorm",
            mlp_type="swiglu",
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 4,
                "block_size": 8,
                "use_rope": True,
                "norm_type": "rmsnorm",
                "mlp_type": "swiglu",
                "tokenizer_format": "gpt2",
                "bos_token_id": tokenizer.bos_id,
                "eos_token_id": tokenizer.eos_id,
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoint.pt"
            tokenizer_path = root / "tokenizer.json"
            export_dir = root / "hf-model"
            torch.save(checkpoint, checkpoint_path)
            tokenizer.save(tokenizer_path)
            export_hf(checkpoint_path, tokenizer_path, export_dir)

            exported_config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
            exported_tokenizer = json.loads((export_dir / "tokenizer.json").read_text(encoding="utf-8"))
            exported_weights = torch.load(
                export_dir / "pytorch_model.bin",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(exported_config["model_type"], "llama")
            self.assertEqual(exported_config["vocab_size"], tokenizer.vocab_size)
            self.assertEqual(exported_tokenizer["model"]["type"], "BPE")
            self.assertFalse(exported_tokenizer["model"].get("byte_fallback", False))
            self.assertEqual(exported_tokenizer["decoder"]["type"], "ByteLevel")
            self.assertEqual(exported_tokenizer["model"]["vocab"][tokenizer.bos_token], tokenizer.bos_id)
            self.assertEqual(exported_tokenizer["model"]["vocab"][tokenizer.eos_token], tokenizer.eos_id)
            self.assertTrue(
                torch.equal(
                    exported_weights["model.layers.0.self_attn.o_proj.weight"],
                    model.layers[0].attn.out_proj.weight,
                )
            )
            self.assertTrue(
                torch.equal(
                    exported_weights["model.layers.0.mlp.gate_proj.weight"],
                    model.layers[0].mlp.gate_proj.weight,
                )
            )
            self.assertTrue((export_dir / "tokenizer.json").is_file())

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "Transformers is optional.")
    def test_exported_tokenizer_has_no_out_of_range_default_unknown_token(self):
        from transformers import AutoTokenizer

        text = "A tokenizer compatibility test. " * 10
        tokenizer = HuggingFaceBpeTokenizer()
        tokenizer.train(text, vocab_size=300)
        model = MinimalLLM(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            use_rope=True,
            norm_type="rmsnorm",
            mlp_type="swiglu",
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 4,
                "block_size": 8,
                "use_rope": True,
                "norm_type": "rmsnorm",
                "mlp_type": "swiglu",
                "tokenizer_format": "gpt2",
                "bos_token_id": tokenizer.bos_id,
                "eos_token_id": tokenizer.eos_id,
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoint.pt"
            tokenizer_path = root / "tokenizer.json"
            export_dir = root / "hf-model"
            torch.save(checkpoint, checkpoint_path)
            tokenizer.save(tokenizer_path)
            export_hf(checkpoint_path, tokenizer_path, export_dir)

            exported_tokenizer = AutoTokenizer.from_pretrained(export_dir, local_files_only=True)
            self.assertLess(max(exported_tokenizer.get_vocab().values()), tokenizer.vocab_size)

    def test_training_can_export_llama_model_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            export_dir = root / "llama-hf"
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--device",
                        "cpu",
                        "--steps",
                        "1",
                        "--eval-interval",
                        "1",
                        "--batch-size",
                        "2",
                        "--block-size",
                        "8",
                        "--hidden-dim",
                        "16",
                        "--layers",
                        "1",
                        "--heads",
                        "4",
                        "--vocab-size",
                        "300",
                        "--tokenizer-format",
                        "gpt2",
                        "--export-hf",
                        str(export_dir),
                        "--generation-tokens",
                        "2",
                        "--output",
                        str(checkpoint),
                    ]
                )
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((export_dir / "config.json").is_file())
            self.assertTrue((export_dir / "pytorch_model.bin").is_file())
            self.assertTrue((export_dir / "tokenizer.json").is_file())

    def test_training_reuses_existing_gpt2_tokenizer(self):
        from tokenizers import Tokenizer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_tokenizer_path = root / "frozen-tokenizer.json"
            training_text_path = root / "new-training-text.txt"
            checkpoint = root / "model.pt"

            tokenizer = HuggingFaceBpeTokenizer()
            tokenizer.train("Tokenizer vocabulary from an earlier corpus. " * 20, vocab_size=300)
            tokenizer.save(source_tokenizer_path)
            training_text_path.write_text(
                "Human: A different prompt reuses the saved token IDs.\n"
                "Assistant: This reply is trained with the existing vocabulary.\n\n" * 20,
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--text-file",
                        str(training_text_path),
                        "--tokenizer-format",
                        "gpt2",
                        "--tokenizer-file",
                        str(source_tokenizer_path),
                        "--device",
                        "cpu",
                        "--steps",
                        "1",
                        "--eval-interval",
                        "1",
                        "--batch-size",
                        "2",
                        "--block-size",
                        "8",
                        "--hidden-dim",
                        "16",
                        "--layers",
                        "1",
                        "--heads",
                        "4",
                        "--generation-tokens",
                        "2",
                        "--output",
                        str(checkpoint),
                    ]
                )

            checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint_data["config"]["vocab_size"], tokenizer.vocab_size)
            reused_tokenizer = Tokenizer.from_file(str(checkpoint.with_suffix(".pt.tokenizer.json")))
            self.assertEqual(
                reused_tokenizer.get_vocab(),
                Tokenizer.from_file(str(source_tokenizer_path)).get_vocab(),
            )


if __name__ == "__main__":
    unittest.main()
