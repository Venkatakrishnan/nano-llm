import json
from collections import Counter
from pathlib import Path


class SimpleTokenizer:
    """A small byte-level BPE tokenizer intended for learning and experiments."""

    def __init__(self):
        self.tokens = []
        self.merges = []
        self.vocab_size = 0

    def train(self, text, vocab_size=512, min_frequency=2):
        if not text:
            raise ValueError("Cannot train a tokenizer on empty text.")
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for byte-level BPE.")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive.")

        self.tokens = [bytes([byte]) for byte in range(256)]
        self.merges = []
        sequences = [list(text.encode("utf-8"))]

        while len(self.tokens) < vocab_size:
            pair_counts = Counter(
                (sequence[index], sequence[index + 1])
                for sequence in sequences
                for index in range(len(sequence) - 1)
            )
            if not pair_counts:
                break

            pair = min(pair_counts, key=lambda item: (-pair_counts[item], item))
            if pair_counts[pair] < min_frequency:
                break

            new_id = len(self.tokens)
            self.tokens.append(self.tokens[pair[0]] + self.tokens[pair[1]])
            self.merges.append(pair)
            sequences = [self._merge_sequence(sequence, pair, new_id) for sequence in sequences]

        self.vocab_size = len(self.tokens)

    @staticmethod
    def _merge_sequence(sequence, pair, new_id):
        merged = []
        index = 0
        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == pair[0]
                and sequence[index + 1] == pair[1]
            ):
                merged.append(new_id)
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return merged

    def encode(self, text):
        if not self.tokens:
            raise RuntimeError("Train or load the tokenizer before encoding text.")
        sequence = list(text.encode("utf-8"))
        for merge_index, pair in enumerate(self.merges):
            sequence = self._merge_sequence(sequence, pair, 256 + merge_index)
        return sequence

    def decode(self, token_ids):
        if not self.tokens:
            raise RuntimeError("Train or load the tokenizer before decoding tokens.")
        decoded = bytearray()
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token ID {token_id} is outside the vocabulary.")
            decoded.extend(self.tokens[token_id])
        return decoded.decode("utf-8", errors="replace")

    def save(self, filepath="tokenizer.json"):
        data = {
            "tokens": [token.hex() for token in self.tokens],
            "merges": [list(pair) for pair in self.merges],
        }
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file)

    def load(self, filepath="tokenizer.json"):
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)

        tokens = [bytes.fromhex(token) for token in data["tokens"]]
        merges = [tuple(pair) for pair in data["merges"]]
        expected_base = [bytes([byte]) for byte in range(256)]
        if tokens[:256] != expected_base or len(tokens) != 256 + len(merges):
            raise ValueError("Invalid byte-level BPE vocabulary.")

        for new_id, pair in enumerate(merges, start=256):
            if (
                len(pair) != 2
                or pair[0] < 0
                or pair[1] < 0
                or pair[0] >= new_id
                or pair[1] >= new_id
                or tokens[new_id] != tokens[pair[0]] + tokens[pair[1]]
            ):
                raise ValueError("Invalid BPE merge table.")

        self.tokens = tokens
        self.merges = merges
        self.vocab_size = len(tokens)


class HuggingFaceBpeTokenizer:
    """Byte-level BPE tokenizer backed by the Hugging Face tokenizers library."""

    def __init__(self):
        self.tokenizer = None
        self.bos_token = "<|bos|>"
        self.eos_token = "<|eos|>"
        self.vocab_size = 0

    def train(self, text, vocab_size=512):
        if not text:
            raise ValueError("Cannot train a tokenizer on empty text.")
        if vocab_size < 258:
            raise ValueError("vocab_size must be at least 258 for byte-level BPE and special tokens.")

        try:
            from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
        except ModuleNotFoundError as exc:
            if exc.name != "tokenizers":
                raise
            raise RuntimeError(
                "Hugging Face BPE support requires `python -m pip install tokenizers`."
            ) from exc

        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=True,
        )
        self.tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[
                AddedToken(self.bos_token, special=True),
                AddedToken(self.eos_token, special=True),
            ],
        )
        self.tokenizer.train_from_iterator([text], trainer=trainer)
        self.vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def bos_id(self):
        if self.tokenizer is None:
            raise RuntimeError("Train or load the tokenizer first.")
        token_id = self.tokenizer.token_to_id(self.bos_token)
        if token_id is None:
            raise RuntimeError("Tokenizer vocabulary has no BOS token.")
        return token_id

    @property
    def eos_id(self):
        if self.tokenizer is None:
            raise RuntimeError("Train or load the tokenizer first.")
        token_id = self.tokenizer.token_to_id(self.eos_token)
        if token_id is None:
            raise RuntimeError("Tokenizer vocabulary has no EOS token.")
        return token_id

    def encode(self, text):
        if self.tokenizer is None:
            raise RuntimeError("Train or load the tokenizer before encoding text.")
        token_ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if text:
            token_ids.insert(0, self.bos_id)
        return token_ids

    def decode(self, token_ids):
        if self.tokenizer is None:
            raise RuntimeError("Train or load the tokenizer before decoding tokens.")
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def save(self, filepath):
        if self.tokenizer is None:
            raise RuntimeError("Train or load the tokenizer before saving it.")
        self.tokenizer.save(str(filepath))

    def load(self, filepath):
        try:
            from tokenizers import Tokenizer
        except ModuleNotFoundError as exc:
            if exc.name != "tokenizers":
                raise
            raise RuntimeError(
                "Hugging Face BPE support requires `python -m pip install tokenizers`."
            ) from exc

        self.tokenizer = Tokenizer.from_file(str(filepath))
        self.vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)
        added_tokens = self.tokenizer.get_added_tokens_decoder()
        special_tokens = {token.content for token in added_tokens.values() if token.special}
        if self.bos_token not in special_tokens or self.eos_token not in special_tokens:
            raise ValueError("Tokenizer file must define the expected BOS and EOS special tokens.")
