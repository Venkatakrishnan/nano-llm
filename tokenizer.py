import os

class SimpleTokenizer:
    """A minimal character-level tokenizer for the nano-llm repository."""
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def train(self, text):
        """Extract unique characters from text and build vocabulary mappings."""
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        """Encode a string into a list of integer token IDs."""
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l):
        """Decode a list of integer token IDs back into a string."""
        return "".join([self.itos[i] for i in l if i in self.itos])

    def save(self, filepath="tokenizer.vocab"):
        """Save vocabulary mappings."""
        import json
        with open(filepath, "w") as f:
            json.dump({"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}, f)

    def load(self, filepath="tokenizer.vocab"):
        """Load vocabulary mappings."""
        import json
        with open(filepath, "r") as f:
            data = json.load(f)
            self.stoi = data["stoi"]
            self.itos = {int(k): v for k, v in data["itos"].items()}
            self.vocab_size = len(self.stoi)
