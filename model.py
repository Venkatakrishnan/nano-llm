import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings for attention queries and keys."""

    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE requires an even attention head dimension.")

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cos_cached", torch.empty(0, dim), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0, dim), persistent=False)
        self._set_cache(max_seq_len, inv_freq.device)

    def _set_cache(self, seq_len, device):
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def forward(self, x, seq_len=None):
        seq_len = x.size(-2) if seq_len is None else seq_len
        if seq_len > self.cos_cached.size(0):
            self._set_cache(seq_len, x.device)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x_normed * self.weight


def make_norm(norm_type, dim):
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError("norm_type must be 'rmsnorm' or 'layernorm'.")


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, use_rope=True):
        super().__init__()
        if num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by a positive num_heads.")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_rope = use_rope

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rope(q, seq_len=seq_len)
            q, k = apply_rope(q, k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, mlp_type):
        super().__init__()
        intermediate_dim = 4 * hidden_dim
        self.mlp_type = mlp_type

        if mlp_type == "mlp":
            self.network = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Linear(intermediate_dim, hidden_dim),
            )
        elif mlp_type == "swiglu":
            self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        else:
            raise ValueError("mlp_type must be 'mlp' or 'swiglu'.")

    def forward(self, x):
        if self.mlp_type == "mlp":
            return self.network(x)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, use_rope=True, norm_type="rmsnorm", mlp_type="swiglu"):
        super().__init__()
        self.norm1 = make_norm(norm_type, hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, use_rope=use_rope)
        self.norm2 = make_norm(norm_type, hidden_dim)
        self.mlp = FeedForward(hidden_dim, mlp_type)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MinimalLLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        use_rope=True,
        norm_type="rmsnorm",
        mlp_type="swiglu",
    ):
        super().__init__()
        if vocab_size <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("vocab_size, hidden_dim, and num_layers must be positive.")

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim,
                    num_heads,
                    use_rope=use_rope,
                    norm_type=norm_type,
                    mlp_type=mlp_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = make_norm(norm_type, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)
        for layer in self.layers:
            x = layer(x)

        logits = self.lm_head(self.norm_final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
