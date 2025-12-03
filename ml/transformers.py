"""
Transformer Tutorial

This module covers Transformer architecture with PyTorch:
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- Transformer blocks
- Vision Transformers (ViT)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention mechanism."""

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Query, Key, Value projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embed_dim)

    def forward(self, x, mask=None):
        # x shape: (batch, seq_len, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Attention scores: (batch, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Output: (batch, seq_len, embed_dim)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)

        return output, attention_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    """Feed-forward network in transformer."""

    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Pass through transformer blocks
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attention_weights.append(attn)

        x = self.norm(x)

        return x, attention_weights


class PatchEmbedding(nn.Module):
    """Patch embedding for Vision Transformer."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = self.proj(x)  # (batch, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for image classification."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        num_heads=12,
        ff_dim=3072,
        num_layers=12,
        dropout=0.1,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x, _ = block(x)

        # Classification head
        x = self.norm(x)
        x = x[:, 0]  # Take class token
        x = self.head(x)

        return x


def demonstrate_attention():
    """Demonstrate attention mechanism."""
    batch_size = 2
    seq_len = 10
    embed_dim = 64

    x = torch.randn(batch_size, seq_len, embed_dim)

    # Self-attention
    self_attn = SelfAttention(embed_dim)
    output, weights = self_attn(x)

    return {
        "input_shape": x.shape,
        "output_shape": output.shape,
        "attention_weights_shape": weights.shape,
    }


def demonstrate_transformer():
    """Demonstrate transformer encoder."""
    vocab_size = 10000
    embed_dim = 256
    num_heads = 8
    ff_dim = 1024
    num_layers = 4

    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
    )

    # Sample input tokens
    x = torch.randint(0, vocab_size, (2, 32))  # (batch, seq_len)

    output, attn_weights = encoder(x)

    return {
        "input_shape": x.shape,
        "output_shape": output.shape,
        "num_attention_layers": len(attn_weights),
        "parameters": sum(p.numel() for p in encoder.parameters()),
    }


def demonstrate_vit():
    """Demonstrate Vision Transformer."""
    # Mini ViT for demonstration
    vit = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=192,
        num_heads=3,
        ff_dim=768,
        num_layers=6,
    )

    # Sample image batch
    images = torch.randn(4, 3, 32, 32)
    output = vit(images)

    return {
        "input_shape": images.shape,
        "output_shape": output.shape,
        "parameters": sum(p.numel() for p in vit.parameters()),
        "num_patches": vit.patch_embed.num_patches,
    }


if __name__ == "__main__":
    print("=== Transformer Tutorial ===")

    print("\nSelf-Attention:")
    attn_demo = demonstrate_attention()
    for name, val in attn_demo.items():
        print(f"  {name}: {val}")

    print("\nTransformer Encoder:")
    trans_demo = demonstrate_transformer()
    for name, val in trans_demo.items():
        print(f"  {name}: {val}")

    print("\nVision Transformer (ViT):")
    vit_demo = demonstrate_vit()
    for name, val in vit_demo.items():
        print(f"  {name}: {val}")

    print("\nArchitecture Components:")
    print("  - Positional Encoding: Sinusoidal encoding")
    print("  - Multi-Head Attention: Parallel attention heads")
    print("  - Feed-Forward: Two-layer MLP with GELU")
    print("  - Layer Normalization: Pre-norm architecture")
    print("  - Residual Connections: Add & Norm pattern")
