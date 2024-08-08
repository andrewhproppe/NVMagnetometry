import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 embed_size: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 activation: str = "ReLU",
                 norm: bool = False):
        super(SelfAttentionBlock, self).__init__()

        self.num_heads = num_heads
        self.embed_size = embed_size

        # Linear layers for query, key, and value transformations
        self.query = nn.Linear(input_size, embed_size * num_heads)
        self.key = nn.Linear(input_size, embed_size * num_heads)
        self.value = nn.Linear(input_size, embed_size * num_heads)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=embed_size * num_heads,
                                               num_heads=num_heads,
                                               dropout=dropout)

        # Output projection
        self.proj = nn.Linear(embed_size * num_heads, output_size)

        # Activation function
        self.activation = getattr(nn, activation)()

        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_size * num_heads) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(output_size) if norm else nn.Identity()

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(output_size, output_size * 4),
            self.activation,
            nn.Linear(output_size * 4, output_size)
        )

    def forward(self, x):
        # Add a sequence length dimension (assume sequence length = 1)
        x = x.unsqueeze(1)  # Shape: (batch_size, sequence_length=1, input_size)

        # Compute query, key, and value
        queries = self.query(x).permute(1, 0, 2)  # Shape: (sequence_length, batch_size, embed_dim * num_heads)
        keys = self.key(x).permute(1, 0, 2)  # Shape: (sequence_length, batch_size, embed_dim * num_heads)
        values = self.value(x).permute(1, 0, 2)  # Shape: (sequence_length, batch_size, embed_dim * num_heads)

        # Apply multi-head attention
        attn_output, _ = self.attention(queries, keys, values)
        attn_output = attn_output.permute(1, 0, 2)  # Shape: (batch_size, sequence_length, embed_dim * num_heads)

        # Remove the sequence length dimension (squeeze it)
        attn_output = attn_output.squeeze(1)  # Shape: (batch_size, embed_dim * num_heads)

        # Add & Norm
        x = x.squeeze(1)  # Shape: (batch_size, input_size)
        x = x + attn_output
        x = self.norm1(x)

        # Feed-forward
        ff_output = self.ff(self.proj(x))

        # Add & Norm
        x = x + ff_output
        x = self.norm2(x)

        return x


# Example usage:
input_size = 128
output_size = 256
embed_size = 32  # Reduced embedding size
num_heads = 2  # Reduced number of heads
dropout = 0.1
activation = "ReLU"
norm = True

self_attention_block = SelfAttentionBlock(input_size, output_size, embed_size, num_heads, dropout, activation, norm)

x = torch.randn(32, 128)  # (batch_size, input_size)
output = self_attention_block(x)
print(output.shape)  # Should print: torch.Size([32, 256])

