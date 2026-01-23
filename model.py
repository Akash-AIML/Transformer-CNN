import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# -------------------------
# CNN Blocks
# -------------------------
class DilatedCausalCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(DilatedCausalCNN, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x):
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x


class PointwiseCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointwiseCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# -------------------------
# Attention
# -------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        Q = self.q_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_linear(context)

# -------------------------
# Feed Forward
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super(FeedForward, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

# -------------------------
# Encoder / Decoder Blocks
# -------------------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.pointwise_cnn = PointwiseCNN(d_model, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.causal_cnn = DilatedCausalCNN(d_model, d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        x = x + self.dropout(self.pointwise_cnn(self.norm2(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.dropout(self.causal_cnn(self.norm3(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.dropout(self.ffn(self.norm4(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.causal_cnn = DilatedCausalCNN(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.pointwise_cnn = PointwiseCNN(d_model, d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output=None):
        x = x + self.dropout(self.causal_cnn(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.dropout(self.self_attn(self.norm2(x)))
        x = x + self.dropout(self.pointwise_cnn(self.norm3(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.dropout(self.ffn(self.norm4(x)))
        return x

# -------------------------
# Full Model
# -------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dropout=0.1,
        output_dim=1,
    ):
        super(TimeSeriesTransformer, self).__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(self, src, tgt):
        src = self.pos_encoding(self.input_projection(src))
        for layer in self.encoder_layers:
            src = layer(src)
        src = self.encoder_norm(src)

        tgt = self.pos_encoding(self.input_projection(tgt))
        for layer in self.decoder_layers:
            tgt = layer(tgt)
        tgt = self.decoder_norm(tgt)

        return self.output_projection(tgt)
