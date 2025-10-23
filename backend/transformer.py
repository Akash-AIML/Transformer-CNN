import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# ----------------- Positional Encoding -----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ----------------- Transformer Model -----------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, num_heads=8, num_encoder_layers=3, 
                 num_decoder_layers=3, dropout=0.1, output_dim=1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        # src, tgt: [batch, seq_len, features]
        src = self.input_projection(src)
        tgt = self.input_projection(tgt)
        src = self.pos_encoding(src)
        tgt = self.pos_encoding(tgt)

        # Transformer expects [seq_len, batch, d_model]
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        memory = self.encoder(src)
        out = self.decoder(tgt, memory)
        out = out.permute(1,0,2)
        return self.output_layer(out)
