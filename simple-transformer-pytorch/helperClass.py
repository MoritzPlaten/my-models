import torch

class FeedForward(torch.nn.Module):

    def __init__(self, d_model, dff) -> None:
        super(FeedForward, self).__init__()

        self.ff1 = torch.nn.Linear(d_model, dff)
        self.ff2 = torch.nn.Linear(dff, d_model)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        
        x = self.ff1(x)
        x = self.relu(x)
        x = self.ff2(x)
        return x


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class EncoderLayer(torch.nn.Module):

    def __init__(self, d_model, num_heads, dff, dropout=0.1) -> None:
        super(EncoderLayer, self).__init__()

        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            add_zero_attn=True,
            dropout=dropout
        )

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

        self.feedForw = FeedForward(
            d_model=d_model,
            dff=dff
            )

        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, mask):

        attn_out = self.self_attn(x, x, x, attn_mask=mask)
        x = x + attn_out
        x = self.norm1(x)
        x = self.dropout(x)

        feed_out = self.feedForw(x)
        x = x + feed_out
        x = self.norm2(x)
        x = self.dropout(x)

        return x
    

class DecoderLayer(torch.nn.Module):

    def __init__(self, d_model, num_heads, dff, dropout=0.1) -> None:
        super(DecoderLayer, self).__init__()

        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            add_zero_attn=True,
            dropout=dropout
        )
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            add_zero_attn=True,
            dropout=dropout
        )
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

        self.feedForw = FeedForward(
            d_model=d_model,
            dff=dff
        )
        self.norm3 = torch.nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask, tgt_mask):

        attn_out = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = x + attn_out
        x = self.norm1(x)

        cross_out = self.cross_attn(enc_output, enc_output, x, attn_mask=src_mask)
        x = x + cross_out
        x = self.norm2(x)

        feed_out = self.feedForw(x)
        x = x + feed_out
        x = self.norm3(x)

        return x
