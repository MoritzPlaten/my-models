import torch

from helperClass import PositionalEncoding, EncoderLayer, DecoderLayer

class MyTransformer(torch.nn.Module):

    def __init__(self, src_vocab_size=5000, tgt_vocab_size=5000, num_layers=6, d_model=512, num_heads=8, dff=2048, max_sequ_length=100, dropout=0.1) -> None:
        super(MyTransformer, self).__init__()

        self.encoder_emb = torch.nn.Embedding(src_vocab_size, d_model)
        self.decoder_emb = torch.nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_length=max_sequ_length
        )

        self.encoder_layers = torch.nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout=dropout
            )
            for _ in range(0, num_layers)
        ])

        self.decoder_layers = torch.nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout=dropout
            )
            for _ in range(0, num_layers)
        ])

        self.output_layer = torch.nn.Linear(d_model, tgt_vocab_size)
        self.dropout = torch.nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):

        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_embbed = self.encoder_emb(src)
        src_embbed = self.pos_encoding(src_embbed)
        src_embbed = self.dropout(src_embbed)

        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(src_embbed, src_mask)

        tgt_embbed = self.decoder_emb(tgt)
        tgt_embbed = self.pos_encoding(tgt_embbed)
        tgt_embbed = self.dropout(tgt_embbed)

        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer(tgt_embbed, enc_output, src_mask, tgt_mask)

        output = self.output_layer(dec_output)
        return output