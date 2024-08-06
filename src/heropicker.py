import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class HeroPicker(nn.Module):
    def __init__(self, vocab_len, emb_dim,
                 num_heads, dim_ff, dropout_rate,
                 num_decod_layers, device):
        super(HeroPicker, self).__init__()
        self.emb_layer = nn.Embedding(num_embeddings=vocab_len, embedding_dim=emb_dim)
        self.pos_enc = PositionalEncoding(emb_dim=emb_dim, max_len=25)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=num_heads,
                                                        dim_feedforward=dim_ff, batch_first=True,
                                                        dropout=dropout_rate)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decod_layers)
        self.fc = nn.Linear(emb_dim, vocab_len)
        self.device = device

    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size), diagonal=1)).bool().to(self.device)
        return mask

    def forward(self, x):
        out = self.emb_layer(x)
        out = self.pos_enc(out)
        mask = self.generate_square_subsequent_mask(out.size(1))
        out = self.decoder(tgt=out, memory=out, tgt_mask = mask, tgt_is_causal=True)
        out = self.fc(out)
        return out    

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, max_len: int):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x