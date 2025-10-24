import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import re

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_DIM = 256
N_HEADS = 2
ENC_LAYERS = 2
DEC_LAYERS = 2
DROPOUT = 0.1
FFN_DIM = 512
MAX_LEN = 64
PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"
MASK = "<mask>"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        b, s, d = x.size()
        x = x.view(b, s, self.n_heads, self.d_k).transpose(1, 2)
        return x

    def combine_heads(self, x):
        b, h, s, dk = x.size()
        return x.transpose(1, 2).contiguous().view(b, s, h * dk)

    def forward(self, q, k, v, mask=None):
        Q = self.split_heads(self.q_lin(q))
        K = self.split_heads(self.k_lin(k))
        V = self.split_heads(self.v_lin(v))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-1e9"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = self.combine_heads(out)
        return self.out_lin(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        sa = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(sa))
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        sa = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(sa))
        ca = self.cross_attn(x, enc_out, enc_out, mask=memory_mask)
        x = self.norm2(x + self.dropout(ca))
        ff = self.ff(x)
        x = self.norm3(x + self.dropout(ff))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.tok_emb(src) * math.sqrt(self.tok_emb.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, tgt_mask=None, memory_mask=None):
        x = self.tok_emb(tgt) * math.sqrt(self.tok_emb.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, memory_mask)
        x = self.norm(x)
        return self.out(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, enc_layers, dec_layers, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.enc = Encoder(vocab_size, d_model, enc_layers, n_heads, d_ff, dropout)
        self.dec = Decoder(vocab_size, d_model, dec_layers, n_heads, d_ff, dropout)

    def make_src_mask(self, src):
        return (src != 0).long()

    def make_tgt_mask(self, tgt):
        b, seq = tgt.size()
        pad_mask = (tgt != 0).long()
        subsequent = torch.tril(torch.ones((seq, seq), device=tgt.device)).long()
        return (pad_mask.unsqueeze(1) * subsequent.unsqueeze(0)).to(tgt.device)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.enc(src, src_mask)
        logits = self.dec(tgt, enc_out, tgt_mask=tgt_mask, memory_mask=src_mask)
        return logits

class UrduChatbot:
    def __init__(self, model_path="best_transformer_bleu.pt", vocab_path="vocab.txt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = MAX_LEN
        
        # Load vocabulary
        self.stoi, self.itos = self.load_vocab(vocab_path)
        self.vocab_size = len(self.itos)
        self.pad_idx = self.stoi[PAD]
        self.sos_idx = self.stoi[SOS]
        self.eos_idx = self.stoi[EOS]
        self.unk_idx = self.stoi[UNK]
        self.mask_idx = self.stoi[MASK]
        
        # Initialize model
        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            d_model=EMB_DIM,
            enc_layers=ENC_LAYERS,
            dec_layers=DEC_LAYERS,
            n_heads=N_HEADS,
            d_ff=FFN_DIM,
            dropout=DROPOUT
        ).to(self.device)
        
        # Load model weights
        self.load_model(model_path)
        self.model.eval()
    
    def load_vocab(self, vocab_path):
        """Load vocabulary from file."""
        itos = []
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                tok = line.strip()
                if tok and tok not in itos:
                    itos.append(tok)
        # Ensure special tokens
        for tok in [PAD, SOS, EOS, UNK, MASK]:
            if tok not in itos:
                itos.insert(0, tok)
        stoi = {w: i for i, w in enumerate(itos)}
        return stoi, itos
    
    def load_model(self, model_path):
        """Load model weights."""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file {model_path} not found")
    
    def normalize_urdu(self, text):
        """Normalize Urdu text."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]', '', text)
        text = text.replace('\u0640', '')
        text = re.sub('[\u0622\u0623\u0625]', 'ا', text)
        text = re.sub('[\u064A\u06D0]', 'ی', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def encode_text(self, text):
        """Encode text to token IDs."""
        toks = str(text).split()
        ids = [self.stoi.get(t, self.unk_idx) for t in toks]
        ids = [self.sos_idx] + ids + [self.eos_idx]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            if ids[-1] != self.eos_idx:
                ids[-1] = self.eos_idx
        return ids
    
    def pad_seq(self, seq):
        """Pad sequence to max length."""
        if len(seq) < self.max_len:
            return seq + [self.pad_idx] * (self.max_len - len(seq))
        return seq[:self.max_len]
    
    def ids_to_sentence(self, ids):
        """Convert token IDs back to text."""
        toks = []
        for i in ids:
            if i == self.pad_idx or i == self.sos_idx or i == self.eos_idx:
                continue
            toks.append(self.itos[i] if i < len(self.itos) else UNK)
        return " ".join(toks)
    
    def generate_response(self, text):
        """Generate response for given input text."""
        normalized_text = self.normalize_urdu(text)
        src_ids = self.pad_seq(self.encode_text(normalized_text))
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            enc_out = self.model.enc(src_tensor, self.model.make_src_mask(src_tensor))
            ys = torch.full((1, 1), self.sos_idx, dtype=torch.long, device=self.device)

            for _ in range(self.max_len - 1):
                tgt_mask = self.model.make_tgt_mask(ys)
                out = self.model.dec(ys, enc_out, tgt_mask=tgt_mask, 
                                   memory_mask=self.model.make_src_mask(src_tensor))
                next_logits = out[:, -1, :]
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == self.eos_idx:
                    break

            pred = self.ids_to_sentence(ys[0].cpu().tolist())
        return pred
