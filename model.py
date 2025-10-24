# Add this to the UrduChatbot class in model.py
import os
from pathlib import Path

class UrduChatbot:
    def __init__(self, model_path="best_transformer_bleu.pt", vocab_path="vocab.txt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = MAX_LEN
        
        # Handle file paths - look in current directory
        self.model_path = model_path
        self.vocab_path = vocab_path
        
        # Load vocabulary
        self.stoi, self.itos = self.load_vocab(self.vocab_path)
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
        self.load_model(self.model_path)
        self.model.eval()
    
    def load_vocab(self, vocab_path):
        """Load vocabulary from file."""
        # Try to find the file
        if not os.path.exists(vocab_path):
            # Look for file in current directory
            possible_paths = [
                vocab_path,
                f"./{vocab_path}",
                f"./app/{vocab_path}"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    vocab_path = path
                    break
        
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
        # Try to find the model file
        if not os.path.exists(model_path):
            possible_paths = [
                model_path,
                f"./{model_path}",
                f"./app/{model_path}"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file {model_path} not found. Available files: {os.listdir('.')}")
