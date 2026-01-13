"""
Sequence-to-Sequence Model for Blender Code Generation

This model takes natural language input and generates Python code for Blender.
No templates, no fallback - pure neural network generation.

Architecture: Encoder-Decoder with Attention (Transformer-based)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architecture."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class BlenderCodeGenerator(nn.Module):
    """
    Transformer-based sequence-to-sequence model for generating Blender Python code.
    
    This model:
    1. Encodes the user's natural language request
    2. Decodes it into valid Python code for Blender
    """
    
    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.input_embedding = nn.Embedding(input_vocab_size, d_model)
        self.output_embedding = nn.Embedding(output_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            src: Source sequence [src_len, batch_size]
            tgt: Target sequence [tgt_len, batch_size]
            src_mask: Source mask
            tgt_mask: Target mask (causal)
            src_padding_mask: Padding mask for source
            tgt_padding_mask: Padding mask for target
            
        Returns:
            Output logits [tgt_len, batch_size, output_vocab_size]
        """
        # Embed and add positional encoding
        src_embedded = self.pos_encoder(self.input_embedding(src) * math.sqrt(self.d_model))
        tgt_embedded = self.pos_encoder(self.output_embedding(tgt) * math.sqrt(self.d_model))
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        
        # Transformer forward
        output = self.transformer(
            src_embedded,
            tgt_embedded,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence."""
        src_embedded = self.pos_encoder(self.input_embedding(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_embedded, src_mask)
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode target sequence given encoder memory."""
        tgt_embedded = self.pos_encoder(self.output_embedding(tgt) * math.sqrt(self.d_model))
        output = self.transformer.decoder(tgt_embedded, memory, tgt_mask)
        return self.output_projection(output)


class CodeGeneratorInference:
    """
    Inference wrapper for the BlenderCodeGenerator model.
    Handles tokenization, generation, and decoding.
    """
    
    def __init__(self, model_path: str, vocab_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.input_vocab = None
        self.output_vocab = None
        self.input_vocab_inv = None
        self.output_vocab_inv = None
        self.is_loaded = False
        
        self._load_model(model_path, vocab_path)
    
    def _load_model(self, model_path: str, vocab_path: str):
        """Load model and vocabulary."""
        import pickle
        import os
        
        try:
            # Load vocabulary
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
            
            self.input_vocab = vocab_data['input_vocab']
            self.output_vocab = vocab_data['output_vocab']
            self.input_vocab_inv = {v: k for k, v in self.input_vocab.items()}
            self.output_vocab_inv = {v: k for k, v in self.output_vocab.items()}
            
            # Special tokens
            self.pad_token = vocab_data.get('pad_token', '<PAD>')
            self.sos_token = vocab_data.get('sos_token', '<SOS>')
            self.eos_token = vocab_data.get('eos_token', '<EOS>')
            self.unk_token = vocab_data.get('unk_token', '<UNK>')
            
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.model = BlenderCodeGenerator(
                input_vocab_size=len(self.input_vocab),
                output_vocab_size=len(self.output_vocab),
                **checkpoint.get('model_config', {})
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            print(f"[CodeGenerator] Model loaded successfully on {self.device}")
            print(f"[CodeGenerator] Input vocab size: {len(self.input_vocab)}")
            print(f"[CodeGenerator] Output vocab size: {len(self.output_vocab)}")
            
        except FileNotFoundError as e:
            print(f"[CodeGenerator] Model or vocab file not found: {e}")
            self.is_loaded = False
        except Exception as e:
            print(f"[CodeGenerator] Error loading model: {e}")
            self.is_loaded = False
    
    def tokenize_input(self, text: str) -> torch.Tensor:
        """Tokenize input text to tensor."""
        text = text.lower().strip()
        tokens = text.split()
        
        # Convert to indices
        indices = []
        for token in tokens:
            if token in self.input_vocab:
                indices.append(self.input_vocab[token])
            else:
                indices.append(self.input_vocab.get(self.unk_token, 0))
        
        return torch.tensor(indices, dtype=torch.long).unsqueeze(1).to(self.device)
    
    def decode_output(self, indices: torch.Tensor) -> str:
        """Decode output indices to code string."""
        tokens = []
        for idx in indices:
            idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            if idx in self.output_vocab_inv:
                token = self.output_vocab_inv[idx]
                if token == self.eos_token:
                    break
                if token not in [self.pad_token, self.sos_token]:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    @torch.no_grad()
    def generate(
        self,
        text: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_k: int = 50
    ) -> dict:
        """
        Generate Blender Python code from natural language input.
        
        Args:
            text: Natural language description
            max_length: Maximum output length
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Top-k sampling parameter
            
        Returns:
            Dictionary with generated code and metadata
        """
        if not self.is_loaded:
            return {
                'success': False,
                'error': 'Model not loaded',
                'code': None
            }
        
        try:
            # Tokenize input
            src = self.tokenize_input(text)
            
            # Encode source
            memory = self.model.encode(src)
            
            # Start with SOS token
            sos_idx = self.output_vocab.get(self.sos_token, 0)
            eos_idx = self.output_vocab.get(self.eos_token, 1)
            
            generated = [sos_idx]
            
            # Autoregressive generation
            for _ in range(max_length):
                tgt = torch.tensor(generated, dtype=torch.long).unsqueeze(1).to(self.device)
                tgt_mask = self.model.generate_square_subsequent_mask(tgt.size(0)).to(self.device)
                
                output = self.model.decode(tgt, memory, tgt_mask)
                
                # Get last token logits
                logits = output[-1, 0, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(0, indices, values)
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                
                if next_token == eos_idx:
                    break
            
            # Decode to code
            code = self.decode_output(generated)
            
            # Post-process code formatting
            code = self._format_code(code)
            
            return {
                'success': True,
                'code': code,
                'input': text,
                'tokens_generated': len(generated)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'code': None
            }
    
    def _format_code(self, code: str) -> str:
        """Format generated code into proper Python syntax."""
        # Replace token placeholders with actual code structure
        # This handles the tokenized representation
        import re
        
        # Handle NEWLINE tokens - need to do this first
        code = code.replace('NEWLINE', '\n')
        
        # Handle INDENT tokens
        code = code.replace('INDENT', '    ')
        
        # Basic reformatting
        code = code.replace(' . ', '.')
        code = code.replace(' ( ', '(')
        code = code.replace(' ) ', ')')
        code = code.replace(' [ ', '[')
        code = code.replace(' ] ', ']')
        code = code.replace(' , ', ', ')
        code = code.replace(' = ', ' = ')
        code = code.replace(' : ', ': ')
        
        # Fix spacing issues
        code = code.replace('( ', '(')
        code = code.replace(' )', ')')
        code = code.replace('[ ', '[')
        code = code.replace(' ]', ']')
        
        # Clean up multiple spaces
        code = re.sub(r' +', ' ', code)
        
        # Fix indentation - process line by line
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove leading single space if not indented code
            stripped = line.lstrip()
            if line.startswith(' ') and not line.startswith('    '):
                line = stripped
            cleaned_lines.append(line.rstrip())
        code = '\n'.join(cleaned_lines)
        
        return code.strip()
