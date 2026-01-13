"""
Training Script for Blender Code Generation Model

Trains a Transformer-based seq2seq model to generate Python code from natural language.
NO FALLBACK - the model learns to generate code directly.
"""

import os
import sys
import json
import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.neural_network.code_generator_model import BlenderCodeGenerator


class CodeGenerationDataset(Dataset):
    """Dataset for code generation training."""
    
    def __init__(
        self,
        data: List[Dict],
        input_vocab: Dict[str, int],
        output_vocab: Dict[str, int],
        max_input_len: int = 64,
        max_output_len: int = 256
    ):
        self.data = data
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        
        # Special tokens
        self.pad_idx = input_vocab.get('<PAD>', 0)
        self.sos_idx = output_vocab.get('<SOS>', 1)
        self.eos_idx = output_vocab.get('<EOS>', 2)
        self.unk_idx = input_vocab.get('<UNK>', 3)
    
    def __len__(self):
        return len(self.data)
    
    def tokenize_input(self, text: str) -> List[int]:
        """Tokenize input text."""
        text = text.lower().strip()
        tokens = text.split()
        indices = []
        for token in tokens:
            if token in self.input_vocab:
                indices.append(self.input_vocab[token])
            else:
                indices.append(self.unk_idx)
        return indices[:self.max_input_len]
    
    def tokenize_output(self, code: str) -> List[int]:
        """Tokenize output code."""
        # Preprocess code for tokenization
        code = self._preprocess_code(code)
        tokens = code.split()
        
        indices = [self.sos_idx]
        for token in tokens:
            if token in self.output_vocab:
                indices.append(self.output_vocab[token])
            else:
                indices.append(self.output_vocab.get('<UNK>', self.unk_idx))
        indices.append(self.eos_idx)
        
        return indices[:self.max_output_len]
    
    def _preprocess_code(self, code: str) -> str:
        """Preprocess code for tokenization."""
        # Add spaces around operators and punctuation
        for char in ['(', ')', '[', ']', '{', '}', ',', '=', '.', ':', '+', '-', '*', '/']:
            code = code.replace(char, f' {char} ')
        
        # Handle newlines
        code = code.replace('\n', ' NEWLINE ')
        
        # Handle indentation (4 spaces = 1 INDENT)
        lines = code.split(' NEWLINE ')
        processed_lines = []
        for line in lines:
            indent_count = 0
            stripped = line.lstrip()
            spaces = len(line) - len(stripped)
            indent_count = spaces // 4
            processed_line = ' INDENT ' * indent_count + stripped
            processed_lines.append(processed_line)
        
        code = ' NEWLINE '.join(processed_lines)
        
        # Normalize whitespace
        code = ' '.join(code.split())
        
        return code
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        
        src_indices = self.tokenize_input(sample['prompt'])
        tgt_indices = self.tokenize_output(sample['code'])
        
        return (
            torch.tensor(src_indices, dtype=torch.long),
            torch.tensor(tgt_indices, dtype=torch.long)
        )


def build_vocab(data: List[Dict], min_freq: int = 1) -> Tuple[Dict, Dict]:
    """Build vocabularies for input and output."""
    
    # Special tokens
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    
    # Count input tokens
    input_counter = Counter()
    for sample in data:
        text = sample['prompt'].lower().strip()
        input_counter.update(text.split())
    
    # Count output tokens
    output_counter = Counter()
    for sample in data:
        code = sample['code']
        # Preprocess
        for char in ['(', ')', '[', ']', '{', '}', ',', '=', '.', ':', '+', '-', '*', '/']:
            code = code.replace(char, f' {char} ')
        code = code.replace('\n', ' NEWLINE ')
        tokens = code.split()
        output_counter.update(tokens)
        output_counter.update(['INDENT'])  # Add INDENT token
    
    # Build input vocab
    input_vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for token, count in input_counter.items():
        if count >= min_freq and token not in input_vocab:
            input_vocab[token] = len(input_vocab)
    
    # Build output vocab
    output_vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for token, count in output_counter.items():
        if count >= min_freq and token not in output_vocab:
            output_vocab[token] = len(output_vocab)
    
    print(f"Input vocabulary size: {len(input_vocab)}")
    print(f"Output vocabulary size: {len(output_vocab)}")
    
    return input_vocab, output_vocab


def collate_fn(batch):
    """Collate function for DataLoader."""
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_padded = pad_sequence(src_batch, padding_value=0)  # [src_len, batch]
    tgt_padded = pad_sequence(tgt_batch, padding_value=0)  # [tgt_len, batch]
    
    return src_padded, tgt_padded


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    clip: float = 1.0
) -> float:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Teacher forcing: input is all but last, target is all but first
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]
        
        optimizer.zero_grad()
        
        # Forward
        output = model(src, tgt_input)
        
        # Reshape for loss calculation
        output = output.view(-1, output.size(-1))
        tgt_output = tgt_output.view(-1)
        
        loss = criterion(output, tgt_output)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """Evaluate model."""
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]
            
            output = model(src, tgt_input)
            
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.view(-1)
            
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    """Main training function."""
    
    print("=" * 60)
    print("BLENDER CODE GENERATION MODEL TRAINING")
    print("=" * 60)
    
    # Configuration
    config = {
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'max_seq_len': 512
    }
    
    training_config = {
        'batch_size': 32,
        'learning_rate': 0.0001,
        'epochs': 100,
        'patience': 15,
        'min_delta': 0.001
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    data_path = PROJECT_ROOT / "data_red" / "generated" / "code_generation_data.json"
    
    if not data_path.exists():
        print("Generating training data...")
        from src.data_acquisition.generate_code_data import BlenderCodeDataGenerator
        generator = BlenderCodeDataGenerator()
        data = generator.generate_dataset(samples_per_operation=100)
        
        os.makedirs(data_path.parent, exist_ok=True)
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Split data
    random.shuffle(data)
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Build vocabulary
    input_vocab, output_vocab = build_vocab(train_data)
    
    # Save vocabulary
    vocab_path = PROJECT_ROOT / "config" / "code_vocab.pkl"
    vocab_data = {
        'input_vocab': input_vocab,
        'output_vocab': output_vocab,
        'pad_token': '<PAD>',
        'sos_token': '<SOS>',
        'eos_token': '<EOS>',
        'unk_token': '<UNK>'
    }
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"Vocabulary saved to {vocab_path}")
    
    # Create datasets
    train_dataset = CodeGenerationDataset(train_data, input_vocab, output_vocab)
    val_dataset = CodeGenerationDataset(val_data, input_vocab, output_vocab)
    test_dataset = CodeGenerationDataset(test_data, input_vocab, output_vocab)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    model = BlenderCodeGenerator(
        input_vocab_size=len(input_vocab),
        output_vocab_size=len(output_vocab),
        **config
    )
    model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.AdamW(model.parameters(), lr=training_config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    for epoch in range(training_config['epochs']):
        print(f"\nEpoch {epoch + 1}/{training_config['epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Early stopping
        if val_loss < best_val_loss - training_config['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            model_path = PROJECT_ROOT / "models" / "code_generator.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': config,
                'epoch': epoch,
                'val_loss': val_loss,
                'input_vocab_size': len(input_vocab),
                'output_vocab_size': len(output_vocab)
            }, model_path)
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= training_config['patience']:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(PROJECT_ROOT / "models" / "code_generator.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save training history
    history_path = PROJECT_ROOT / "results" / "code_gen_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Test generation
    print("\n" + "=" * 60)
    print("GENERATION EXAMPLES")
    print("=" * 60)
    
    from src.neural_network.code_generator_model import CodeGeneratorInference
    
    inference = CodeGeneratorInference(
        model_path=str(PROJECT_ROOT / "models" / "code_generator.pt"),
        vocab_path=str(PROJECT_ROOT / "config" / "code_vocab.pkl"),
        device=device
    )
    
    test_prompts = [
        "create a cube",
        "cub",
        "sfera",
        "apply red material",
        "add point light"
    ]
    
    for prompt in test_prompts:
        result = inference.generate(prompt)
        print(f"\nPrompt: {prompt}")
        if result['success']:
            print(f"Generated Code:\n{result['code']}")
        else:
            print(f"Error: {result['error']}")
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
