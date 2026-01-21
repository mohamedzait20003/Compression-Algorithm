# Text Compression using Arithmetic Coding

A Python text compression system using **word-level arithmetic coding** for near-entropy compression.

## Overview

This project compresses natural language text by:
1. Building a word frequency vocabulary from training data
2. Using arithmetic coding to encode words with near-optimal bit lengths
3. Achieving ~2.5x compression ratio (~9 bits/word)

## Results

Tested on 1,000 sentences from the DailyDialog dataset:

| Metric | Value |
|--------|-------|
| Compression ratio | 2.45x |
| Space savings | 54.5% |
| Bits per word | 9.19 |
| Entropy limit | 8.64 bits/word |
| Vocabulary coverage | 98.9% |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python compress.py
```

### Programmatic Usage

```python
from utilities import TextCompressor, DataLoader

# Load training data
loader = DataLoader()
train_data = loader.load_train()

# Train compressor
compressor = TextCompressor(vocab_path="models/vocab.json")
compressor.train(train_data)

# Compress text
text = "Hello, how are you doing today?"
compressed = compressor.compress(text)
decompressed = compressor.decompress(compressed)

print(f"Original: {len(text)} bytes")
print(f"Compressed: {len(compressed)} bytes")
```

## Project Structure

```
├── compress.py              # Main demo script
├── requirements.txt         # Dependencies
├── utilities/
│   ├── __init__.py
│   ├── compressor.py        # TextCompressor & ArithmeticCoder
│   └── data_loader.py       # DailyDialog data loader
└── models/
    └── vocab.json           # Trained vocabulary
```

## How It Works

### Arithmetic Coding

Unlike Huffman coding which assigns integer bit codes, arithmetic coding represents an entire message as a single fractional number, achieving compression closer to the theoretical entropy limit.

### Word-Level Encoding

- Common words ("the", "you", "is") → fewer bits
- Rare words → more bits + fallback encoding
- Punctuation treated as separate tokens

### Compression Process

1. **Tokenize**: Split text into words and punctuation
2. **Map to IDs**: Convert words to vocabulary IDs
3. **Arithmetic encode**: Compress ID sequence to bits
4. **Handle unknowns**: Encode out-of-vocabulary words as UTF-8

## Limitations

The theoretical entropy of the DailyDialog corpus is ~8.64 bits/word. To achieve lower compression (e.g., 7 bits/word), you would need **context-based prediction** (PPM, neural language models) which exploits word-to-word dependencies.

## Requirements

- Python 3.10+
- datasets
- tqdm
