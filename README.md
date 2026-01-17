# Text Compression using Huffman Coding

A Python-based text compression system implemented from scratch using **Word-Level Huffman Coding** algorithm, pre-trained on 10,000 most common English words.

## Overview

This project implements a lossless compression/decompression system using Huffman coding at the **word level**. Unlike character-level Huffman, this approach:
- Pre-trains on word frequencies from `wordfreq` package (billions of words corpus)
- Assigns shorter codes to common words ("the", "is", "you")
- Achieves **~60% compression** on typical chat messages
- Eliminates per-message codebook overhead (shared pre-trained codebook)

## How It Works

1. **Pre-train with WordFreq**: Load top 10,000 English words by frequency
2. **Build Huffman Tree**: Create a binary tree where:
   - Common words ("the", "is", "you") get shorter codes (3-8 bits)
   - Rare words get longer codes (12+ bits)
   - Punctuation and spaces included
3. **Tokenize Text**: Split into words, punctuation, and whitespace
4. **Encode**: Replace each token with its Huffman code
5. **Decode**: Use the pre-trained codebook to reconstruct text

## Example

For the text "Hello, how are you?":
```
Pre-trained Huffman Codes:
  ' ': 000          (3 bits - most common)
  'you': 01101      (5 bits - common word)
  'how': 011011     (6 bits)
  'are': 0110110    (7 bits)
  ',': 0110101      (7 bits)
  '?': 011010010    (9 bits)
  'hello': ...      (12 bits - less common)

Original (ASCII): 19 chars × 8 bits = 152 bits
Huffman encoded:  ~60 bits
Compression ratio: ~2.5x
Savings: ~60%
```

## Features

- **Word-Level Encoding**: Treats words as atomic units, not characters
- **Pre-trained Codebook**: Uses `wordfreq` (10,000 most common English words)
- **~60% Compression**: Average savings on typical chat messages
- **Dependency Injection**: Clean architecture with `HuffmanProtocol`
- **Lossless**: Perfect reconstruction of original text

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the demo:

```bash
python main.py
```

## Project Structure

```
RA Test/
├── main.py                    # Demo with 10 test examples and statistics
├── requirements.txt           # Dependencies (wordfreq)
├── README.md                  # This file
├── lib/
│   ├── __init__.py            # Exports Huffman, HuffmanProtocol
│   └── Huffman.py             # Core Huffman coding algorithm
├── utilities/
│   ├── __init__.py            # Exports Compressor, Decompressor
│   ├── Compressor.py          # Word-level compression with DI
│   └── Decompressor.py        # Word-level decompression with DI
```

## Usage

### Basic Example with Pre-training

```python
from lib.Huffman import Huffman
from utilities import Compressor, Decompressor
from wordfreq import top_n_list
import re

# Load training data (10,000 most common English words)
common_words = top_n_list('en', 10000)

# Build training corpus with frequency weights
training_tokens = []
for i, word in enumerate(common_words):
    weight = max(1, (10000 - i) // 100)
    training_tokens.extend([word] * weight)

# Add punctuation
training_tokens.extend([' '] * 50000)
training_tokens.extend(['.'] * 5000)
training_tokens.extend([','] * 4000)

# Create and train Huffman model
huffman = Huffman()
huffman.train(training_tokens)

# Initialize compressor/decompressor with DI
compressor = Compressor(huffman=huffman, max_words=50)
decompressor = Decompressor(huffman=huffman)

# Encode
text = "Hello, how are you?"
tokens = re.findall(r'\w+|[^\w\s]|\s+', text.lower())
encoded = huffman.encode(tokens)
print(f"Encoded: {len(encoded)} bits")

# Decode
decoded_tokens = huffman.decode(encoded)
restored = ''.join(decoded_tokens)
print(f"Restored: {restored}")
```

## API Reference

### Compressor Class

| Method | Description |
|--------|-------------|
| `__init__(max_words=20)` | Initialize compressor with max word limit |
| `compress_to_binary(text)` | Compress text with codebook, return binary string |
| `compress_to_binary_without_codebook(text)` | Compress without codebook (for analysis) |
| `get_codebook()` | Return the current Huffman codebook |
| `get_stats(text)` | Return compression statistics dictionary |

### Decompressor Class

| Method | Description |
|--------|-------------|
| `decompress(binary_string)` | Decompress binary string (with embedded codebook) |
| `decompress_with_codebook(binary, codebook)` | Decompress using provided codebook |
| `get_codebook()` | Return the extracted codebook |

## Compression Statistics

| Metric | Description |
|--------|-------------|
| `original_bits` | Size in bits using ASCII encoding (8 bits/char) |
| `huffman_bits` | Size using Huffman codes (without codebook) |
| `compressed_bits` | Total size including codebook overhead |
| `compression_ratio` | original_bits / huffman_bits |
| `savings_percent` | Percentage reduction in size |
| `overhead_bits` | Codebook size in bits |

## Performance Results (10 Test Samples)

| Metric | Value |
|--------|-------|
| Average Compression Ratio | **2.57x** |
| Average Savings | **60.2%** |
| Average Bits per Word | **15.72** |
| Total Original | 1,944 bits |
| Total Compressed | 774 bits |

### Sample Results

| Message | Original | Compressed | Savings |
|---------|----------|------------|--------|
| "Hello, how are you doing today?" | 248 bits | 106 bits | 57.3% |
| "Good morning everyone" | 168 bits | 44 bits | 73.8% |
| "Please send me the report" | 200 bits | 75 bits | 62.5% |
| "Thanks for your help!" | 168 bits | 67 bits | 60.1% |

## Advantages Over Per-Message Encoding

1. **No Codebook Overhead**: Pre-trained codebook shared between sender/receiver
2. **Consistent Performance**: Common words always get short codes
3. **Fast Encoding**: No tree building needed per message
4. **Scales Well**: Works great for chat applications with many short messages

## Author

Mohamed Zaitoun

## License

MIT License
