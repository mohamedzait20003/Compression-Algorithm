"""
Arithmetic Coding Text Compressor

Achieves near-entropy compression (~9 bits/word) on natural language text
using word-level arithmetic coding.
"""

import os
import re
import json
import math
from collections import Counter


class ArithmeticCoder:
    """
    Arithmetic coder using 32-bit integer precision.
    Achieves near-optimal compression (close to entropy limit).
    """
    
    PRECISION_BITS = 32
    MAX_RANGE = (1 << PRECISION_BITS) - 1
    HALF = 1 << (PRECISION_BITS - 1)
    QUARTER = 1 << (PRECISION_BITS - 2)
    THREE_QUARTERS = HALF + QUARTER
    
    def __init__(self, frequencies: dict[int, int]):
        """
        Initialize with symbol frequencies.
        
        Args:
            frequencies: Dict mapping symbol_id -> frequency count
        """
        self.frequencies = frequencies
        total = sum(frequencies.values())
        
        # Build cumulative frequency table
        self.cum_freq = {}
        self.total_freq = total
        
        cumulative = 0
        for symbol in sorted(frequencies.keys()):
            self.cum_freq[symbol] = (cumulative, cumulative + frequencies[symbol])
            cumulative += frequencies[symbol]
    
    def encode(self, symbols: list[int]) -> bytes:
        """Encode a list of symbols to bytes."""
        low = 0
        high = self.MAX_RANGE
        pending_bits = 0
        bits = []
        
        for symbol in symbols:
            if symbol not in self.cum_freq:
                raise ValueError(f"Unknown symbol: {symbol}")
            
            range_size = high - low + 1
            sym_low, sym_high = self.cum_freq[symbol]
            
            high = low + (range_size * sym_high) // self.total_freq - 1
            low = low + (range_size * sym_low) // self.total_freq
            
            while True:
                if high < self.HALF:
                    bits.append(0)
                    bits.extend([1] * pending_bits)
                    pending_bits = 0
                elif low >= self.HALF:
                    bits.append(1)
                    bits.extend([0] * pending_bits)
                    pending_bits = 0
                    low -= self.HALF
                    high -= self.HALF
                elif low >= self.QUARTER and high < self.THREE_QUARTERS:
                    pending_bits += 1
                    low -= self.QUARTER
                    high -= self.QUARTER
                else:
                    break
                
                low = low << 1
                high = (high << 1) | 1
        
        # Flush remaining bits
        pending_bits += 1
        if low < self.QUARTER:
            bits.append(0)
            bits.extend([1] * pending_bits)
        else:
            bits.append(1)
            bits.extend([0] * pending_bits)
        
        # Pad to byte boundary
        while len(bits) % 8 != 0:
            bits.append(0)
        
        # Convert bits to bytes
        result = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            result.append(byte)
        
        return bytes(result)
    
    def decode(self, data: bytes, num_symbols: int) -> list[int]:
        """Decode bytes back to symbols."""
        # Convert bytes to bits
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        
        low = 0
        high = self.MAX_RANGE
        value = 0
        
        # Initialize value from first bits
        for i in range(self.PRECISION_BITS):
            value = (value << 1) | (bits[i] if i < len(bits) else 0)
        
        bit_index = self.PRECISION_BITS
        symbols = []
        
        for _ in range(num_symbols):
            range_size = high - low + 1
            scaled_value = ((value - low + 1) * self.total_freq - 1) // range_size
            
            # Find symbol
            found_symbol = None
            for symbol, (sym_low, sym_high) in self.cum_freq.items():
                if sym_low <= scaled_value < sym_high:
                    found_symbol = symbol
                    break
            
            if found_symbol is None:
                break
            
            symbols.append(found_symbol)
            sym_low, sym_high = self.cum_freq[found_symbol]
            
            high = low + (range_size * sym_high) // self.total_freq - 1
            low = low + (range_size * sym_low) // self.total_freq
            
            while True:
                if high < self.HALF:
                    pass
                elif low >= self.HALF:
                    low -= self.HALF
                    high -= self.HALF
                    value -= self.HALF
                elif low >= self.QUARTER and high < self.THREE_QUARTERS:
                    low -= self.QUARTER
                    high -= self.QUARTER
                    value -= self.QUARTER
                else:
                    break
                
                low = low << 1
                high = (high << 1) | 1
                next_bit = bits[bit_index] if bit_index < len(bits) else 0
                value = (value << 1) | next_bit
                bit_index += 1
        
        return symbols


class TextCompressor:
    """
    Word-level text compressor using arithmetic coding.
    
    Trains on a corpus to build a word frequency vocabulary,
    then uses arithmetic coding for near-entropy compression.
    """
    
    ESCAPE_SYMBOL = 0  # Reserved for unknown words
    
    def __init__(self, vocab_path: str = "models/vocab.json", max_vocab_size: int = 16384):
        """
        Initialize the compressor.
        
        Args:
            vocab_path: Path to save/load vocabulary
            max_vocab_size: Maximum vocabulary size
        """
        self.vocab_path = vocab_path
        self.max_vocab_size = max_vocab_size
        
        # Word <-> ID mappings
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}
        
        # Frequencies for arithmetic coding
        self.word_frequencies: dict[int, int] = {}
        
        # Arithmetic coder (built after training)
        self.coder = None
        
        # Statistics
        self.entropy = 0.0
        
        if os.path.exists(self.vocab_path):
            self.load_vocab()
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words and punctuation."""
        return re.findall(r"[\w']+|[.,!?;:\-\"'()]+", text.lower())
    
    def train(self, texts: list[str], min_frequency: int = 2):
        """
        Build vocabulary from training texts.
        
        Args:
            texts: List of training sentences
            min_frequency: Minimum word frequency to include in vocabulary
        """
        print("Building vocabulary...")
        
        # Count words
        word_counts = Counter()
        for text in texts:
            word_counts.update(self._tokenize(text))
        
        # Get top words meeting frequency threshold
        common_words = [
            (word, count) for word, count in word_counts.most_common()
            if count >= min_frequency
        ][:self.max_vocab_size - 1]
        
        # Estimate frequency for unknown words
        unknown_freq = sum(c for w, c in word_counts.items() if w not in dict(common_words))
        unknown_freq = max(1, unknown_freq // 10)
        
        # Build word <-> ID mappings (0 reserved for escape)
        self.word_to_id = {word: i + 1 for i, (word, _) in enumerate(common_words)}
        self.id_to_word = {i + 1: word for i, (word, _) in enumerate(common_words)}
        self.id_to_word[self.ESCAPE_SYMBOL] = '<UNK>'
        
        # Build frequency table
        self.word_frequencies = {self.ESCAPE_SYMBOL: unknown_freq}
        for word, count in common_words:
            self.word_frequencies[self.word_to_id[word]] = count
        
        # Create arithmetic coder
        self.coder = ArithmeticCoder(self.word_frequencies)
        
        # Calculate entropy
        total = sum(self.word_frequencies.values())
        self.entropy = -sum(
            (f / total) * math.log2(f / total)
            for f in self.word_frequencies.values() if f > 0
        )
        
        # Coverage stats
        covered = sum(c for w, c in common_words)
        total_words = sum(word_counts.values())
        coverage = covered / total_words * 100 if total_words > 0 else 0
        
        print(f"  Vocabulary size: {len(self.word_to_id)} words")
        print(f"  Entropy: {self.entropy:.2f} bits/word")
        print(f"  Coverage: {coverage:.1f}%")
        
        self.save_vocab()
    
    def save_vocab(self):
        """Save vocabulary to disk."""
        data = {
            'word_to_id': self.word_to_id,
            'word_frequencies': self.word_frequencies,
            'entropy': self.entropy
        }
        os.makedirs(os.path.dirname(self.vocab_path) or '.', exist_ok=True)
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Saved to: {self.vocab_path}")
    
    def load_vocab(self):
        """Load vocabulary from disk."""
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.word_to_id = data['word_to_id']
        self.id_to_word = {int(v): k for k, v in self.word_to_id.items()}
        self.id_to_word[self.ESCAPE_SYMBOL] = '<UNK>'
        self.word_frequencies = {int(k): v for k, v in data['word_frequencies'].items()}
        self.entropy = data.get('entropy', 0)
        
        self.coder = ArithmeticCoder(self.word_frequencies)
        
        print(f"Vocabulary loaded: {len(self.word_to_id)} words, entropy: {self.entropy:.2f} bits/word")
    
    def compress(self, text: str) -> bytes:
        """
        Compress text to bytes.
        
        Args:
            text: Input text string
            
        Returns:
            Compressed bytes
        """
        if self.coder is None:
            raise RuntimeError("No vocabulary. Call train() first.")
        
        words = self._tokenize(text)
        
        # Convert words to symbol IDs
        symbols = []
        unknown_words = []
        
        for word in words:
            if word in self.word_to_id:
                symbols.append(self.word_to_id[word])
            else:
                symbols.append(self.ESCAPE_SYMBOL)
                unknown_words.append(word)
        
        # Encode with arithmetic coder
        encoded = self.coder.encode(symbols)
        
        # Encode unknown words as length-prefixed UTF-8
        unknown_data = bytearray()
        for word in unknown_words:
            word_bytes = word.encode('utf-8')
            unknown_data.append(len(word_bytes))
            unknown_data.extend(word_bytes)
        
        # Header: [num_symbols (3 bytes), encoded_len (3 bytes)]
        header = bytes([
            (len(symbols) >> 16) & 0xFF,
            (len(symbols) >> 8) & 0xFF,
            len(symbols) & 0xFF,
            (len(encoded) >> 16) & 0xFF,
            (len(encoded) >> 8) & 0xFF,
            len(encoded) & 0xFF,
        ])
        
        return header + encoded + bytes(unknown_data)
    
    def decompress(self, compressed: bytes) -> str:
        """
        Decompress bytes back to text.
        
        Args:
            compressed: Compressed bytes
            
        Returns:
            Decompressed text string
        """
        if self.coder is None:
            raise RuntimeError("No vocabulary loaded.")
        
        if len(compressed) < 6:
            return ""
        
        # Parse header
        num_symbols = (compressed[0] << 16) | (compressed[1] << 8) | compressed[2]
        encoded_len = (compressed[3] << 16) | (compressed[4] << 8) | compressed[5]
        
        encoded_data = compressed[6:6 + encoded_len]
        unknown_data = compressed[6 + encoded_len:]
        
        # Decode symbols
        symbols = self.coder.decode(encoded_data, num_symbols)
        
        # Convert symbols back to words
        words = []
        unknown_offset = 0
        
        for symbol in symbols:
            if symbol == self.ESCAPE_SYMBOL:
                if unknown_offset < len(unknown_data):
                    length = unknown_data[unknown_offset]
                    word = unknown_data[unknown_offset + 1:unknown_offset + 1 + length].decode('utf-8')
                    unknown_offset += 1 + length
                    words.append(word)
                else:
                    words.append('<UNK>')
            else:
                words.append(self.id_to_word.get(symbol, '<UNK>'))
        
        # Join with smart spacing
        result = ""
        for i, word in enumerate(words):
            if i > 0 and word not in '.,!?;:"\')' and (not words[i-1] or words[i-1] not in '("\''):
                result += " "
            result += word
        
        return result
    
    def analyze(self, text: str) -> dict:
        """
        Analyze compression statistics for a text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with compression statistics
        """
        if self.coder is None:
            raise RuntimeError("No vocabulary.")
        
        original_bytes = len(text.encode('utf-8'))
        words = self._tokenize(text)
        
        # Count known vs unknown tokens
        known = sum(1 for w in words if w in self.word_to_id)
        unknown = len(words) - known
        
        # Calculate theoretical bits
        total_freq = sum(self.word_frequencies.values())
        theoretical_bits = 0
        for word in words:
            if word in self.word_to_id:
                freq = self.word_frequencies[self.word_to_id[word]]
            else:
                freq = self.word_frequencies[self.ESCAPE_SYMBOL]
                theoretical_bits += len(word.encode('utf-8')) * 8  # Unknown word overhead
            theoretical_bits += -math.log2(freq / total_freq)
        
        # Actual compression
        compressed = self.compress(text)
        compressed_bytes = len(compressed)
        word_count = len(text.split())
        
        return {
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'theoretical_bits': theoretical_bits,
            'token_count': len(words),
            'word_count': word_count,
            'known_tokens': known,
            'unknown_tokens': unknown,
            'coverage': known / len(words) * 100 if words else 0,
            'bits_per_word': theoretical_bits / word_count if word_count else 0,
            'compression_ratio': original_bytes / compressed_bytes if compressed_bytes else 0,
            'savings_percent': (1 - compressed_bytes / original_bytes) * 100 if original_bytes else 0,
        }
