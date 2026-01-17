import re
from typing import List
from lib import HuffmanProtocol


class Compressor:
    def __init__(self, huffman: HuffmanProtocol, max_words: int = 20):
        self.huffman = huffman
        self.max_words = max_words
    
    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'\w+|[^\w\s]|\s+', text)
        return tokens
    
    def _detokenize(self, tokens: List[str]) -> str:
        return "".join(tokens)
    
    def _preprocess(self, text: str) -> str:
        word_count = len(text.split())
        if word_count > self.max_words:
            raise ValueError(f"Input text exceeds maximum word limit of {self.max_words}.")
        return text
    
    def compress_to_binary(self, text: str) -> str:
        self._preprocess(text)
        
        tokens = self._tokenize(text)
        self.huffman.train(tokens)
        
        encoded_text = self.huffman.encode(tokens)
        codebook_binary = self.huffman.serialize_codebook()
        
        total_bits = len(codebook_binary) + 3 + len(encoded_text)
        padding = (8 - (total_bits % 8)) % 8
        padding_binary = f"{padding:03b}"
        
        full_binary = codebook_binary + padding_binary + encoded_text + ("0" * padding)
        
        return full_binary
    
    def compress_to_binary_without_codebook(self, text: str) -> str:
        self._preprocess(text)
        
        tokens = self._tokenize(text)
        self.huffman.train(tokens)
        encoded_text = self.huffman.encode(tokens)
        
        return encoded_text
    
    def get_codebook(self) -> dict:
        return self.huffman.get_codebook()
    
    def get_stats(self, text: str) -> dict:
        original_bits = len(text.encode('utf-8')) * 8
        
        huffman_only = self.compress_to_binary_without_codebook(text)
        huffman_bits = len(huffman_only)
        
        full_binary = self.compress_to_binary(text)
        compressed_bits = len(full_binary)
        
        word_count = len(text.split())
        token_count = len(self._tokenize(text))
        
        return {
            'original_bits': original_bits,
            'huffman_bits': huffman_bits,
            'compressed_bits': compressed_bits,
            'original_bytes': original_bits // 8,
            'huffman_bytes': (huffman_bits + 7) // 8,
            'compressed_bytes': compressed_bits // 8,
            'bits_per_word': huffman_bits / word_count if word_count > 0 else 0,
            'bits_per_token': huffman_bits / token_count if token_count > 0 else 0,
            'compression_ratio': original_bits / huffman_bits if huffman_bits > 0 else 0,
            'savings_percent': (1 - huffman_bits / original_bits) * 100,
            'overhead_bits': compressed_bits - huffman_bits,
            'token_count': token_count
        }
