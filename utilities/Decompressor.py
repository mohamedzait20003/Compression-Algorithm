from typing import List
from lib import HuffmanProtocol


class Decompressor:
    def __init__(self, huffman: HuffmanProtocol):
        self.huffman = huffman
    
    def _detokenize(self, tokens: List[str]) -> str:
        return "".join(tokens)
    
    def decompress(self, binary_string: str) -> str:
        remaining = self.huffman.deserialize_codebook(binary_string)

        padding = int(remaining[:3], 2)
        encoded_text = remaining[3:]
        
        if padding > 0:
            encoded_text = encoded_text[:-padding]
        
        tokens = self.huffman.decode(encoded_text)
        return self._detokenize(tokens)
    
    def decompress_with_codebook(self, binary_string: str, codebook: dict) -> str:
        self.huffman.load_codebook(codebook)
        tokens = self.huffman.decode(binary_string)
        return self._detokenize(tokens)
    
    def get_codebook(self) -> dict:
        return self.huffman.get_codebook()