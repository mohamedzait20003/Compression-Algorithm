import json
import heapq
from collections import Counter


class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


class Huffman:
    def __init__(self):
        self.codebook = {}
        self.reverse_codebook = {}
    
    def build_frequency_table(self, tokens):
        return Counter(tokens)
    
    def build_tree(self, freq_table):
        heap = [HuffmanNode(symbol=symbol, freq=freq) for symbol, freq in freq_table.items()]
        heapq.heapify(heap)
        
        if len(heap) == 1:
            node = heapq.heappop(heap)
            root = HuffmanNode(freq=node.freq, left=node)
            return root
        
        if len(heap) == 0:
            return None
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)
        
        return heap[0]
    
    def build_codebook(self, node, code="", codebook=None):
        if codebook is None:
            codebook = {}
        
        if node is None:
            return codebook
        
        if node.symbol is not None:
            codebook[node.symbol] = code if code else "0"
        
        self.build_codebook(node.left, code + "0", codebook)
        self.build_codebook(node.right, code + "1", codebook)
        
        return codebook
    
    def train(self, tokens):
        freq_table = self.build_frequency_table(tokens)
        tree = self.build_tree(freq_table)
        self.codebook = self.build_codebook(tree)
        self.reverse_codebook = {code: symbol for symbol, code in self.codebook.items()}
        return self.codebook
    
    def encode(self, tokens, auto_retrain=True):
        if not self.codebook:
            raise ValueError("Huffman codebook not trained. Call train() first.")
        
        unknown_tokens = [token for token in tokens if token not in self.codebook]
        
        if unknown_tokens:
            if auto_retrain:
                existing_tokens = list(self.codebook.keys())
                combined_tokens = existing_tokens + list(tokens)
                self.train(combined_tokens)
            else:
                raise ValueError(f"Token '{unknown_tokens[0]}' not found in codebook.")
        
        encoded_bits = []
        for token in tokens:
            encoded_bits.append(self.codebook[token])
        
        return "".join(encoded_bits)
    
    def decode(self, binary_string):
        if not self.reverse_codebook:
            raise ValueError("Huffman codebook not loaded. Call train() or load_codebook() first.")
        
        decoded_tokens = []
        current_code = ""
        
        for bit in binary_string:
            current_code += bit
            if current_code in self.reverse_codebook:
                decoded_tokens.append(self.reverse_codebook[current_code])
                current_code = ""
        
        if current_code:
            raise ValueError(f"Invalid binary string: leftover bits '{current_code}'")
        
        return decoded_tokens
    
    def get_codebook(self):
        return self.codebook.copy()
    
    def load_codebook(self, codebook):
        self.codebook = codebook.copy()
        self.reverse_codebook = {code: symbol for symbol, code in self.codebook.items()}
    
    def serialize_codebook(self):
        json_str = json.dumps(self.codebook)
        binary = "".join(f"{ord(c):08b}" for c in json_str)
        length_binary = f"{len(binary):032b}"
        return length_binary + binary
    
    def deserialize_codebook(self, binary_string):
        length = int(binary_string[:32], 2)
        codebook_binary = binary_string[32:32 + length]
        
        json_chars = []
        for i in range(0, len(codebook_binary), 8):
            byte = codebook_binary[i:i + 8]
            json_chars.append(chr(int(byte, 2)))
        json_str = "".join(json_chars)

        self.codebook = json.loads(json_str)
        self.reverse_codebook = {code: symbol for symbol, code in self.codebook.items()}
        
        return binary_string[32 + length:]
