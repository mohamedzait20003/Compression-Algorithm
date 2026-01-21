import os
import sys
import argparse

# Add llama-zip to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'llama-zip'))

from llama_zip import LlamaZip
from huggingface_hub import hf_hub_download

# Configuration
MODELS_DIR = "./models"
CACHE_DIR = "./models/.cache"
MODEL_REPO = "bartowski/Llama-3.2-1B-Instruct-GGUF"
MODEL_FILE = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"


def get_model_path() -> str:
    """Download model if needed and return path."""
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    
    if os.path.exists(model_path):
        return model_path
    
    print(f"Downloading {MODEL_FILE}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=MODELS_DIR,
        cache_dir=CACHE_DIR,
    )
    return model_path


def bytes_to_binary_string(data: bytes) -> str:
    """Convert bytes to binary string representation."""
    return ''.join(format(byte, '08b') for byte in data)


def compress_text(compressor: LlamaZip, text: str) -> tuple[bytes, dict]:
    original = text.encode('utf-8')
    compressed = compressor.compress(original)
    
    original_bits = len(original) * 8
    compressed_bits = len(compressed) * 8
    word_count = len(text.split())
    tokens = compressor.model.tokenize(original, add_bos=False)
    
    stats = {
        'original_bytes': len(original),
        'original_bits': original_bits,
        'compressed_bytes': len(compressed),
        'compressed_bits': compressed_bits,
        'word_count': word_count,
        'token_count': len(tokens),
        'bits_per_word': compressed_bits / word_count if word_count > 0 else 0,
        'bits_per_token': compressed_bits / len(tokens) if tokens else 0,
        'compression_ratio': original_bits / compressed_bits if compressed_bits > 0 else 0,
        'savings_percent': (1 - compressed_bits / original_bits) * 100 if original_bits > 0 else 0,
    }
    
    return compressed, stats


def print_result(text: str, compressed: bytes, stats: dict, verified: bool):
    binary = bytes_to_binary_string(compressed)
    
    print(f"\n{'='*70}")
    print(f"INPUT: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
    print(f"{'='*70}")
    print(f"\nCOMPRESSED BINARY ({stats['compressed_bits']} bits):")
    
    for i in range(0, len(binary), 64):
        print(f"  {binary[i:i+64]}")
    
    print(f"\nSTATISTICS:")
    print(f"  Original:          {stats['original_bytes']:>6} bytes ({stats['original_bits']:>6} bits)")
    print(f"  Compressed:        {stats['compressed_bytes']:>6} bytes ({stats['compressed_bits']:>6} bits)")
    print(f"  Compression ratio: {stats['compression_ratio']:>6.2f}x")
    print(f"  Space savings:     {stats['savings_percent']:>6.1f}%")
    print(f"  Bits per word:     {stats['bits_per_word']:>6.2f}")
    print(f"  Bits per token:    {stats['bits_per_token']:>6.2f}")
    print(f"  Verified:          {'✓ Yes' if verified else '✗ No'}")


def process_file(compressor: LlamaZip, filepath: str):
    """Process a text file where each line is a sentence."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        print("Error: File is empty")
        sys.exit(1)
    
    print(f"\nProcessing {len(lines)} sentences from: {filepath}")
    
    all_stats = []
    for i, line in enumerate(lines, 1):
        compressed, stats = compress_text(compressor, line)
        decompressed = compressor.decompress(compressed)
        verified = decompressed == line.encode('utf-8')
        all_stats.append((line, compressed, stats, verified))
        
        print_result(line, compressed, stats, verified)
    
    if len(lines) > 1:
        print_summary(all_stats)


def print_summary(all_stats: list):
    """Print summary statistics for multiple sentences."""
    total_original = sum(s['original_bits'] for _, _, s, _ in all_stats)
    total_compressed = sum(s['compressed_bits'] for _, _, s, _ in all_stats)
    total_words = sum(s['word_count'] for _, _, s, _ in all_stats)
    all_verified = all(v for _, _, _, v in all_stats)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Sentences:         {len(all_stats)}")
    print(f"  Total original:    {total_original // 8:,} bytes ({total_original:,} bits)")
    print(f"  Total compressed:  {total_compressed // 8:,} bytes ({total_compressed:,} bits)")
    print(f"  Overall ratio:     {total_original / total_compressed:.2f}x")
    print(f"  Avg bits/word:     {total_compressed / total_words:.2f}")
    print(f"  All verified:      {'✓ Yes' if all_verified else '✗ No'}")


def main():
    parser = argparse.ArgumentParser(
        description="Compress text using LLM-based arithmetic coding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                python main.py -t "Hello, how are you?"
                python main.py -f sentences.txt
                python main.py  (interactive mode)
        """
    )
    
    parser.add_argument('-t', '--text', type=str, help='Text to compress')
    parser.add_argument('-f', '--file', type=str, help='Path to .txt file (one sentence per line)')
    
    args = parser.parse_args()
    
    # Initialize compressor
    print("Loading model...")
    model_path = get_model_path()
    compressor = LlamaZip(model_path=model_path, verbose=False)
    print(f"Model loaded: {MODEL_FILE}")
    
    # Interactive mode if no arguments provided
    if not args.text and not args.file:
        print("\n" + "="*70)
        print("INTERACTIVE MODE")
        print("="*70)
        print("\nChoose input method:")
        print("  1. Enter text directly")
        print("  2. Load from file")
        
        while True:
            choice = input("\nEnter choice (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")
        
        if choice == '1':
            text = input("\nEnter text to compress: ").strip()
            if not text:
                print("Error: Empty text provided")
                sys.exit(1)
            compressed, stats = compress_text(compressor, text)
            decompressed = compressor.decompress(compressed)
            verified = decompressed == text.encode('utf-8')
            print_result(text, compressed, stats, verified)
        else:
            filepath = input("\nEnter file path: ").strip()
            process_file(compressor, filepath)
    elif args.text:
        # Direct text input
        compressed, stats = compress_text(compressor, args.text)
        decompressed = compressor.decompress(compressed)
        verified = decompressed == args.text.encode('utf-8')
        print_result(args.text, compressed, stats, verified)
    else:
        # File input
        process_file(compressor, args.file)


if __name__ == "__main__":
    main()
