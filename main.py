import os
from llama_zip import LlamaZip
from huggingface_hub import hf_hub_download

# Model configuration
MODELS_DIR = "./models"
CACHE_DIR = "./models/.cache"
MODEL_REPO = "bartowski/Llama-3.2-1B-Instruct-GGUF"
MODEL_FILE = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

def download_model():
    """Download the model to local directory if not already present."""
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    
    if os.path.exists(model_path):
        print(f"Model already exists: {model_path}")
        return model_path
    
    print(f"Downloading {MODEL_FILE} from {MODEL_REPO}...")
    print("This may take a few minutes depending on your connection speed.")
    
    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Download to local directory with local cache
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=MODELS_DIR,
        cache_dir=CACHE_DIR,
    )
    
    print(f"Model downloaded to: {downloaded_path}")
    return downloaded_path

# Download/locate the model
model_path = download_model()
print(f"Using model: {model_path}")
print("-" * 60)

# Initialize the compressor with Llama 3.2 1B
compressor = LlamaZip(model_path=model_path, verbose=False)

test_messages = [
    "Hello, how are you doing today?",
    "Thanks for your help!",
    "See you later",
    "Good morning everyone",
    "I will be there in five minutes",
    "What time is the meeting?",
    "Please send me the report",
    "Have a great day!",
    "The weather is nice today",
    "Can you help me with this problem?",
    'Ahmed,Mohamed helped me!',
    'Ahmed, Mohamed helped me!',
]


def analyze_compression(original_text: str, compressed_data: bytes, compressor: LlamaZip) -> dict:
    """Analyze compression statistics for a message."""
    original_bytes = original_text.encode('utf-8')
    
    # Calculate bits
    original_bits = len(original_bytes) * 8
    compressed_bits = len(compressed_data) * 8
    
    # Huffman estimation (using compressed size as approximation)
    huffman_bits = compressed_bits  # The arithmetic coding output
    
    # Word and token counts
    word_count = len(original_text.split())
    
    # Get token count from the model
    tokens = compressor.model.tokenize(original_bytes, add_bos=False)
    token_count = len(tokens)
    
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


def print_stats(message: str, stats: dict, verified: bool):
    """Print compression statistics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Message: \"{message[:50]}{'...' if len(message) > 50 else ''}\"")
    print(f"{'='*60}")
    print(f"  Original:          {stats['original_bytes']:>6} bytes  ({stats['original_bits']:>6} bits)")
    print(f"  Compressed:        {stats['compressed_bytes']:>6} bytes  ({stats['compressed_bits']:>6} bits)")
    print(f"  Huffman estimate:  {stats['huffman_bytes']:>6} bytes  ({stats['huffman_bits']:>6} bits)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Compression ratio: {stats['compression_ratio']:>6.2f}x")
    print(f"  Space savings:     {stats['savings_percent']:>6.1f}%")
    print(f"  Bits per word:     {stats['bits_per_word']:>6.2f}")
    print(f"  Bits per token:    {stats['bits_per_token']:>6.2f}")
    print(f"  Token count:       {stats['token_count']:>6}")
    print(f"  Overhead bits:     {stats['overhead_bits']:>6}")
    print(f"  Verified:          {'✓ Yes' if verified else '✗ No'}")


def main():
    print("\n" + "=" * 60)
    print("  LLAMA-ZIP COMPRESSION TEST WITH LLAMA 3.2 1B")
    print("=" * 60)
    
    all_stats = []
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n[{i}/{len(test_messages)}] Processing: \"{message[:40]}...\"" if len(message) > 40 else f"\n[{i}/{len(test_messages)}] Processing: \"{message}\"")
        
        # Compress
        original = message.encode('utf-8')
        compressed = compressor.compress(original)
        
        # Decompress and verify
        decompressed = compressor.decompress(compressed)
        verified = decompressed == original
        
        # Analyze
        stats = analyze_compression(message, compressed, compressor)
        all_stats.append((message, stats, verified))
        
        # Print individual stats
        print_stats(message, stats, verified)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("  SUMMARY STATISTICS")
    print("=" * 60)
    
    total_original = sum(s['original_bits'] for _, s, _ in all_stats)
    total_compressed = sum(s['compressed_bits'] for _, s, _ in all_stats)
    total_tokens = sum(s['token_count'] for _, s, _ in all_stats)
    all_verified = all(v for _, _, v in all_stats)
    
    avg_ratio = sum(s['compression_ratio'] for _, s, _ in all_stats) / len(all_stats)
    avg_savings = sum(s['savings_percent'] for _, s, _ in all_stats) / len(all_stats)
    avg_bits_per_word = sum(s['bits_per_word'] for _, s, _ in all_stats) / len(all_stats)
    avg_bits_per_token = sum(s['bits_per_token'] for _, s, _ in all_stats) / len(all_stats)
    
    print(f"\n  Messages tested:       {len(test_messages)}")
    print(f"  Total original:        {total_original // 8} bytes ({total_original} bits)")
    print(f"  Total compressed:      {total_compressed // 8} bytes ({total_compressed} bits)")
    print(f"  Total tokens:          {total_tokens}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Avg compression ratio: {avg_ratio:.2f}x")
    print(f"  Avg space savings:     {avg_savings:.1f}%")
    print(f"  Avg bits per word:     {avg_bits_per_word:.2f}")
    print(f"  Avg bits per token:    {avg_bits_per_token:.2f}")
    print(f"  Overall ratio:         {total_original / total_compressed:.2f}x")
    print(f"  All verified:          {'✓ Yes' if all_verified else '✗ No'}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()