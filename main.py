import smaz
from tqdm import tqdm
from datasets import load_dataset

print("=" * 70)
print("SMAZ COMPRESSION")
print("=" * 70 + "\n")

def load_sentences(max_sentences=None):
    print(f"Loading DailyDialog dataset")
    ds = load_dataset(
        "parquet",
        data_files={
            "train": "https://huggingface.co/datasets/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full/train/0000.parquet",
            "validation": "https://huggingface.co/datasets/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full/validation/0000.parquet",
            "test": "https://huggingface.co/datasets/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full/test/0000.parquet"
        }
    )
    
    sentences = []
    for item in tqdm(ds['train'], desc="Extracting sentences"):
        utterances = item['utterances']
        for utterance in utterances:
            utterance = utterance.strip()
            if utterance:
                sentences.append(utterance)
                if max_sentences and len(sentences) >= max_sentences:
                    print(f"Extracted {len(sentences)} sentences")
                    return sentences
    
    print(f"Extracted {len(sentences)} sentences from train split")
    return sentences

# Test messages - varied chat/text scenarios
test_messages = load_sentences(max_sentences=1000)

# Track stats for average
all_stats = []

for i, message in enumerate(test_messages, 1):
    print(f"Test {i}: '{message}'")
    print(f"  Words: {len(message.split())}, Characters: {len(message)}")
    
    # Get original size in bytes
    original_bytes = len(message.encode('utf-8'))
    original_bits = original_bytes * 8
    
    # Compress with smaz
    compressed = smaz.compress(message)
    compressed_bytes = len(compressed)
    compressed_bits = compressed_bytes * 8
    
    stats = {
        'original_bytes': original_bytes,
        'original_bits': original_bits,
        'compressed_bytes': compressed_bytes,
        'compressed_bits': compressed_bits,
        'compression_ratio': original_bytes / compressed_bytes if compressed_bytes > 0 else 0,
        'savings_percent': (1 - compressed_bytes / original_bytes) * 100 if original_bytes > 0 else 0,
        'bits_per_word': compressed_bits / len(message.split()) if len(message.split()) > 0 else 0
    }
    all_stats.append(stats)
    
    # Decompress and verify
    decompressed = smaz.decompress(compressed)
    match = decompressed == message
    
    print(f"  Original: {original_bytes} bytes ({original_bits} bits) → Compressed: {compressed_bytes} bytes ({compressed_bits} bits)")
    print(f"  Savings: {stats['savings_percent']:.1f}% | Bits/word: {stats['bits_per_word']:.2f}")
    print(f"  Restored: {'✓' if match else '✗'}")
    print()

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70 + "\n")

avg_savings = sum(s['savings_percent'] for s in all_stats) / len(all_stats)
avg_ratio = sum(s['compression_ratio'] for s in all_stats) / len(all_stats)
avg_bits_per_word = sum(s['bits_per_word'] for s in all_stats) / len(all_stats)
total_original_bytes = sum(s['original_bytes'] for s in all_stats)
total_compressed_bytes = sum(s['compressed_bytes'] for s in all_stats)
total_original_bits = sum(s['original_bits'] for s in all_stats)
total_compressed_bits = sum(s['compressed_bits'] for s in all_stats)

print(f"Total samples: {len(all_stats)}")
print(f"Average compression ratio: {avg_ratio:.2f}x")
print(f"Average savings: {avg_savings:.1f}%")
print(f"Average bits per word: {avg_bits_per_word:.2f}")
print(f"Total original: {total_original_bytes:,} bytes ({total_original_bits:,} bits)")
print(f"Total compressed: {total_compressed_bytes:,} bytes ({total_compressed_bits:,} bits)")
print(f"Overall savings: {(1 - total_compressed_bytes/total_original_bytes) * 100:.1f}%")

print("\n" + "=" * 70)
print("NOTE: SMAZ is optimized for short English text and small strings.")
print("The compression is dictionary-based and works best for common English words.")
print("=" * 70)