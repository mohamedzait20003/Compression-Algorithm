"""
Text Compression using Arithmetic Coding

Demonstrates near-entropy compression on the DailyDialog dataset.
"""

from tqdm import tqdm
from utilities import TextCompressor, DataLoader


def main():
    print("\n" + "=" * 60)
    print("  TEXT COMPRESSION - ARITHMETIC CODING")
    print("=" * 60)

    loader = DataLoader()
    
    # =========================================================================
    # TRAIN: Build vocabulary from training data
    # =========================================================================
    print("\n[1] TRAINING")
    print("-" * 60)
    
    train_sentences = loader.load_train()
    
    compressor = TextCompressor(
        vocab_path="models/vocab.json",
        max_vocab_size=16384
    )
    
    if compressor.coder is None:
        compressor.train(train_sentences, min_frequency=2)
    
    # =========================================================================
    # TEST: Evaluate on test set
    # =========================================================================
    print("\n[2] TESTING")
    print("-" * 60)
    
    test_sentences = loader.load_test(max_sentences=1000)
    
    stats_list = []
    failures = 0
    
    for sentence in tqdm(test_sentences, desc="Compressing"):
        try:
            compressed = compressor.compress(sentence)
            decompressed = compressor.decompress(compressed)
            
            # Verify (token-level)
            if compressor._tokenize(sentence) != compressor._tokenize(decompressed):
                failures += 1
            
            stats_list.append(compressor.analyze(sentence))
        except Exception:
            failures += 1
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n[3] RESULTS")
    print("-" * 60)
    
    total_original = sum(s['original_bytes'] for s in stats_list)
    total_compressed = sum(s['compressed_bytes'] for s in stats_list)
    total_words = sum(s['word_count'] for s in stats_list)
    total_theoretical = sum(s['theoretical_bits'] for s in stats_list)
    total_known = sum(s['known_tokens'] for s in stats_list)
    total_tokens = sum(s['token_count'] for s in stats_list)
    
    bits_per_word = total_theoretical / total_words
    coverage = total_known / total_tokens * 100
    ratio = total_original / total_compressed
    savings = (1 - total_compressed / total_original) * 100
    
    print(f"""
  Sentences:         {len(stats_list):,}
  Words:             {total_words:,}
  
  Original size:     {total_original:,} bytes
  Compressed size:   {total_compressed:,} bytes
  
  Compression ratio: {ratio:.2f}x
  Space savings:     {savings:.1f}%
  Bits per word:     {bits_per_word:.2f}
  
  Vocabulary size:   {len(compressor.word_to_id):,}
  Entropy limit:     {compressor.entropy:.2f} bits/word
  Coverage:          {coverage:.1f}%
  
  Verification:      {len(stats_list) - failures}/{len(stats_list)} passed
""")
    
    # =========================================================================
    # EXAMPLES
    # =========================================================================
    print("[4] EXAMPLES")
    print("-" * 60)
    
    examples = [
        "Hello, how are you doing today?",
        "The weather is beautiful this morning.",
        "Can you help me with this problem?",
    ]
    
    for text in examples:
        compressed = compressor.compress(text)
        stats = compressor.analyze(text)
        print(f'\n  "{text}"')
        print(f"    {stats['original_bytes']} → {stats['compressed_bytes']} bytes  "
              f"({stats['compression_ratio']:.1f}x, {stats['bits_per_word']:.1f} bits/word)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
