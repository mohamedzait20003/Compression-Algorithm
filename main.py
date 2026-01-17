import re
from wordfreq import top_n_list
from lib.Huffman import Huffman
from utilities import Compressor, Decompressor

print("=" * 70)
print("HUFFMAN CODING (Word-Level) - Pre-trained with WordFreq")
print("=" * 70 + "\n")

# Get top 10,000 most common English words
print("Loading training data from wordfreq...")
common_words = top_n_list('en', 10000)

training_tokens = []
for i, word in enumerate(common_words):
    weight = max(1, 10000 // (i + 1))
    training_tokens.extend([word] * weight)

total_words = len(training_tokens)
training_tokens.extend([' '] * (total_words // 2))  
training_tokens.extend(['.'] * (total_words // 20)) 
training_tokens.extend([','] * (total_words // 15))
training_tokens.extend(['!'] * (total_words // 100))
training_tokens.extend(['?'] * (total_words // 100))
training_tokens.extend(["'"] * (total_words // 30))
training_tokens.extend(['"'] * (total_words // 200))
training_tokens.extend(['-'] * (total_words // 100))
training_tokens.extend([':'] * (total_words // 200))
training_tokens.extend([';'] * (total_words // 300))

print(f"Training corpus size: {len(training_tokens):,} tokens")
print(f"Unique tokens: {len(set(training_tokens)):,}")

# Create and train Huffman model
huffman = Huffman()
huffman.train(training_tokens)

print(f"Huffman codebook built with {len(huffman.get_codebook()):,} entries")
print("\nSample codebook entries (shortest codes = most frequent):")
sorted_codes = sorted(huffman.get_codebook().items(), key=lambda x: len(x[1]))
for token, code in sorted_codes[:15]:
    display = repr(token) if token.strip() == '' else token
    print(f"  {display}: {code} ({len(code)} bits)")

print("\n" + "=" * 70)
print("COMPRESSION TESTS (10 Examples)")
print("=" * 70 + "\n")

# Initialize compressor and decompressor with pre-trained Huffman
compressor = Compressor(huffman=huffman, max_words=50)
decompressor = Decompressor(huffman=huffman)

# Test messages - varied chat/text scenarios
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

def tokenize(text: str):
    """Tokenize text into words and punctuation"""
    return re.findall(r'\w+|[^\w\s]|\s+', text)

# Track stats for average
all_stats = []

for i, message in enumerate(test_messages, 1):
    print(f"Test {i}: '{message}'")
    print(f"  Words: {len(message.split())}, Characters: {len(message)}")
    
    # Tokenize for encoding (need tokens that exist in codebook)
    tokens = tokenize(message.lower())
    
    # Check for unknown tokens and retrain if needed
    unknown = [t for t in tokens if t not in huffman.get_codebook()]
    if unknown:
        print(f"  ⚠ Unknown tokens: {unknown} - adding to codebook")
        huffman.train(training_tokens + tokens)
    
    # Get original size
    original_bits = len(message.encode('utf-8')) * 8
    
    # Encode with pre-trained codebook
    encoded_bits = huffman.encode(tokens)
    huffman_bits = len(encoded_bits)
    
    stats = {
        'original_bits': original_bits,
        'huffman_bits': huffman_bits,
        'compression_ratio': original_bits / huffman_bits if huffman_bits > 0 else 0,
        'savings_percent': (1 - huffman_bits / original_bits) * 100,
        'bits_per_word': huffman_bits / len(message.split())
    }
    all_stats.append(stats)
    
    # Decode and verify
    decoded_tokens = huffman.decode(encoded_bits)
    restored = ''.join(decoded_tokens)
    match = restored == message.lower()
    
    print(f"  Original: {original_bits} bits → Huffman: {huffman_bits} bits")
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
total_original = sum(s['original_bits'] for s in all_stats)
total_huffman = sum(s['huffman_bits'] for s in all_stats)

print(f"Total samples: {len(all_stats)}")
print(f"Average compression ratio: {avg_ratio:.2f}x")
print(f"Average savings (Huffman only): {avg_savings:.1f}%")
print(f"Average bits per word: {avg_bits_per_word:.2f}")
print(f"Total original bits: {total_original:,}")
print(f"Total Huffman bits: {total_huffman:,}")
print(f"Overall savings: {(1 - total_huffman/total_original) * 100:.1f}%")

print("\n" + "=" * 70)
print("NOTE: These savings are for Huffman encoding only (no codebook overhead).")
print("In practice, the pre-trained codebook is shared between sender/receiver,")
print("eliminating the need to transmit it with each message.")
print("=" * 70)