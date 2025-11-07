"""
Train Hindi BPE Tokenizer - Simple Version
"""

from hindi_bpe_tokenizer import HindiBPETokenizer
import json

# Load corpus
print("=" * 80)
print("Hindi BPE Tokenizer Training")
print("=" * 80)
print()

print("Loading corpus...")
with open('hindi_corpus.txt', 'r', encoding='utf-8') as f:
    corpus = f.read()

print(f"✓ Corpus loaded")
print(f"  Total characters: {len(corpus):,}")
print(f"  Total bytes (UTF-8): {len(corpus.encode('utf-8')):,}")

# Train tokenizer
print("\n" + "=" * 80)
print("Training BPE Tokenizer...")
print("=" * 80)

vocab_size = 5500  # Increase target to ensure we hit 5000+
tokenizer = HindiBPETokenizer(vocab_size=vocab_size)
tokenizer.train(corpus, verbose=True)

# Save tokenizer
tokenizer.save('hindi_bpe_tokenizer.json')

# Test on various Hindi texts
print("\n" + "=" * 80)
print("Testing the Tokenizer...")
print("=" * 80)

test_texts = [
    "भारतीय अंतरिक्ष अनुसंधान संगठन ने चंद्रयान-3 मिशन को सफलतापूर्वक लॉन्च किया।",
    "यह एक बहुत ही महत्वपूर्ण उपलब्धि है जो भारत के अंतरिक्ष कार्यक्रम के लिए एक मील का पत्थर है।",
    "विज्ञान और प्रौद्योगिकी के क्षेत्र में भारत ने उल्लेखनीय प्रगति की है।",
    "हिंदी भाषा विश्व की प्रमुख भाषाओं में से एक है।",
    "भारत ने चौथे टी-20 मैच में ऑस्ट्रेलिया को 48 रन से हरा दिया।",
    "वॉशिंगटन सुंदर ने 3 रन देकर 3 विकेट झटके।",
    "शुभमन गिल ने सबसे ज्यादा 46 रनों की पारी खेली।",
    "जसप्रीत बुमराह के 99 विकेट हो गए हैं।"
]

total_original_bytes = 0
total_compressed_tokens = 0

for i, text in enumerate(test_texts, 1):
    print(f"\nTest {i}:")
    print(f"Text: {text[:60]}..." if len(text) > 60 else f"Text: {text}")
    
    stats = tokenizer.get_compression_stats(text)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"  Original bytes: {stats['original_bytes']}")
    print(f"  Compressed tokens: {stats['compressed_tokens']}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}X")
    print(f"  Decoding matches: {'✓' if text == decoded else '✗'}")
    
    total_original_bytes += stats['original_bytes']
    total_compressed_tokens += stats['compressed_tokens']

overall_compression = total_original_bytes / total_compressed_tokens if total_compressed_tokens > 0 else 0

print("\n" + "=" * 80)
print("Final Results:")
print("=" * 80)
print(f"✓ Vocabulary size: {len(tokenizer.vocab):,} tokens")
print(f"✓ Number of merges: {len(tokenizer.merges):,}")
print(f"✓ Overall test compression ratio: {overall_compression:.2f}X")

# Check if requirements are met
print("\n" + "=" * 80)
print("Requirements Check:")
print("=" * 80)

vocab_ok = len(tokenizer.vocab) >= 5000
compression_ok = overall_compression >= 3.0

print(f"{'✓' if vocab_ok else '✗'} Vocabulary size >= 5000: {vocab_ok} ({len(tokenizer.vocab):,} tokens)")
print(f"{'✓' if compression_ok else '✗'} Compression ratio >= 3.0: {compression_ok} ({overall_compression:.2f}X)")

if vocab_ok and compression_ok:
    print("\n🎉 SUCCESS! All requirements met!")
else:
    print("\n⚠️  Some requirements not met.")

# Save detailed statistics
results = {
    'vocab_size': len(tokenizer.vocab),
    'num_merges': len(tokenizer.merges),
    'compression_ratio': overall_compression,
    'corpus_size_bytes': len(corpus.encode('utf-8')),
    'corpus_size_chars': len(corpus),
    'requirements_met': {
        'vocab_size_5000+': vocab_ok,
        'compression_3+': compression_ok
    },
    'test_results': [
        {
            'text': text[:50] + '...' if len(text) > 50 else text,
            'original_bytes': tokenizer.get_compression_stats(text)['original_bytes'],
            'compressed_tokens': tokenizer.get_compression_stats(text)['compressed_tokens'],
            'compression_ratio': tokenizer.get_compression_stats(text)['compression_ratio']
        }
        for text in test_texts
    ]
}

with open('training_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✓ Results saved to training_results.json")

