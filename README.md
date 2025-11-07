# Hindi BPE Tokenizer

Byte Pair Encoding (BPE) tokenizer optimized for Hindi text using Devanagari script.

## Requirements ✅
- ✅ **Vocabulary**: 5,500 tokens (target: 5,000+)
- ✅ **Compression**: 6.52X (target: 3.0X+)

## Project Structure

```
week11/
├── hindi_bpe_tokenizer.py          # Core implementation (8KB)
├── hindi_bpe_tokenizer.json        # Trained model (543MB)
├── train_bpe_simple.py             # Training script (5KB)
├── hindi_corpus.txt                # Training data (1.5MB)
├── training_results.json           # Training stats (2KB)
└── pyproject.toml & uv.lock        # Dependencies
```

## Key Features

- **Hindi-Optimized**: Devanagari Unicode prioritization (`\u0900-\u097F`)
- **High Compression**: 6.52X average, up to 10.44X on technical text
- **Perfect Decoding**: 100% accuracy in text reconstruction
- **Simple API**: Easy encode/decode with compression stats

## Performance

| Metric | Value |
|--------|-------|
| Vocabulary Size | 5,500 tokens |
| Compression Ratio | 6.52X (avg), 10.44X (best) |
| Decoding Accuracy | 100% |
| Corpus Size | 575K chars, 1.5MB |

## Quick Start

### Use Pre-trained Model
```python
from hindi_bpe_tokenizer import HindiBPETokenizer

tokenizer = HindiBPETokenizer()
tokenizer.load('hindi_bpe_tokenizer.json')

# Encode & Decode
tokens = tokenizer.encode("भारत एक महान देश है।")
decoded = tokenizer.decode(tokens)
stats = tokenizer.get_compression_stats("भारत")
```

### Train New Model
```bash
uv run train_bpe_simple.py    # Train tokenizer (includes tests)
```

## Technical Details

**BPE Algorithm:**
1. Start with 256 byte vocabulary
2. Find most frequent byte pair
3. Merge into new token
4. Repeat until target vocab size

**Hindi Optimizations:**
- Devanagari blocks: `\u0900-\u097F`, `\u0980-\u09FF`
- Optimized regex for Hindi word patterns
- JSON-based serialization

**Dependencies:** `regex`, `numpy` (Python 3.13+)

---

🙏 **धन्यवाद (Thank you) for using Hindi BPE Tokenizer!**

