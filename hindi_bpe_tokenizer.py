"""
Hindi BPE Tokenizer
This implements a Byte Pair Encoding tokenizer specifically for Hindi text.
Target: 5000+ vocabulary size, 3+ compression ratio
"""

import regex as re
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import json

class HindiBPETokenizer:
    """Byte Pair Encoding tokenizer optimized for Hindi text"""
    
    def __init__(self, vocab_size: int = 5000):
        """
        Initialize the tokenizer
        
        Args:
            vocab_size: Target vocabulary size (must be >= 256)
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        
        self.vocab_size = vocab_size
        self.merges = {}  # (int, int) -> int
        self.vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        # Optimized pattern for Hindi/Devanagari text
        # Matches: optional space + Devanagari letters, numbers, punctuation, whitespace
        self.pattern = re.compile(
            r""" ?[\u0900-\u097F]+| ?[\u0980-\u09FF]+| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.UNICODE
        )
        
    def get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        """
        Count frequency of consecutive byte pairs
        
        Args:
            ids: List of token IDs
            
        Returns:
            Dictionary mapping pairs to their counts
        """
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts
    
    def merge(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """
        Merge all occurrences of a pair into a new token
        
        Args:
            ids: List of token IDs
            pair: Pair to merge
            idx: New token ID for the merged pair
            
        Returns:
            New list with pairs merged
        """
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def train(self, text: str, verbose: bool = True):
        """
        Train the BPE tokenizer on the given text
        
        Args:
            text: Training text corpus
            verbose: Whether to print progress
        """
        # Tokenize text into chunks using the pattern
        text_chunks = re.findall(self.pattern, text)
        
        # Convert each chunk to UTF-8 bytes and then to integer list
        tokens_list = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = list(chunk_bytes)
            tokens_list.append(chunk_ids)
        
        # Flatten for training
        all_tokens = []
        for tokens in tokens_list:
            all_tokens.extend(tokens)
        
        if verbose:
            print(f"Training corpus size: {len(text)} characters")
            print(f"Total UTF-8 bytes: {len(all_tokens)}")
            print(f"Target vocabulary size: {self.vocab_size}")
        
        # Number of merges to perform
        num_merges = self.vocab_size - 256
        ids = list(all_tokens)
        
        # Perform BPE merges
        merges = {}
        for i in range(num_merges):
            stats = self.get_stats(ids)
            
            if not stats:
                if verbose:
                    print(f"No more pairs to merge at iteration {i}")
                break
            
            # Find most frequent pair
            pair = max(stats, key=stats.get)
            idx = 256 + i
            
            if verbose and (i % 100 == 0 or i < 10):
                print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} (freq: {stats[pair]})")
            
            # Merge the pair
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx
            
            # Update vocabulary
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        
        self.merges = merges
        
        if verbose:
            compression_ratio = len(all_tokens) / len(ids)
            print(f"\nTraining complete!")
            print(f"Final vocabulary size: {len(self.vocab)}")
            print(f"Number of merges: {len(merges)}")
            print(f"Original bytes: {len(all_tokens)}")
            print(f"Compressed tokens: {len(ids)}")
            print(f"Compression ratio: {compression_ratio:.2f}X")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        # Split text into chunks
        text_chunks = re.findall(self.pattern, text)
        
        # Encode each chunk
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = list(chunk_bytes)
            
            # Apply merges
            while len(chunk_ids) >= 2:
                stats = self.get_stats(chunk_ids)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                chunk_ids = self.merge(chunk_ids, pair, idx)
            
            ids.extend(chunk_ids)
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def save(self, file_path: str):
        """Save the tokenizer to a file"""
        data = {
            'vocab_size': self.vocab_size,
            'merges': {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},  # Convert tuple keys to string
            'vocab': {k: list(v) for k, v in self.vocab.items()},  # Convert bytes to list for JSON
            'pattern': self.pattern.pattern
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {file_path}")
    
    def get_compression_stats(self, text: str) -> Dict:
        """
        Get compression statistics for given text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with compression statistics
        """
        original_bytes = len(text.encode("utf-8"))
        encoded = self.encode(text)
        compressed_tokens = len(encoded)
        compression_ratio = original_bytes / compressed_tokens if compressed_tokens > 0 else 0
        
        return {
            'original_chars': len(text),
            'original_bytes': original_bytes,
            'compressed_tokens': compressed_tokens,
            'compression_ratio': compression_ratio,
            'vocab_size': len(self.vocab)
        }


if __name__ == "__main__":
    # This will be used for testing with a small corpus
    sample_text = """
    इस वाक्य का अनुवाद गूगल ट्रांसलेट द्वारा किया जा रहा है।
    यह एक नमूना हिंदी पाठ है।
    हम बाइट पेयर एन्कोडिंग का उपयोग कर रहे हैं।
    """
    
    tokenizer = HindiBPETokenizer(vocab_size=500)
    tokenizer.train(sample_text)
    
    test_text = "इस वाक्य का परीक्षण किया जा रहा है।"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nTest encoding:")
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

