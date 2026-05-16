import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id["<PAD>"] = self.vocab_size
        self.id_to_word["0"] = "<PAD>"
        self.vocab_size += 1
        self.word_to_id["<UNK>"] = self.vocab_size
        self.id_to_word["1"] = "<UNK>"
        self.vocab_size += 1
        self.word_to_id["<BOS>"] = self.vocab_size
        self.id_to_word["2"] = "<BOS>"
        self.vocab_size += 1
        self.word_to_id["<EOS>"] = self.vocab_size
        self.id_to_word["3"] = "<EOS>"
        self.vocab_size += 1

        words = set()
        for text in texts:
            for word in text.split():
                words.add(word.lower())
        words = sorted(list(words))
        for w in words:
            self.word_to_id[w] = self.vocab_size
            self.id_to_word[self.vocab_size] = w
            self.vocab_size += 1
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        res = []
        text = text.split()
        for word in text:
            word = word.lower()
            res.append(self.word_to_id.get(word, 1))
        return res
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        res = []
        for word in ids:
            res.append(self.id_to_word.get(word, "<UNK>"))
        return " ".join(res)
    
