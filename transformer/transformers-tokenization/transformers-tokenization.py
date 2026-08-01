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
        unique_words=set()
        word_id={}
        for str in texts:
            words=str.strip().lower().split()
            for word in words:
                unique_words.add(word)

        sorted_unique_words=sorted(unique_words)
        self.vocab_size=len(sorted_unique_words)+4
        #print(self.vocab_size)
        #print(sorted_unique_words)
        self.word_to_id={self.pad_token:0, self.unk_token:1, self.bos_token:2, self.eos_token:3}

       
        id=4
        for word in sorted_unique_words:
            self.word_to_id[word]=id
            self.id_to_word[id]=word
            id+=1


        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        unique_words=set()
        word_id={}
        #print(text)
        words=text.strip().lower().split()
        for word in words:
            #print(word)
            unique_words.add(word)

        sorted_unique_words=sorted(unique_words)
        words_id=[]
        #print(sorted_unique_words)
        for word in sorted_unique_words:
            #print(word)
            if word in self.word_to_id:
                
                words_id.append(self.word_to_id[word])
            else:
                words_id.append(1)

        #print(words_id)

        return words_id
            

        
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        #print(ids)
        text=""
        #id_=[4,9]
        id_=[999,4]
        text1=""
        for id in ids:
            if id in self.id_to_word:
                text=text+" "+self.id_to_word[id]
            else:
                text=text+" "+"<UNK>"

        text=text.strip()

        for id in id_:
            if id in self.id_to_word:
                text1=text1+" "+self.id_to_word[id]
            else:
                text1=text1+" "+"<UNK>"

        text1=text1.strip()
        print(text1)

        #print(text)
                
            
            

        return text
        