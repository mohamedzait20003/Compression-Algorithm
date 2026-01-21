"""
Data Loader for DailyDialog dataset.
"""

from tqdm import tqdm
from datasets import load_dataset


class DataLoader:
    """
    Loads sentences from the DailyDialog dataset.
    
    DailyDialog is a multi-turn dialogue dataset containing ~13k dialogues
    with natural conversational language.
    """
    
    DATASET_FILES = {
        "train": "https://huggingface.co/datasets/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full/train/0000.parquet",
        "validation": "https://huggingface.co/datasets/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full/validation/0000.parquet",
        "test": "https://huggingface.co/datasets/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full/test/0000.parquet"
    }
    
    def __init__(self):
        self._dataset = None
        self._cache = {}
    
    def _load_dataset(self):
        """Lazy load the dataset."""
        if self._dataset is None:
            print("Loading DailyDialog dataset...")
            self._dataset = load_dataset("parquet", data_files=self.DATASET_FILES)
        return self._dataset
    
    def load(self, split: str = "train", max_sentences: int = None) -> list[str]:
        """
        Load sentences from a dataset split.
        
        Args:
            split: One of "train", "validation", "test"
            max_sentences: Maximum sentences to load (None for all)
            
        Returns:
            List of sentence strings
        """
        if split not in self.DATASET_FILES:
            raise ValueError(f"Invalid split. Must be one of: {list(self.DATASET_FILES.keys())}")
        
        cache_key = (split, max_sentences)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        ds = self._load_dataset()
        sentences = []
        
        for item in tqdm(ds[split], desc=f"Extracting {split} sentences"):
            for utterance in item['utterances']:
                utterance = utterance.strip()
                if utterance:
                    sentences.append(utterance)
                    if max_sentences and len(sentences) >= max_sentences:
                        print(f"Extracted {len(sentences)} sentences from {split}")
                        self._cache[cache_key] = sentences
                        return sentences
        
        print(f"Extracted {len(sentences)} sentences from {split}")
        self._cache[cache_key] = sentences
        return sentences
    
    def load_train(self, max_sentences: int = None) -> list[str]:
        """Load training sentences."""
        return self.load("train", max_sentences)
    
    def load_test(self, max_sentences: int = None) -> list[str]:
        """Load test sentences."""
        return self.load("test", max_sentences)
    
    def load_validation(self, max_sentences: int = None) -> list[str]:
        """Load validation sentences."""
        return self.load("validation", max_sentences)
