# mt_benchmark/datasets/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass

@dataclass
class TranslationSample:
    """A single translation sample with metadata."""
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    sample_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class LanguagePair:
    """A language pair with direction."""
    source_lang: str
    target_lang: str
    
    def __str__(self):
        return f"{self.source_lang}→{self.target_lang}"
    
    def reverse(self) -> 'LanguagePair':
        """Get the reverse direction."""
        return LanguagePair(self.target_lang, self.source_lang)

class BaseDataset(ABC):
    """Abstract base class for translation datasets."""
    
    @abstractmethod
    def get_language_pair(self, source_lang: str, target_lang: str) -> List[TranslationSample]:
        """Get all samples for a specific language pair.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of TranslationSample objects
        """
        pass
    
    @abstractmethod
    def list_language_pairs(self) -> List[LanguagePair]:
        """List all available language pairs including both directions.
        
        Returns:
            List of LanguagePair objects
        """
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset metadata and statistics.
        
        Returns:
            Dictionary with dataset information
        """
        pass
    
    def get_language_pair_batch(self, source_lang: str, target_lang: str, 
                               batch_size: int = 1) -> Iterator[List[TranslationSample]]:
        """Get samples in batches for a language pair.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Number of samples per batch
            
        Yields:
            Batches of TranslationSample objects
        """
        samples = self.get_language_pair(source_lang, target_lang)
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]
    
    def get_all_pairs_data(self) -> Dict[str, List[TranslationSample]]:
        """Get data for all available language pairs.
        
        Returns:
            Dictionary mapping language pair strings to sample lists
        """
        result = {}
        for pair in self.list_language_pairs():
            pair_key = f"{pair.source_lang}_{pair.target_lang}"
            result[pair_key] = self.get_language_pair(pair.source_lang, pair.target_lang)
        return result