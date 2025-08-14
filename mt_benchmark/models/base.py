# mt_benchmark/models/base.py

from abc import ABC, abstractmethod
from typing import List, Dict

class BaseTranslationModel(ABC):
    @abstractmethod
    def translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate a list of texts from source_lang to target_lang.
        
        Args:
            texts: List of strings to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated strings
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the model.
        
        Returns:
            Dictionary containing model metadata
        """
        pass
    
    @property
    @abstractmethod
    def supports_batch(self) -> bool:
        """Whether the model supports batch processing.
        
        Returns:
            True if batch processing is supported
        """
        pass