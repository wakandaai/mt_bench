# src/models/api_services/base.py
import dspy
import time
from typing import List, Dict, Any
from ..base import BaseTranslationModel

class TranslationSignature(dspy.Signature):
    """Translate text from source language to target language using provided language codes."""
    source_text = dspy.InputField(desc="Text to translate")
    source_lang = dspy.InputField(desc="Source language code")
    target_lang = dspy.InputField(desc="Target language code")
    translation = dspy.OutputField(desc="Translated text in the target language")

class DSPyAPIModel(BaseTranslationModel):
    """Generic DSPy model for any API service."""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        """Initialize DSPy API model.
        
        Args:
            model_name: Human-readable model name
            model_config: Model configuration dictionary
        """
        self.model_name = model_name
        self.config = model_config
        
        # Create DSPy LM with the model string
        self.dspy_lm = dspy.LM(
            model=model_config['model'],
            **model_config.get('lm_kwargs', {})
        )
        
        # Configure DSPy to use this model
        dspy.settings.configure(lm=self.dspy_lm)
        
        # Create the translation module
        self.translator = dspy.Predict(TranslationSignature)
    
    def translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate a list of texts.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        translations = []
        
        for text in texts:
            try:
                # Use DSPy to generate translation
                result = self.translator(
                    source_text=text,
                    source_lang=source_lang,
                    target_lang=target_lang
                )
                dspy.inspect_history(n=1)
                translations.append(result.translation)
                
                # Add small delay to respect rate limits
                if self.config.get('rate_limit_delay', 0) > 0:
                    time.sleep(self.config['rate_limit_delay'])
                    
            except Exception as e:
                print(f"Translation error for text '{text[:50]}...': {e}")
                translations.append("")  # Empty translation on error
        
        return translations
    
    def get_model_info(self) -> Dict[str, str]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'model_type': self.__class__.__name__,
            'dspy_model': self.config.get('model', 'unknown'),
        }
    
    @property
    def supports_batch(self) -> bool:
        """DSPy processes one at a time by default."""
        return True  # We handle batching in translate method