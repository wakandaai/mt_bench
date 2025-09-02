# mt_benchmark/models/api_services/base.py
import dspy
import time
from typing import List, Dict, Any, Optional, Set
from mt_benchmark.models.base import BaseTranslationModel
from google.cloud import translate_v2 as translate
from google.api_core import exceptions as google_exceptions
from mt_benchmark.config.language_support.code_mapping import iso639_1_to_iso639_3_and_script

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
                print(dspy.inspect_history(n=1))
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

class GoogleTranslateModel(BaseTranslationModel):
    """Google Cloud Translate API model."""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        """Initialize Google Cloud Translate model.
        
        Args:
            model_name: Human-readable model name
            model_config: Configuration including project_id, credentials, etc.
        """
        self.model_name = model_name
        self.config = model_config
        
        # Initialize the translate client
        client_kwargs = {}
        if 'credentials_path' in model_config:
            import os
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = model_config['credentials_path']
        
        if 'project_id' in model_config:
            client_kwargs['project'] = model_config['project_id']
            
        self.translate_client = translate.Client(**client_kwargs)
        
        # Load language code mapping
        self.lang_code_map = model_config.get('lang_code_map', {})
        
        # Cache supported languages
        self._supported_languages = None
        self._load_supported_languages()
    
    def _load_supported_languages(self):
        """Load and cache supported languages from Google Translate."""
        try:
            languages = self.translate_client.get_languages()
            self._supported_languages = {(lang['language']): {'name': lang['name']} for lang in languages}
            print(f"Google Translate supports {len(self._supported_languages)} languages")
        except Exception as e:
            print(f"Warning: Could not fetch supported languages: {e}")
            self._supported_languages = {}
    
    def get_supported_languages(self) -> Set[str]:
        """Get set of language codes supported by Google Translate.
        
            Returns:
            Dict of format {'language_code': {'name': 'language_name'}}
        """
        if self._supported_languages is None:
            self._load_supported_languages()
        return self._supported_languages.copy()
    
    def _map_language_code(self, lang_code: str) -> Optional[str]:
        """Map dataset language code to Google Translate language code."""
        # First check explicit mapping
        if lang_code in self.lang_code_map:
            mapped_code = self.lang_code_map[lang_code]
            if mapped_code is None:
                return None  # Explicitly marked as unsupported
            return mapped_code
        
        # Handle FLORES format codes (e.g., 'eng_Latn' -> 'eng')
        if '_' in lang_code:
            base_code = lang_code.split('_')[0]
        else:
            base_code = lang_code
        
        # Try using the existing mapping utility
        from mt_benchmark.config.language_support.code_mapping import iso639_3_to_iso639_1
        iso_mapping = iso639_3_to_iso639_1()
        
        if base_code in iso_mapping:
            return iso_mapping[base_code]
        
        # Return as-is if already 2 characters (ISO 639-1)
        if len(base_code) == 2:
            return base_code
            
        return None  # Unable to map
    
    def _check_language_support(self, source_lang: str, target_lang: str) -> tuple[Optional[str], Optional[str]]:
        """Check if language pair is supported and return mapped codes."""
        src_mapped = self._map_language_code(source_lang)
        tgt_mapped = self._map_language_code(target_lang)
        
        if src_mapped is None or tgt_mapped is None:
            return None, None
            
        if (src_mapped not in self._supported_languages or 
            tgt_mapped not in self._supported_languages):
            return None, None
            
        return src_mapped, tgt_mapped
    
    def supports_language_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if Google Translate supports this language pair."""
        src_mapped, tgt_mapped = self._check_language_support(source_lang, target_lang)
        return src_mapped is not None and tgt_mapped is not None
    
    def translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate a list of texts.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        # Check language support
        src_mapped, tgt_mapped = self._check_language_support(source_lang, target_lang)
        if src_mapped is None or tgt_mapped is None:
            print(f"Language pair {source_lang}→{target_lang} not supported by Google Translate")
            return [""] * len(texts)  # Return empty translations
        
        translations = []
        batch_size = self.config.get('batch_size', 50)  # Google Translate batch limit
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Call Google Translate API
                results = self.translate_client.translate(
                    batch_texts,
                    source_language=src_mapped,
                    target_language=tgt_mapped,
                    format_='text'
                )
                
                # Extract translated texts
                if isinstance(results, list):
                    batch_translations = [result['translatedText'] for result in results]
                else:
                    batch_translations = [results['translatedText']]
                
                translations.extend(batch_translations)
                
                # Rate limiting
                if self.config.get('rate_limit_delay', 0) > 0:
                    time.sleep(self.config['rate_limit_delay'])
                    
            except google_exceptions.GoogleAPIError as e:
                print(f"Google Translate API error: {e}")
                translations.extend([""] * len(batch_texts))
            except Exception as e:
                print(f"Translation error for batch: {e}")
                translations.extend([""] * len(batch_texts))
        
        return translations
    
    def get_model_info(self) -> Dict[str, str]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'model_type': self.__class__.__name__,
            'service': 'Google Cloud Translate',
            'supported_languages': len(self._supported_languages) if self._supported_languages else 0
        }
    
    @property
    def supports_batch(self) -> bool:
        """Google Translate supports batch processing."""
        return True
    
    def print_language_support_info(self):
        """Print information about language support for debugging."""
        print("Google Translate Language Support:")
        print(f"Total supported languages: {len(self._supported_languages)}")
        
        # Check your dataset languages
        test_languages = ['eng', 'fra', 'swa', 'dik', 'fon', 'kin', 'zul', 'xho']
        print("\nDataset language support:")
        for lang in test_languages:
            mapped = self._map_language_code(lang)
            if mapped is None:
                status = "❌ Not mappable"
            elif mapped in self._supported_languages:
                status = f"✅ Supported as '{mapped}'"
            else:
                status = f"❌ Mapped to '{mapped}' but not supported"
            print(f"  {lang}: {status}")