# src/models/huggingface/local.py
from typing import List, Dict, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MT5ForConditionalGeneration
from ..base import BaseTranslationModel

class HuggingFaceModel(BaseTranslationModel):
    """Base class for local HuggingFace models."""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        self.model_name = model_name
        self.config = model_config
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer based on configuration."""
        # Get model class - default to AutoModelForSeq2SeqLM
        model_class = self.config.get('model_class', 'auto')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with specified parameters
        model_kwargs = {
            'torch_dtype': getattr(torch, self.config.get('torch_dtype', 'float32')),
            'device_map': self.config.get('device_map', 'auto')
        }
        
        if model_class == 'MT5ForConditionalGeneration':
            self.model = MT5ForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **model_kwargs)
        
        self.model.eval()
    
    def translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate texts using the loaded model."""
        formatted_texts = [self._format_input(text, source_lang, target_lang) for text in texts]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            formatted_texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=self.config.get('max_input_length', 1024)
        )
        
        # Move to device if needed
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        generation_config = self._get_generation_config(texts)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_config)
        
        # Decode
        translations = self.tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            skip_prompt=True
        )
        
        return [self._postprocess_output(trans) for trans in translations]
    
    def _format_input(self, text: str, source_lang: str, target_lang: str) -> str:
        """Format input text according to model requirements. Override in subclasses."""
        return text
    
    def _postprocess_output(self, text: str) -> str:
        """Postprocess model output. Override in subclasses if needed."""
        return text.strip()
    
    def _get_generation_config(self, texts: List[str]) -> Dict[str, Any]:
        """Get generation configuration."""
        config = self.config.get('generation_config', {})
        
        # Set max_new_tokens based on input length if not specified
        if 'max_new_tokens' not in config:
            config['max_new_tokens'] = 128
        
        return config
    
    def get_model_info(self) -> Dict[str, str]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'model_type': self.__class__.__name__,
            'device': str(next(self.model.parameters()).device),
            'dtype': str(next(self.model.parameters()).dtype)
        }
    
    @property
    def supports_batch(self) -> bool:
        return True


class ToucanModel(HuggingFaceModel):
    """Toucan model implementation."""
    
    def _format_input(self, text: str, source_lang: str, target_lang: str) -> str:
        """Format input with target language prefix for Toucan."""
        return f"{target_lang}: {text}"


class NLLBModel(HuggingFaceModel):
    """NLLB model implementation."""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        # NLLB language code mapping
        self.lang_code_map = model_config.get('lang_code_map', {})
        super().__init__(model_name, model_config)
    
    def _format_input(self, text: str, source_lang: str, target_lang: str) -> str:
        """NLLB uses language codes in tokenizer, handled during tokenization."""
        return text
    
    def translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate with NLLB-specific language handling."""
        # Map language codes if mapping provided
        src_lang_code = self.lang_code_map.get(source_lang, source_lang)
        tgt_lang_code = self.lang_code_map.get(target_lang, target_lang)
        
        # Set source language for tokenizer
        if hasattr(self.tokenizer, 'src_lang'):
            self.tokenizer.src_lang = src_lang_code
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get('max_input_length', 1024)
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with target language
        generation_config = self._get_generation_config(texts)
        
        # Set target language token
        if hasattr(self.tokenizer, 'lang_code_to_id'):
            forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang_code]
            generation_config['forced_bos_token_id'] = forced_bos_token_id
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_config)
        
        # Decode
        translations = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return [self._postprocess_output(trans) for trans in translations]