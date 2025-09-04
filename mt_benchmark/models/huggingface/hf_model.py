# mt_benchmark/models/huggingface/hf_model.py
from typing import List, Dict, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoProcessor, M2M100ForConditionalGeneration, MT5ForConditionalGeneration, NllbTokenizerFast, SeamlessM4Tv2Model, T5Tokenizer
from mt_benchmark.models.base import BaseTranslationModel
from mt_benchmark.config.language_support.nllb import nllb_languages_supported
from mt_benchmark.config.language_support.seamless import seamless_languages_supported
from mt_benchmark.config.language_support.toucan import toucan_languages_supported, TOUCAN_TO_FLORES_MAPPING

class HuggingFaceModel(BaseTranslationModel):
    """Base class for local HuggingFace models."""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        self.model_name = model_name
        self.config = model_config
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._supported_languages = None
        self._load_model()
        self._load_supported_languages()
    
    def _load_model(self):
        """Load the model and tokenizer based on configuration."""
        # Get model class - default to AutoModelForSeq2SeqLM
        model_class = self.config.get('model_class', 'auto')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model with specified parameters
        model_kwargs = {
            'torch_dtype': getattr(torch, self.config.get('torch_dtype', 'float32')),
            'device_map': self.config.get('device_map', 'auto')
        }
        
        if model_class == 'MT5ForConditionalGeneration':
            self.model = MT5ForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        elif model_class == 'SeamlessM4Tv2Model':
            self.model = SeamlessM4Tv2Model.from_pretrained(self.model_name, **model_kwargs)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        elif model_class == 'NllbModel':
            self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
            self.tokenizer = NllbTokenizerFast.from_pretrained(self.model_name)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model.eval()
    
    def _load_supported_languages(self):
        """Load supported languages. Must be implemented by subclasses."""
        self._supported_languages = {}
    
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
    
    def get_supported_languages(self) -> Dict[str, Dict[str, str]]:
        """Get supported languages. Must be implemented by subclasses."""
        if self._supported_languages is None:
            self._load_supported_languages()
        return self._supported_languages.copy()
    
    def supports_language_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if model supports translating between language pair."""
        supported = self.get_supported_languages()
        return source_lang in supported and target_lang in supported
    
    @property
    def supports_batch(self) -> bool:
        return True


class ToucanModel(HuggingFaceModel):
    """Toucan model implementation."""
    
    def _load_supported_languages(self):
        """Load Toucan supported languages."""
        self._supported_languages = toucan_languages_supported()
    
    def _format_input(self, text: str, source_lang: str, target_lang: str) -> str:
        """Format input with target language prefix for Toucan."""
        # Convert from FLORES format to Toucan format if needed
        target_code = target_lang
        if target_lang in TOUCAN_TO_FLORES_MAPPING.values():
            # Reverse lookup to find toucan code
            for toucan_code, flores_code in TOUCAN_TO_FLORES_MAPPING.items():
                if flores_code == target_lang:
                    target_code = toucan_code.split('_')[0]  # Get just the language part
                    break
        else:
            # Use the base language code
            target_code = target_lang.split('_')[0]
        
        return f"{target_code}: {text}"

    def supports_language_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if Toucan supports this language pair."""
        supported = self.get_supported_languages()
        
        # Check direct support
        if source_lang in supported and target_lang in supported:
            return True
        
        # Check with FLORES mapping
        flores_source = TOUCAN_TO_FLORES_MAPPING.get(source_lang, source_lang)
        flores_target = TOUCAN_TO_FLORES_MAPPING.get(target_lang, target_lang)
        
        return flores_source in supported and flores_target in supported


class SeamlessModel(HuggingFaceModel):
    """Seamless M4T-V2 model implementation."""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        super().__init__(model_name, model_config)
    
    def _load_supported_languages(self):
        """Load Seamless supported languages."""
        self._supported_languages = seamless_languages_supported()

    def translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate with Seamless-specific language handling."""
        if not self.processor:
            raise RuntimeError("Processor not loaded for SeamlessModel")

        # Map language codes to seamless format: ISO-639-3 codes only
        src_lang_code = source_lang.split('_')[0]
        tgt_lang_code = target_lang.split('_')[0]

        try:
            text_inputs = self.processor(
                text=texts,
                src_lang=src_lang_code,
                return_tensors="pt"
            )
            
            # Move to device
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items() if isinstance(v, torch.Tensor)}
            
            # Generate
            with torch.no_grad():
                output_tokens = self.model.generate(
                    **text_inputs,
                    tgt_lang=tgt_lang_code,
                    generate_speech=False
                )
            
            # Decode
            translated_text = self.processor.batch_decode(
                output_tokens[0].tolist(),
                skip_special_tokens=True
            )
            
            return [self._postprocess_output(trans) for trans in translated_text]
            
        except Exception as e:
            print(f"SeamlessM4T translation error: {e}")
            import traceback
            traceback.print_exc()
            return [""] * len(texts)  # Return empty translations for all texts

    def supports_language_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if Seamless supports this language pair."""
        supported = self.get_supported_languages()
        return source_lang in supported and target_lang in supported


class NLLBModel(HuggingFaceModel):
    """NLLB model implementation."""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        super().__init__(model_name, model_config)
    
    def _load_supported_languages(self):
        """Load NLLB supported languages."""
        self._supported_languages = nllb_languages_supported()
    
    def _format_input(self, text: str, source_lang: str, target_lang: str) -> str:
        """NLLB uses language codes in tokenizer, handled during tokenization."""
        return text
    
    def translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate with NLLB-specific language handling."""

        # handle language codes, if they have glottocodes, remove it for example:
        # twi_Latn_akua1239 -> twi_Latn
        if (len(source_lang.split('_')) == 3):
            source_lang_code = '_'.join(source_lang.split('_')[:2])
        else:
            source_lang_code = source_lang
        if (len(target_lang.split('_')) == 3):
            tgt_lang_code = '_'.join(target_lang.split('_')[:2])
        else:
            tgt_lang_code = target_lang
        self.tokenizer.src_lang = source_lang_code
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
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang_code)
        generation_config['forced_bos_token_id'] = forced_bos_token_id
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_config)
        
        # Decode
        translations = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        return [self._postprocess_output(trans) for trans in translations]

    def supports_language_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if NLLB supports this language pair."""
        supported = self.get_supported_languages()
        return source_lang in supported and target_lang in supported