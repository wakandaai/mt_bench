# mt_benchmark/models/factory.py
import yaml
from typing import Dict, Any
from pathlib import Path
from mt_benchmark.models.base import BaseTranslationModel
from mt_benchmark.models.huggingface.hf_model import ToucanModel, NLLBModel, SeamlessModel
from mt_benchmark.models.api_services.base import DSPyAPIModel, GoogleTranslateModel

class ModelFactory:
    """Factory for creating translation models."""
    
    _model_registry = {
        'toucan': ToucanModel,
        'nllb': NLLBModel,
        'seamless': SeamlessModel,
        'api': DSPyAPIModel,
        'google_cloud_translate': GoogleTranslateModel,
    }
    
    @classmethod
    def create_model(cls, model_id: str, config_path: str = None) -> BaseTranslationModel:
        """Create a model instance from configuration.
        
        Args:
            model_id: Model identifier from config (e.g., 'toucan_base', 'gpt4o_mini', 'google_translate')
            config_path: Path to models.yaml config file
            
        Returns:
            BaseTranslationModel instance
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if model_id not in config:
            raise ValueError(f"Model '{model_id}' not found in config")
        
        model_config = config[model_id]
        
        # Determine model type
        if 'type' in model_config:
            # Explicit type specified (e.g., Google Cloud Translate)
            model_type = model_config['type']
            model_name = model_id
        elif 'model' in model_config:
            # API model (has DSPy model string)
            model_type = 'api'
            model_name = model_config['model']
        else:
            # HuggingFace model
            model_name = model_config['model_name']
            model_type = cls._get_model_type(model_id, model_name, model_config)
        
        if model_type not in cls._model_registry:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create and return model
        model_class = cls._model_registry[model_type]
        return model_class(model_name, model_config)
    
    @classmethod
    def _get_model_type(cls, model_id: str, model_name: str, model_config: Dict[str, Any] = None) -> str:
        """Determine model type from model_id, model_name, or config."""
        
        # First check if model_type is explicitly specified in config
        if model_config and 'model_type' in model_config:
            model_type = model_config['model_type']
            # Map generic types to specific handlers
            if model_type == 'seq2seq':
                # Check for specific models
                if 'toucan' in model_id.lower():
                    return 'toucan'
                elif 'seamless' in model_id.lower() or 'm4t' in model_name.lower():
                    return 'seamless'
                else:
                    return 'nllb'  # Use NLLB as generic seq2seq handler
            return model_type
        
        # Then check model_class in config
        if model_config and 'model_class' in model_config:
            model_class = model_config['model_class']
            
            # Map model classes to types
            seq2seq_classes = ['MT5ForConditionalGeneration', 'AutoModelForSeq2SeqLM', 
                              'M2M100ForConditionalGeneration']
            if model_class in seq2seq_classes:
                # Check for specific models first
                if 'toucan' in model_id.lower() or 'toucan' in model_name.lower():
                    return 'toucan'
                elif 'seamless' in model_id.lower() or 'm4t' in model_name.lower():
                    return 'seamless'
                else:
                    return 'nllb'  # Generic seq2seq handler
        
        # Finally, try string matching on model_id/model_name
        if 'toucan' in model_id.lower() or 'toucan' in model_name.lower():
            return 'toucan'
        elif 'nllb' in model_id.lower() or 'nllb' in model_name.lower():
            return 'nllb'
        elif 'seamless' in model_id.lower() or 'm4t' in model_name.lower():
            return 'seamless'
        elif 'madlad' in model_id.lower() or 'madlad' in model_name.lower():
            return 'nllb'  # Use NLLB handler for MADLAD
        else:
            raise ValueError(f"Cannot determine model type for {model_id} / {model_name}")
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type):
        """Register a new model type."""
        cls._model_registry[model_type] = model_class
    
    @classmethod
    def list_available_models(cls, config_path: str = None) -> Dict[str, Dict[str, Any]]:
        """List all available models from config."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config