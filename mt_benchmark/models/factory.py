# mt_benchmark/models/factory.py
import yaml
from typing import Dict, Any
from pathlib import Path
from mt_benchmark.models.base import BaseTranslationModel
from mt_benchmark.models.huggingface.hf_model import ToucanModel, NLLBModel
from mt_benchmark.models.api_services.base import DSPyAPIModel

class ModelFactory:
    """Factory for creating translation models."""
    
    _model_registry = {
        'toucan': ToucanModel,
        'nllb': NLLBModel,
        'api': DSPyAPIModel,
    }
    
    @classmethod
    def create_model(cls, model_id: str, config_path: str = None) -> BaseTranslationModel:
        """Create a model instance from configuration.
        
        Args:
            model_id: Model identifier from config (e.g., 'toucan_base', 'gpt4o_mini')
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
        if 'model' in model_config:
            # API model (has DSPy model string)
            model_type = 'api'
            model_name = model_config['model']
        else:
            # HuggingFace model
            model_name = model_config['model_name']
            model_type = cls._get_model_type(model_id, model_name)
        
        if model_type not in cls._model_registry:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create and return model
        model_class = cls._model_registry[model_type]
        return model_class(model_name, model_config)
    
    @classmethod
    def _get_model_type(cls, model_id: str, model_name: str) -> str:
        """Determine model type from model_id or model_name."""
        if 'toucan' in model_id.lower() or 'toucan' in model_name.lower():
            return 'toucan'
        elif 'nllb' in model_id.lower() or 'nllb' in model_name.lower():
            return 'nllb'
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