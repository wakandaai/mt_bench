# MT-Benchmark: Machine Translation Benchmarking Framework

A comprehensive framework for evaluating machine translation models on standard datasets like FLORES, supporting both local HuggingFace models and API-based services.

## Features

- **Multi-model Support**: Local HuggingFace models (Toucan, NLLB, Seamless M4T) and API services (OpenAI GPT, Google Translate)
- **Standard datasets**: FLORES dataset support with automatic language pair discovery
- **Comprehensive Metrics**: BLEU and chrF++ scores with configurable parameters
- **Batch Processing**: Efficient evaluation with configurable batch sizes
- **Structured Results**: Organized output with predictions, metrics, and experiment summaries
- **Extensible Architecture**: Easy to add new models and datasets

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for local models)
- conda or pip package manager

### Environment Setup

1. **Create conda environment**:
```bash
conda create -n mt python=3.10
conda activate mt
```

2. **Install**
```bash
pip install -e .
```

### Dataset Setup
Download and dataprep of [FLORESPLUS](https://huggingface.co/datasets/openlanguagedata/flores_plus) can be done by running the script:
```bash
python scripts/flores_dataprep.py --output_dir FLORES_PLUS
```

## Quick Start

### Basic Evaluation

```python
from mt_benchmark.models.factory import ModelFactory
from mt_benchmark.datasets.flores import FloresDataset
from mt_benchmark.evaluation.pipeline import EvaluationPipeline

# Load model and dataset
model = ModelFactory.create_model('nllb_200_3.3B')
dataset = FloresDataset("FLORES_PLUS/devtest")

# Create evaluation pipeline
evaluator = EvaluationPipeline(
    output_dir="results",
    batch_size=16,
    bleu_config={"lowercase": False},
    chrf_config={"word_order": 2}  # chrF++
)

# Evaluate single language pair
result = evaluator.evaluate_model_on_pair(
    model=model,
    dataset=dataset,
    source_lang="eng",
    target_lang="fra",
    experiment_name="my_experiment"
)

print(f"BLEU: {result.corpus_metrics.bleu:.2f}")
print(f"chrF++: {result.corpus_metrics.chrf:.2f}")
```

### Using the Evaluation Script

The framework includes a complete evaluation script at `scripts/evaluate.py`. Run it directly:

```bash
python scripts/evaluate.py
```

This script evaluates multiple language pairs and saves results to the `results/` directory.

### Script Structure Example

```python
# scripts/evaluate.py example
from mt_benchmark.models.factory import ModelFactory
from mt_benchmark.datasets.flores import FloresDataset
from mt_benchmark.evaluation.pipeline import EvaluationPipeline

# Load models and dataset
model = ModelFactory.create_model('nllb_200_3.3B')
dataset = FloresDataset("FLORES_PLUS/devtest")

# Configure evaluation
evaluator = EvaluationPipeline(
    output_dir="results",
    batch_size=16,
    bleu_config={"lowercase": False},
    chrf_config={"word_order": 2}
)

# Define language pairs to evaluate
language_pairs = [
    ("eng", "fra"), ("fra", "eng"),
    ("eng", "swa"), ("swa", "eng"),
    ("fra", "swa"), ("swa", "fra"),
    ("eng", "kin"), ("kin", "eng")
]

# Evaluate all pairs
for source_lang, target_lang in language_pairs:
    result = evaluator.evaluate_model_on_pair(
        model=model,
        dataset=dataset,
        source_lang=source_lang,
        target_lang=target_lang,
        experiment_name="multilingual_eval"
    )
    print(f"{source_lang}→{target_lang}: BLEU={result.corpus_metrics.bleu:.2f}")
```

```
sample code to evaluate the models
python3 evaluate.py toucan_1.2B --single-pair --source-lang eng_Latn --target-lang kin_Latn
```

## Configuration

### Model Configuration

Models are configured in `mt_benchmark/config/models.yaml`:

```yaml
# Local HuggingFace Models
nllb_200_3.3B:
  model_name: "facebook/nllb-200-3.3B"
  model_class: "auto"
  torch_dtype: "float16"
  device_map: "auto"
  max_input_length: 1024
  generation_config:
    num_beams: 4
    max_new_tokens: 256
  lang_code_map:
    eng: "eng_Latn"
    swa: "swh_Latn"
    fra: "fra_Latn"

# API Models
gpt-5:
  model: "openai/gpt-5"
  rate_limit_delay: 0.05
  lm_kwargs:
    temperature: 1.0
    max_tokens: 20000

# Google Translate
google-translate:
  type: "google_cloud_translate"
  credentials_path: "credentials.json"
  batch_size: 16
```

### Environment Variables

For API models, set required environment variables:

```bash
# OpenAI models
export OPENAI_API_KEY="your_api_key_here"

# Google Cloud Translate
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

## Adding New HuggingFace Models

### 1. Add Model Configuration

Add your model to `mt_benchmark/config/models.yaml`:

```yaml
my_custom_model:
  model_name: "huggingface/my-translation-model"
  model_class: "auto"  # or specific class like "MT5ForConditionalGeneration"
  torch_dtype: "float16"
  device_map: "auto"
  max_input_length: 1024
  generation_config:
    num_beams: 5
    temperature: 0.6
    max_new_tokens: 256
  lang_code_map:  # Optional language code mapping
    eng: "en"
    fra: "fr"
```

### 2. Create Custom Model Class (if needed)

For models requiring special handling, create a new model class in `mt_benchmark/models/huggingface/`:

```python
# mt_benchmark/models/huggingface/custom_model.py
from mt_benchmark.models.huggingface.hf_model import HuggingFaceModel

class MyCustomModel(HuggingFaceModel):
    """Custom model implementation."""
    
    def _format_input(self, text: str, source_lang: str, target_lang: str) -> str:
        """Custom input formatting."""
        return f"Translate from {source_lang} to {target_lang}: {text}"
    
    def _postprocess_output(self, text: str) -> str:
        """Custom output postprocessing."""
        # Remove any special tokens or formatting
        return text.replace("<extra_token>", "").strip()
```

### 3. Register Model in Factory

Update the model factory in `mt_benchmark/models/factory.py`:

```python
# Add import
from mt_benchmark.models.huggingface.custom_model import MyCustomModel

class ModelFactory:
    _model_registry = {
        'toucan': ToucanModel,
        'nllb': NLLBModel,
        'seamless': SeamlessModel,
        'my_custom': MyCustomModel,  # Add your model
        # ... other models
    }
    
    @classmethod
    def _get_model_type(cls, model_id: str, model_name: str) -> str:
        """Add detection logic for your model."""
        if 'my_custom' in model_id.lower():
            return 'my_custom'
        # ... existing logic
```

### 4. Test Your Model

```python
# Test the new model
from mt_benchmark.models.factory import ModelFactory

model = ModelFactory.create_model('my_custom_model')
translations = model.translate(
    ["Hello, world!"], 
    source_lang="eng", 
    target_lang="fra"
)
print(translations[0])
```

## Results Structure

Evaluation results are organized in a structured directory:

```
results/
├── experiment_name/
│   ├── predictions/
│   │   ├── model_name_eng_fra.csv       # Individual predictions
│   │   ├── model_name_fra_eng.csv
│   │   └── ...
│   ├── metrics/
│   │   ├── model_name_eng_fra_metrics.json  # Detailed metrics
│   │   └── ...
│   └── experiment_name_summary.csv      # Experiment summary
```

### Results Files

- **Predictions CSV**: Source text, hypothesis, reference, sample ID
- **Metrics JSON**: BLEU, chrF++, timing information, configuration
- **Summary CSV**: Aggregate results across all language pairs

## API Models

### OpenAI Models

```python
# Configure in models.yaml
gpt-4o:
  model: "openai/gpt-4o"
  rate_limit_delay: 0.1
  lm_kwargs:
    temperature: 0.7
    max_tokens: 1000

# Use in code
model = ModelFactory.create_model('gpt-4o')
```

### Google Cloud Translate

1. **Setup credentials**:
```bash
# Download service account key from Google Cloud Console
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
```

2. **Configure and use**:
```python
model = ModelFactory.create_model('google-translate')
translations = model.translate(texts, "en", "fr")
```

## Advanced Usage

### Custom Evaluation Metrics

```python
from mt_benchmark.evaluation.pipeline import EvaluationPipeline

# Custom BLEU configuration
evaluator = EvaluationPipeline(
    bleu_config={"lowercase": True, "tokenize": "intl"},
    chrf_config={"word_order": 2, "char_order": 6}
)
```

### Batch Processing

```python
# Adjust batch size based on GPU memory
evaluator = EvaluationPipeline(batch_size=32)  # Larger batches

# Or use small batches for large models
evaluator = EvaluationPipeline(batch_size=4)
```

### Multiple Model Comparison

```python
models = ['toucan_base', 'nllb_200_3.3B', 'gpt-4o']

for model_id in models:
    model = ModelFactory.create_model(model_id)
    evaluator.evaluate_model_on_dataset(
        model=model,
        dataset=dataset,
        experiment_name=f"comparison_{model_id}"
    )
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use smaller models
2. **Language code errors**: Check `lang_code_map` in model config
3. **API rate limits**: Increase `rate_limit_delay` in model config
4. **Missing credentials**: Ensure API keys are set as environment variables

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{mt_benchmark_2025,
  title={MT-Benchmark: Machine Translation Benchmarking Framework},
  author={Alex Gichamba},
  year={2025},
  url={https://github.com/wakandaai/mt-benchmark}
}
```