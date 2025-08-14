# mt_benchmark/evaluation/base.py

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from mt_benchmark.datasets.base import TranslationSample

@dataclass
class PredictionResult:
    """A single prediction result."""
    sample: TranslationSample
    hypothesis: str
    prediction_time: Optional[float] = None
    
@dataclass
class CorpusMetrics:
    """Corpus-level evaluation metrics."""
    bleu: float
    chrf: float
    num_samples: int
    metric_config: Dict[str, Any]
    
@dataclass
class EvaluationResult:
    """Complete evaluation result for a language pair."""
    source_lang: str
    target_lang: str
    model_name: str
    predictions: List[PredictionResult]
    corpus_metrics: CorpusMetrics
    experiment_name: str
    total_time: Optional[float] = None