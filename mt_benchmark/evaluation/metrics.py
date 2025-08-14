# mt_benchmark/evaluation/metrics.py

import sacrebleu
from typing import List, Dict, Any
from mt_benchmark.evaluation.base import CorpusMetrics

class MetricsCalculator:
    """Calculator for corpus-level translation metrics."""
    
    def __init__(self, bleu_config: Dict[str, Any] = None, chrf_config: Dict[str, Any] = None):
        """Initialize metrics calculator.
        
        Args:
            bleu_config: Configuration for BLEU metric (passed to sacrebleu.BLEU)
            chrf_config: Configuration for chrF++ metric (passed to sacrebleu.CHRF)
        """
        self.bleu_config = bleu_config or {}
        self.chrf_config = chrf_config or {}
        
        # Set up metric instances
        self.bleu_metric = sacrebleu.BLEU(**self.bleu_config)
        self.chrf_metric = sacrebleu.CHRF(**self.chrf_config)
    
    def calculate_corpus_metrics(self, hypotheses: List[str], references: List[str]) -> CorpusMetrics:
        """Calculate corpus-level BLEU and chrF++ scores.
        
        Args:
            hypotheses: List of translation hypotheses
            references: List of reference translations
            
        Returns:
            CorpusMetrics object with scores and configuration
        """
        if len(hypotheses) != len(references):
            raise ValueError(f"Number of hypotheses ({len(hypotheses)}) != references ({len(references)})")
        
        # Calculate BLEU
        bleu_score = self.bleu_metric.corpus_score(hypotheses, [references])
        
        # Calculate chrF++
        chrf_score = self.chrf_metric.corpus_score(hypotheses, [references])
        
        return CorpusMetrics(
            bleu=bleu_score.score,
            chrf=chrf_score.score,
            num_samples=len(hypotheses),
            metric_config={
                "bleu_config": self.bleu_config,
                "chrf_config": self.chrf_config
            }
        )
    
    @classmethod
    def get_default_configs(cls) -> Dict[str, Dict[str, Any]]:
        """Get commonly used metric configurations."""
        return {
            "bleu_default": {},
            "bleu_case_insensitive": {"lowercase": True},
            "bleu_detokenized": {"tokenize": "none"},
            "chrf_default": {},
            "chrf_plus": {"word_order": 2},  # chrF++
        }