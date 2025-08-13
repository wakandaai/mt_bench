# src/evaluation/pipeline.py
import time
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from ..models.base import BaseTranslationModel
from ..datasets.base import BaseDataset, LanguagePair
from .base import PredictionResult, EvaluationResult
from .metrics import MetricsCalculator

class EvaluationPipeline:
    """Main evaluation pipeline for translation models."""
    
    def __init__(self, 
                 output_dir: str = "results",
                 batch_size: int = 1,
                 bleu_config: Dict[str, Any] = None,
                 chrf_config: Dict[str, Any] = None):
        """Initialize evaluation pipeline.
        
        Args:
            output_dir: Directory to save results
            batch_size: Default batch size for predictions
            bleu_config: Configuration for BLEU metric
            chrf_config: Configuration for chrF++ metric
        """
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.metrics_calculator = MetricsCalculator(bleu_config, chrf_config)
    
    def evaluate_model_on_pair(self,
                              model: BaseTranslationModel,
                              dataset: BaseDataset,
                              source_lang: str,
                              target_lang: str,
                              experiment_name: str,
                              batch_size: Optional[int] = None) -> EvaluationResult:
        """Evaluate a model on a specific language pair.
        
        Args:
            model: Translation model to evaluate
            dataset: Dataset containing the language pair
            source_lang: Source language code
            target_lang: Target language code
            experiment_name: Name for this experiment
            batch_size: Batch size for predictions (overrides default)
            
        Returns:
            EvaluationResult object
        """
        print(f"Evaluating {model.get_model_info()['model_name']} on {source_lang}→{target_lang}")
        
        # Get samples for the language pair
        samples = dataset.get_language_pair(source_lang, target_lang)
        if not samples:
            raise ValueError(f"No samples found for {source_lang}→{target_lang}")
        
        # Generate predictions
        start_time = time.time()
        predictions = self._generate_predictions(model, samples, batch_size or self.batch_size)
        total_time = time.time() - start_time
        
        # Calculate corpus metrics
        hypotheses = [pred.hypothesis for pred in predictions]
        references = [pred.sample.target_text for pred in predictions]
        corpus_metrics = self.metrics_calculator.calculate_corpus_metrics(hypotheses, references)
        
        # Create evaluation result
        result = EvaluationResult(
            source_lang=source_lang,
            target_lang=target_lang,
            model_name=model.get_model_info()['model_name'].replace("/", "_"),
            predictions=predictions,
            corpus_metrics=corpus_metrics,
            experiment_name=experiment_name,
            total_time=total_time
        )
        
        # Save results
        self._save_evaluation_result(result)
        
        print(f"  BLEU: {corpus_metrics.bleu:.2f}, chrF++: {corpus_metrics.chrf:.2f}")
        return result
    
    def evaluate_model_on_dataset(self,
                                 model: BaseTranslationModel,
                                 dataset: BaseDataset,
                                 experiment_name: str,
                                 batch_size: Optional[int] = None) -> List[EvaluationResult]:
        """Evaluate a model on all language pairs in dataset.
        
        Args:
            model: Translation model to evaluate
            dataset: Dataset to evaluate on
            experiment_name: Name for this experiment
            batch_size: Batch size for predictions
            
        Returns:
            List of EvaluationResult objects
        """
        model_name = model.get_model_info()['model_name']
        language_pairs = dataset.list_language_pairs()
        
        print(f"Evaluating {model_name} on {len(language_pairs)} language pairs")
        
        results = []
        for pair in tqdm(language_pairs, desc="Language pairs"):
            try:
                result = self.evaluate_model_on_pair(
                    model, dataset, pair.source_lang, pair.target_lang, 
                    experiment_name, batch_size
                )
                results.append(result)
            except Exception as e:
                print(f"  Error evaluating {pair}: {e}")
                continue
    
        # Save summary
        self._save_experiment_summary(results, experiment_name)
        return results
    
    def _generate_predictions(self,
                            model: BaseTranslationModel,
                            samples: List,
                            batch_size: int) -> List[PredictionResult]:
        """Generate predictions for a list of samples."""
        predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(samples), batch_size), desc="Generating predictions", leave=False):
            batch_samples = samples[i:i + batch_size]
            
            # Extract texts and languages
            texts = [sample.source_text for sample in batch_samples]
            source_lang = batch_samples[0].source_lang
            target_lang = batch_samples[0].target_lang
            
            # Generate translations
            start_time = time.time()
            try:
                hypotheses = model.translate(texts, source_lang, target_lang)
                prediction_time = (time.time() - start_time) / len(batch_samples)
            except Exception as e:
                print(f"    Translation error: {e}")
                hypotheses = [""] * len(texts)  # Empty translations on error
                prediction_time = None
            
            # Create prediction results
            for sample, hypothesis in zip(batch_samples, hypotheses):
                predictions.append(PredictionResult(
                    sample=sample,
                    hypothesis=hypothesis,
                    prediction_time=prediction_time
                ))
        
        return predictions
    
    def _save_evaluation_result(self, result: EvaluationResult):
        """Save evaluation result to files."""
        # Create experiment directory
        exp_dir = self.output_dir / result.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions CSV
        predictions_dir = exp_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        
        csv_filename = f"{result.model_name}_{result.source_lang}_{result.target_lang}.csv"
        csv_path = predictions_dir / csv_filename
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["source", "hypothesis", "reference", "sample_id"])
            
            for pred in result.predictions:
                writer.writerow([
                    pred.sample.source_text,
                    pred.hypothesis,
                    pred.sample.target_text,
                    pred.sample.sample_id
                ])
        
        # Save metrics JSON
        metrics_dir = exp_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        metrics_filename = f"{result.model_name}_{result.source_lang}_{result.target_lang}_metrics.json"
        metrics_path = metrics_dir / metrics_filename
        
        metrics_data = {
            "model_name": result.model_name,
            "language_pair": f"{result.source_lang},{result.target_lang}",
            "bleu": result.corpus_metrics.bleu,
            "chrf": result.corpus_metrics.chrf,
            "num_samples": result.corpus_metrics.num_samples,
            "total_time": result.total_time,
            "avg_time_per_sample": result.total_time / len(result.predictions) if result.total_time else None,
            "metric_config": result.corpus_metrics.metric_config
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def _save_experiment_summary(self, results: List[EvaluationResult], experiment_name: str):
        """Save experiment summary CSV."""
        exp_dir = self.output_dir / experiment_name
        summary_path = exp_dir / f"{experiment_name}_summary.csv"
        
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "model_name", "language_pair", "source_lang", "target_lang", 
                "bleu", "chrf", "num_samples", "total_time"
            ])
            
            for result in results:
                writer.writerow([
                    result.model_name,
                    f"{result.source_lang}→{result.target_lang}",
                    result.source_lang,
                    result.target_lang,
                    result.corpus_metrics.bleu,
                    result.corpus_metrics.chrf,
                    result.corpus_metrics.num_samples,
                    result.total_time
                ])
        
        print(f"Experiment summary saved to {summary_path}")