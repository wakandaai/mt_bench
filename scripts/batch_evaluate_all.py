#!/usr/bin/env python3
"""
Batch evaluation script for all models and supported language pairs.
Automatically detects model language support and evaluates on matching FLORES pairs.
"""
import argparse
import sys
import os
import yaml
from pathlib import Path
from datetime import datetime
import json
import traceback
import torch
from tqdm import tqdm

from mt_benchmark.models.factory import ModelFactory
from mt_benchmark.datasets.flores import FloresDataset
from mt_benchmark.evaluation.pipeline import EvaluationPipeline


def load_models_config(config_path: str = None) -> dict:
    """Load models configuration from YAML file."""
    if config_path is None:
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / "mt_benchmark" / "config" / "models.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_supported_language_pairs(model, dataset, allowed_languages: set = None, verbose=False):
    """
    Find language pairs that both the model and dataset support.
    
    Args:
        model: Model instance
        dataset: Dataset instance
        verbose: Print detailed information
    
    Returns:
        List of (source_lang, target_lang) tuples
    """
    # Get model supported languages
    model_langs = model.get_supported_languages()
    model_lang_codes = set(model_langs.keys())
    
    # Get dataset languages
    dataset_lang_codes = dataset.get_languages()
    
    # Find intersection
    common_langs = model_lang_codes & dataset_lang_codes
    # If caller provided an allowed_languages set (e.g., 60 African FLORES codes), filter to that set
    if allowed_languages is not None:
        common_langs = common_langs & set(allowed_languages)
    
    if verbose:
        print(f"Model supports {len(model_lang_codes)} languages")
        print(f"Dataset has {len(dataset_lang_codes)} languages")
        print(f"Common languages: {len(common_langs)}")
    
    # Generate all possible pairs from common languages
    supported_pairs = []
    for source_lang in common_langs:
        for target_lang in common_langs:
            if source_lang != target_lang:
                # Check if dataset has this pair
                if dataset.has_language_pair(source_lang, target_lang):
                    supported_pairs.append((source_lang, target_lang))
    
    return supported_pairs


def evaluate_model_all_pairs(model_id: str, dataset_path: str, output_dir: str, 
                             batch_size: int = 4, max_pairs: int = None,
                             skip_existing: bool = True, verbose: bool = True):
    """
    Evaluate a single model on all supported language pairs.
    
    Args:
        model_id: Model ID from config
        dataset_path: Path to FLORES dataset
        output_dir: Output directory for results
        batch_size: Batch size for inference
        max_pairs: Maximum number of pairs to evaluate (for testing)
        skip_existing: Skip pairs that already have results
        verbose: Print progress information
    
    Returns:
        Dict with evaluation statistics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating model: {model_id}")
    print(f"{'='*80}\n")
    
    # Clear CUDA cache before loading model
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    stats = {
        'model_id': model_id,
        'start_time': datetime.now().isoformat(),
        'total_pairs': 0,
        'evaluated_pairs': 0,
        'skipped_pairs': 0,
        'failed_pairs': 0,
        'results': []
    }
    
    try:
        # Load model
        if verbose:
            print(f"Loading model: {model_id}...")
        model = ModelFactory.create_model(model_id)
        
        # Load dataset
        if verbose:
            print(f"Loading dataset: {dataset_path}...")
        dataset = FloresDataset(dataset_path)
        
        # Get supported language pairs
        if verbose:
            print("\nFinding supported language pairs...")
        supported_pairs = get_supported_language_pairs(model, dataset, verbose=verbose)
        
        if not supported_pairs:
            print(f"WARNING: No common language pairs found for {model_id}")
            stats['end_time'] = datetime.now().isoformat()
            return stats
        
        # Limit pairs if max_pairs specified
        if max_pairs and max_pairs < len(supported_pairs):
            print(f"\nLimiting to {max_pairs} pairs for testing")
            supported_pairs = supported_pairs[:max_pairs]
        
        stats['total_pairs'] = len(supported_pairs)
        
        print(f"\nFound {len(supported_pairs)} supported language pairs")
        print(f"Starting evaluation...\n")
        
        # Configure metrics
        bleu_config = {"lowercase": False}  # BLEU
        chrf_config = {"word_order": 2}    # chrF++
        
        # Create evaluation pipeline
        evaluator = EvaluationPipeline(
            output_dir=output_dir,
            batch_size=batch_size,
            bleu_config=bleu_config,
            chrf_config=chrf_config
        )
        
        # Evaluate each pair
        for source_lang, target_lang in tqdm(supported_pairs, desc=f"Evaluating {model_id}"):
            pair_key = f"{source_lang}-{target_lang}"
            
            # Check if results already exist
            if skip_existing:
                result_file = Path(output_dir) / f"{model_id}_{source_lang}_{target_lang}_results.json"
                if result_file.exists():
                    if verbose:
                        tqdm.write(f"Skipping {pair_key} (results exist)")
                    stats['skipped_pairs'] += 1
                    continue
            
            try:
                # Evaluate pair
                result = evaluator.evaluate_model_on_pair(
                    model=model,
                    dataset=dataset,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    experiment_name=f"{model_id}"
                )
                
                stats['evaluated_pairs'] += 1
                stats['results'].append({
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'bleu': result.corpus_metrics.bleu,
                    'chrf': result.corpus_metrics.chrf,
                    'status': 'success'
                })
                
                if verbose:
                    tqdm.write(f"✓ {pair_key}: BLEU={result.corpus_metrics.bleu:.2f}, chrF++={result.corpus_metrics.chrf:.2f}")
                
            except Exception as e:
                stats['failed_pairs'] += 1
                stats['results'].append({
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'status': 'failed',
                    'error': str(e)
                })
                tqdm.write(f"✗ {pair_key}: FAILED - {str(e)}")
                
                if verbose:
                    tqdm.write(f"Error details: {traceback.format_exc()}")
        
        stats['end_time'] = datetime.now().isoformat()
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Model {model_id} Summary:")
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Evaluated: {stats['evaluated_pairs']}")
        print(f"  Skipped: {stats['skipped_pairs']}")
        print(f"  Failed: {stats['failed_pairs']}")
        print(f"{'='*80}\n")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\nERROR loading or initializing model {model_id}: {e}")
        if verbose:
            print(traceback.format_exc())
        stats['end_time'] = datetime.now().isoformat()
        stats['error'] = str(e)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Batch evaluate all models on supported language pairs'
    )
    parser.add_argument(
        '--dataset-path',
        default='/ocean/projects/cis250145p/shared/datasets/FLORES_PLUS/devtest',
        help='Path to FLORES dataset'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='Specific models to evaluate (default: all models in config)'
    )
    parser.add_argument(
        '--exclude-models',
        nargs='+',
        default=[],
        help='Models to exclude from evaluation'
    )
    parser.add_argument(
        '--max-pairs',
        type=int,
        help='Maximum number of pairs per model (for testing)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip language pairs that already have results'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_false',
        dest='skip_existing',
        help='Re-evaluate even if results exist'
    )
    parser.add_argument(
        '--save-summary',
        action='store_true',
        default=True,
        help='Save evaluation summary to JSON'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    parser.add_argument(
        '--languages-file',
        help='Path to a file with one FLORES language code per line to restrict evaluation'
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        help='List of FLORES language codes to restrict evaluation (overrides languages-file)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models config
    models_config = load_models_config()
    
    # Determine which models to evaluate
    if args.models:
        models_to_evaluate = args.models
    else:
        # Evaluate all models except API models and excluded ones
        models_to_evaluate = [
            model_id for model_id in models_config.keys()
            if 'model_name' in models_config[model_id]  # HuggingFace models only
            and model_id not in args.exclude_models
        ]
    
    print(f"\n{'='*80}")
    print(f"Batch Evaluation Pipeline")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Models to evaluate: {len(models_to_evaluate)}")
    for model_id in models_to_evaluate:
        print(f"  - {model_id}")
    print(f"{'='*80}\n")
    
    # Evaluate each model
    all_stats = []
    for model_id in models_to_evaluate:
        try:
            # Prepare allowed languages set for this run
            allowed_languages = None
            if args.languages:
                allowed_languages = set(args.languages)
            elif args.languages_file:
                lf = Path(args.languages_file)
                if lf.exists():
                    allowed_languages = set([l.strip() for l in lf.read_text().splitlines() if l.strip()])

            stats = evaluate_model_all_pairs(
                model_id=model_id,
                dataset_path=args.dataset_path,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                max_pairs=args.max_pairs,
                skip_existing=args.skip_existing,
                verbose=args.verbose,
                # pass allowed languages to filter pairs
                allowed_languages=allowed_languages
            )
            all_stats.append(stats)
            
        except Exception as e:
            print(f"\nFATAL ERROR with model {model_id}: {e}")
            if args.verbose:
                print(traceback.format_exc())
            all_stats.append({
                'model_id': model_id,
                'status': 'fatal_error',
                'error': str(e)
            })
    
    # Save summary
    if args.save_summary:
        summary_path = output_dir / f"batch_evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    total_evaluated = sum(s.get('evaluated_pairs', 0) for s in all_stats)
    total_failed = sum(s.get('failed_pairs', 0) for s in all_stats)
    total_skipped = sum(s.get('skipped_pairs', 0) for s in all_stats)
    
    print(f"Models evaluated: {len(models_to_evaluate)}")
    print(f"Total pairs evaluated: {total_evaluated}")
    print(f"Total pairs failed: {total_failed}")
    print(f"Total pairs skipped: {total_skipped}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
