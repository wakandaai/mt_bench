#!/usr/bin/env python3
import argparse
import sys
from mt_benchmark.models.factory import ModelFactory
from mt_benchmark.datasets.flores import FloresDataset
from mt_benchmark.evaluation.pipeline import EvaluationPipeline

def main():
    parser = argparse.ArgumentParser(description='Machine Translation Evaluation Pipeline')
    parser.add_argument('model_id', help='Model ID (e.g., nllb_200_3.3B)')
    parser.add_argument('--dataset_id', default='FLORES_PLUS/devtest',
                        help='Dataset ID (e.g., FLORES_PLUS/devtest)')
    parser.add_argument('--single-pair', action='store_true', 
                       help='Evaluate on single language pair instead of full dataset')
    parser.add_argument('--source-lang', default='fra_Latn',
                       help='Source language for single pair evaluation (default: fra_Latn)')
    parser.add_argument('--target-lang', default='eng_Latn', 
                       help='Target language for single pair evaluation (default: eng_Latn)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for processing (default: 16)')
    
    args = parser.parse_args()
    
    try:
        # Load model and dataset
        print(f"Loading model: {args.model_id}")
        model = ModelFactory.create_model(args.model_id)
        
        print(f"Loading dataset: {args.dataset_id}")
        dataset = FloresDataset(args.dataset_id)
        
        # Configure metrics
        bleu_config = {"lowercase": False}  # BLEU
        chrf_config = {"word_order": 2}    # chrF++
        
        # Create evaluation pipeline
        evaluator = EvaluationPipeline(
            output_dir="results",
            batch_size=args.batch_size,
            bleu_config=bleu_config,
            chrf_config=chrf_config
        )
        
        if args.single_pair:
            # Evaluate single language pair
            print(f"Evaluating {args.source_lang} -> {args.target_lang}")
            result = evaluator.evaluate_model_on_pair(
                model=model,
                dataset=dataset,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                experiment_name=f"{args.model_id}"
            )
            print(f"BLEU: {result.corpus_metrics.bleu:.2f}")
            print(f"chrF++: {result.corpus_metrics.chrf:.2f}")
        else:
            # Evaluate on full dataset
            print("Evaluating on all language pairs in dataset")
            all_results = evaluator.evaluate_model_on_dataset(
                model=model,
                dataset=dataset,
                experiment_name=f"{args.model_id}"
            )
            print("Evaluation complete. Results saved in results/")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()