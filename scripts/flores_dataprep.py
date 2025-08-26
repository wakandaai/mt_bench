#!/usr/bin/env python3
"""
FLORES Dataset Preparation Script

Downloads FLORES+ dataset and creates parallel CSV files for MT-Benchmark.

Usage: 
    python scripts/flores_dataprep.py
    python scripts/flores_dataprep.py --output_dir my_data --languages eng_Latn fra_Latn
"""

import argparse
import pandas as pd
import os
from datasets import load_dataset
from itertools import combinations

# Default configuration
DEFAULT_LANGUAGES = ["eng_Latn", "fra_Latn", "swh_Latn", "kin_Latn", "fon_Latn", "dik_Latn"]
DEFAULT_SPLITS = ["dev", "devtest"]
DEFAULT_OUTPUT_DIR = "FLORES_PLUS"

# Language code mapping
LANG_MAPPING = {
    "eng_Latn": "eng",
    "fra_Latn": "fra", 
    "swh_Latn": "swa",
    "kin_Latn": "kin",
    "fon_Latn": "fon",
    "dik_Latn": "dik"
}

def download_languages(languages, splits, output_dir):
    """Download individual language files."""
    print("Downloading FLORES+ dataset...")
    os.makedirs(output_dir, exist_ok=True)
    
    for lang in languages:
        for split in splits:
            output_file = f"{output_dir}/{lang}_{split}.csv"
            
            if os.path.exists(output_file):
                print(f"✓ {lang}_{split}.csv already exists")
                continue
                
            try:
                dataset = load_dataset("openlanguagedata/flores_plus", lang, split=split)
                # Convert to pandas directly (no ['test'] key needed)
                df = dataset.to_pandas()
                df.to_csv(output_file, index=False)
                print(f"✓ Saved {lang} {split}: {len(df)} rows")
            except Exception as e:
                print(f"✗ Error downloading {lang} {split}: {e}")

def create_parallel_files(languages, splits, output_dir):
    """Create parallel CSV files for all language pairs."""
    print("\nCreating parallel files...")
    
    for split in splits:
        split_dir = f"{output_dir}/{split}"
        os.makedirs(split_dir, exist_ok=True)
        
        # Load all datasets for this split
        datasets = {}
        for lang in languages:
            csv_file = f"{output_dir}/{lang}_{split}.csv"
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file).sort_values('id').reset_index(drop=True)
                datasets[lang] = df
        
        # Create language pairs
        for lang1, lang2 in combinations(languages, 2):
            if lang1 in datasets and lang2 in datasets:
                df1, df2 = datasets[lang1], datasets[lang2]
                
                # Create parallel CSV
                code1, code2 = LANG_MAPPING[lang1], LANG_MAPPING[lang2]
                parallel_df = pd.DataFrame({
                    code1: df1['text'],
                    code2: df2['text']
                })
                
                output_file = f"{split_dir}/{code1}_{code2}.csv"
                parallel_df.to_csv(output_file, index=False, encoding='utf-8')
                print(f"✓ Created {code1}_{code2}.csv: {len(parallel_df)} pairs")

def main():
    parser = argparse.ArgumentParser(description="Prepare FLORES+ dataset for MT-Benchmark")
    
    parser.add_argument(
        "--output_dir", 
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_LANGUAGES,
        help=f"Languages to download (default: {DEFAULT_LANGUAGES})"
    )
    
    parser.add_argument(
        "--splits",
        nargs="+", 
        default=DEFAULT_SPLITS,
        help=f"Dataset splits (default: {DEFAULT_SPLITS})"
    )
    
    args = parser.parse_args()
    
    print(f"Languages: {args.languages}")
    print(f"Splits: {args.splits}")
    print(f"Output: {args.output_dir}")
    
    download_languages(args.languages, args.splits, args.output_dir)
    create_parallel_files(args.languages, args.splits, args.output_dir)
    print(f"\nDone! Dataset ready in {args.output_dir}/")

if __name__ == "__main__":
    main()