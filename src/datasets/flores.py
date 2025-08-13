# src/datasets/flores.py
import os
import csv
from pathlib import Path
from typing import List, Dict, Any, Set
from .base import BaseDataset, TranslationSample, LanguagePair

class FloresDataset(BaseDataset):
    """FLORES dataset loader for preprocessed CSV files."""
    
    def __init__(self, dataset_path: str):
        """Initialize FLORES dataset.
        
        Args:
            dataset_path: Path to directory containing CSV files
        """
        self.dataset_path = Path(dataset_path)
        self.data_cache: Dict[str, List[TranslationSample]] = {}
        self._discover_language_pairs()
        self._load_all_data()
    
    def _discover_language_pairs(self):
        """Discover available language pairs from CSV filenames."""
        self.available_files = []
        self.language_codes = set()
        
        for csv_file in self.dataset_path.glob("*.csv"):
            filename = csv_file.stem
            if "_" in filename:
                source_lang, target_lang = filename.split("_", 1)
                self.available_files.append((source_lang, target_lang, csv_file))
                self.language_codes.add(source_lang)
                self.language_codes.add(target_lang)
    
    def _load_all_data(self):
        """Load all CSV files into memory."""
        for source_lang, target_lang, csv_file in self.available_files:
            # Load forward direction
            forward_key = f"{source_lang}_{target_lang}"
            self.data_cache[forward_key] = self._load_csv_file(csv_file, source_lang, target_lang)
            
            # Load reverse direction (swap columns)
            reverse_key = f"{target_lang}_{source_lang}"
            self.data_cache[reverse_key] = self._load_csv_file(csv_file, target_lang, source_lang, reverse=True)
    
    def _load_csv_file(self, csv_file: Path, source_lang: str, target_lang: str, 
                      reverse: bool = False) -> List[TranslationSample]:
        """Load a single CSV file and return TranslationSample objects."""
        samples = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip header row
            
            for idx, row in enumerate(reader):
                if len(row) >= 2:
                    if reverse:
                        # Swap source and target for reverse direction
                        source_text = row[1].strip().strip('"')
                        target_text = row[0].strip().strip('"')
                    else:
                        source_text = row[0].strip().strip('"')
                        target_text = row[1].strip().strip('"')
                    
                    sample = TranslationSample(
                        source_text=source_text,
                        target_text=target_text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        sample_id=f"{csv_file.stem}_{idx}",
                        metadata={
                            "file": str(csv_file),
                            "row_index": idx,
                            "direction": "reverse" if reverse else "forward"
                        }
                    )
                    samples.append(sample)
        
        return samples
    
    def get_language_pair(self, source_lang: str, target_lang: str) -> List[TranslationSample]:
        """Get all samples for a specific language pair."""
        pair_key = f"{source_lang}_{target_lang}"
        
        if pair_key not in self.data_cache:
            raise ValueError(f"Language pair {source_lang}→{target_lang} not available. "
                           f"Available pairs: {list(self.data_cache.keys())}")
        
        return self.data_cache[pair_key]
    
    def list_language_pairs(self) -> List[LanguagePair]:
        """List all available language pairs including both directions."""
        pairs = []
        for pair_key in self.data_cache.keys():
            source_lang, target_lang = pair_key.split("_", 1)
            pairs.append(LanguagePair(source_lang, target_lang))
        return pairs
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset metadata and statistics."""
        total_samples = sum(len(samples) for samples in self.data_cache.values())
        unique_pairs = len(self.available_files)  # Original file pairs
        total_pairs_with_reverse = len(self.data_cache)
        
        # Calculate samples per language pair
        pair_stats = {}
        for pair_key, samples in self.data_cache.items():
            pair_stats[pair_key] = len(samples)
        
        return {
            "dataset_name": "FLORES",
            "dataset_path": str(self.dataset_path),
            "total_samples": total_samples,
            "unique_language_codes": sorted(list(self.language_codes)),
            "num_language_codes": len(self.language_codes),
            "original_pairs": unique_pairs,
            "total_pairs_with_reverse": total_pairs_with_reverse,
            "samples_per_pair": pair_stats,
            "available_pairs": [str(pair) for pair in self.list_language_pairs()]
        }
    
    def get_languages(self) -> Set[str]:
        """Get all available language codes."""
        return self.language_codes.copy()
    
    def has_language_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if a language pair is available."""
        pair_key = f"{source_lang}_{target_lang}"
        return pair_key in self.data_cache