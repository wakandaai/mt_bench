# Example usage of the FLORES dataset
from src.datasets.flores import FloresDataset

# Load dataset
dataset = FloresDataset("FLORES_PLUS/dev")

# Get dataset info
info = dataset.get_dataset_info()
print(f"Dataset: {info['dataset_name']}")
print(f"Languages: {info['unique_language_codes']}")
print(f"Total pairs: {info['total_pairs_with_reverse']}")
print(f"Available pairs: {info['available_pairs'][:5]}...")  # Show first 5

# Get specific language pair
eng_dik_samples = dataset.get_language_pair("eng", "dik")
print(f"\neng→dik samples: {len(eng_dik_samples)}")
print(f"First sample: {eng_dik_samples[0].source_text[:50]}... → {eng_dik_samples[0].target_text[:50]}...")

# Get reverse direction
dik_eng_samples = dataset.get_language_pair("dik", "eng")
print(f"dik→eng samples: {len(dik_eng_samples)}")

# Batch processing
print("\nBatch processing example:")
for i, batch in enumerate(dataset.get_language_pair_batch("eng", "dik", batch_size=2)):
    print(f"Batch {i+1}: {len(batch)} samples")
    if i >= 2:  # Show first 3 batches
        break

# Get all pairs for benchmarking
all_data = dataset.get_all_pairs_data()
print(f"\nTotal language pair combinations: {len(all_data)}")

# Check specific pair availability
print(f"Has eng→fra: {dataset.has_language_pair('eng', 'fra')}")
print(f"Has swc→yor: {dataset.has_language_pair('swc', 'yor')}")

# List all available pairs
print(f"\nFirst 10 available pairs:")
for pair in dataset.list_language_pairs()[:10]:
    print(f"  {pair}")