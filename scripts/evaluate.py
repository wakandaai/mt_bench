# Example usage of the evaluation pipeline
from mt_benchmark.models.factory import ModelFactory
from mt_benchmark.datasets.flores import FloresDataset
from mt_benchmark.evaluation.pipeline import EvaluationPipeline

# Load models and dataset
seamless_model = ModelFactory.create_model('nllb_200_3.3B')
dataset = FloresDataset("FLORES_PLUS/devtest")

# Configure metrics
bleu_config = {"lowercase": False}  # BLEU
chrf_config = {"word_order": 2}    # chrF++

# Create evaluation pipeline
batch_size = 16  # Process {batch_size} samples at once
evaluator = EvaluationPipeline(
    output_dir="results",
    batch_size=batch_size,
    bleu_config=bleu_config,
    chrf_config=chrf_config
)

# Evaluate single model on single language pair
result = evaluator.evaluate_model_on_pair(
    model=seamless_model,
    dataset=dataset,
    source_lang="fra",
    target_lang="eng",
    experiment_name="nllb_200_3.3B"
)

result = evaluator.evaluate_model_on_pair(
    model=seamless_model,
    dataset=dataset,
    source_lang="eng",
    target_lang="fra",
    experiment_name="nllb_200_3.3B"
)

result = evaluator.evaluate_model_on_pair(
    model=seamless_model,
    dataset=dataset,
    source_lang="eng",
    target_lang="swa",
    experiment_name="nllb_200_3.3B"
)

result = evaluator.evaluate_model_on_pair(
    model=seamless_model,
    dataset=dataset,
    source_lang="swa",
    target_lang="eng",
    experiment_name="nllb_200_3.3B"
)

result = evaluator.evaluate_model_on_pair(
    model=seamless_model,
    dataset=dataset,
    source_lang="fra",
    target_lang="swa",
    experiment_name="nllb_200_3.3B"
)

result = evaluator.evaluate_model_on_pair(
    model=seamless_model,
    dataset=dataset,
    source_lang="swa",
    target_lang="fra",
    experiment_name="nllb_200_3.3B"
)

result = evaluator.evaluate_model_on_pair(
    model=seamless_model,
    dataset=dataset,
    source_lang="kin",
    target_lang="fra",
    experiment_name="nllb_200_3.3B"
)

result = evaluator.evaluate_model_on_pair(
    model=seamless_model,
    dataset=dataset,
    source_lang="fra",
    target_lang="kin",
    experiment_name="nllb_200_3.3B"
)

result = evaluator.evaluate_model_on_pair(
    model=seamless_model,
    dataset=dataset,
    source_lang="eng",
    target_lang="kin",
    experiment_name="nllb_200_3.3B"
)

result = evaluator.evaluate_model_on_pair(
    model=seamless_model,
    dataset=dataset,
    source_lang="kin",
    target_lang="eng",
    experiment_name="nllb_200_3.3B"
)



print(f"BLEU: {result.corpus_metrics.bleu:.2f}")
print(f"chrF++: {result.corpus_metrics.chrf:.2f}")

# Evaluate single model on all language pairs
# all_results = evaluator.evaluate_model_on_dataset(
#     model=toucan_model,
#     dataset=dataset,
#     experiment_name="toucan_full_flores-devtest"
# )

# # Compare multiple models
# for model, model_name in [(toucan_model, "toucan"), (nllb_model, "nllb")]:
#     evaluator.evaluate_model_on_dataset(
#         model=model,
#         dataset=dataset,
#         experiment_name=f"model_comparison_{model_name}"
#     )

# Results will be saved in:
# results/
# ├── toucan_baseline/
# │   ├── predictions/
# │   │   └── UBC-NLP/toucan-base_eng_dik.csv
# │   └── metrics/
# │       └── UBC-NLP/toucan-base_eng_dik_metrics.json
# ├── toucan_full_eval/
# │   ├── predictions/
# │   │   ├── UBC-NLP/toucan-base_eng_dik.csv
# │   │   ├── UBC-NLP/toucan-base_eng_fra.csv
# │   │   └── ...
# │   ├── metrics/
# │   │   ├── UBC-NLP/toucan-base_eng_dik_metrics.json
# │   │   └── ...
# │   └── toucan_full_eval_summary.csv
# └── ...