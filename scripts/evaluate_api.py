# Example usage of unified API models via DSPy
from src.models.factory import ModelFactory
from src.datasets.flores import FloresDataset
from src.evaluation.pipeline import EvaluationPipeline

# Load dataset
dataset = FloresDataset("FLORES_PLUS/devtest")

# Create any API models using DSPy's unified interface
gpt5_model = ModelFactory.create_model('gpt-5-nano')

# Create evaluation pipeline
evaluator = EvaluationPipeline(
    output_dir="results",
    batch_size=1  # API models process one at a time
)

# Evaluate any model - same interface!
print("Evaluating GPT-5")
gpt5_results = evaluator.evaluate_model_on_pair(
    model=gpt5_model,
    dataset=dataset,
    source_lang="eng",
    target_lang="swa",
    experiment_name="gpt5"
)

# Same evaluation interface for everything!
print("Results:")
print(f"GPT-5 BLEU: {gpt5_results.corpus_metrics.bleu:.2f}")