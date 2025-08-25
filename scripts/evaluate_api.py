# Example usage of unified API models via DSPy
from mt_benchmark.models.factory import ModelFactory
from mt_benchmark.datasets.flores import FloresDataset
from mt_benchmark.evaluation.pipeline import EvaluationPipeline

# Load dataset
dataset = FloresDataset("FLORES_PLUS/devtest")

# Create any models
gpt_model = ModelFactory.create_model('gpt-5-nano')

# Create evaluation pipeline
evaluator = EvaluationPipeline(
    output_dir="results",
    batch_size=16
)

# Evaluate any model - same interface!
print("Evaluating google cloud translate")

gpt_results = evaluator.evaluate_model_on_pair(
    model=gpt_model,
    dataset=dataset,
    source_lang="fra",
    target_lang="kin",
    experiment_name="gpt5"
)

gpt_results = evaluator.evaluate_model_on_pair(
    model=gpt_model,
    dataset=dataset,
    source_lang="kin",
    target_lang="fra",
    experiment_name="gpt5"
)

gpt_results = evaluator.evaluate_model_on_pair(
    model=gpt_model,
    dataset=dataset,
    source_lang="fra",
    target_lang="swa",
    experiment_name="gpt5"
)

gpt_results = evaluator.evaluate_model_on_pair(
    model=gpt_model,
    dataset=dataset,
    source_lang="swa",
    target_lang="fra",
    experiment_name="gpt5"
)