#!/usr/bin/env bash
source /ocean/projects/cis250145p/gichamba/miniconda3/etc/profile.d/conda.sh
conda activate mt
echo "Active environment: $CONDA_DEFAULT_ENV"
export OPENAI_API_KEY="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"
python3 scripts/evaluate_api.py