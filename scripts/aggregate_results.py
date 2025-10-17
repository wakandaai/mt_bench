#!/usr/bin/env python3
"""
Aggregate evaluation metrics from results folders into a single CSV.

Scans directory structure like:
  results/<experiment_name>/metrics/<model>_<src>_<tgt>_metrics.json
and writes a CSV with columns:
  model, lang_pair, source_lang, target_lang, BLEU, ChrF

Usage:
  python3 aggregate_results.py --results-dir ../results --out-file all_results.csv
"""
import argparse
import json
import csv
from pathlib import Path
from typing import Optional


def parse_metrics_file(path: Path) -> Optional[dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return None


def infer_langs_from_filename(stem: str):
    # Expected patterns: <model>_<src>_<tgt>_metrics or <model>_<src>_<tgt>_metrics.json
    parts = stem.split('_')
    if len(parts) >= 3:
        # last part may be 'metrics' so we take the three last meaningful parts
        # Try to find pattern where last part == 'metrics' and before are target and source
        if parts[-1] == 'metrics' and len(parts) >= 4:
            tgt = parts[-2]
            src = parts[-3]
            model = '_'.join(parts[:-3])
            return model, src, tgt
        else:
            # assume pattern model_src_tgt
            model = '_'.join(parts[:-2])
            src = parts[-2]
            tgt = parts[-1]
            return model, src, tgt
    return None, None, None


def extract_record(metrics_path: Path, metrics: dict, experiment_name: str):
    # Attempt to extract model and language info
    model = None
    src = None
    tgt = None

    # Try fields inside metrics JSON
    if isinstance(metrics, dict):
        model = metrics.get('model_name') or metrics.get('model')
        lp = metrics.get('language_pair')
        if lp and isinstance(lp, str) and (',' in lp or '→' in lp or '\t' in lp):
            # language_pair stored as 'src,tgt' or 'src→tgt'
            if ',' in lp:
                parts = [p.strip() for p in lp.split(',')]
                if len(parts) >= 2:
                    src, tgt = parts[0], parts[1]
            elif '→' in lp:
                parts = [p.strip() for p in lp.split('→')]
                if len(parts) >= 2:
                    src, tgt = parts[0], parts[1]
        # fallback: try keys explicitly
        if not src:
            src = metrics.get('source_lang')
        if not tgt:
            tgt = metrics.get('target_lang')

    # If still missing, infer from filename
    if not (src and tgt):
        stem = metrics_path.stem
        # remove common suffixes
        if stem.endswith('_metrics'):
            stem_core = stem[:-8]
        else:
            stem_core = stem
        inf_model, inf_src, inf_tgt = infer_langs_from_filename(stem_core)
        if not model and inf_model:
            model = inf_model
        if not src and inf_src:
            src = inf_src
        if not tgt and inf_tgt:
            tgt = inf_tgt

    # If model still missing, derive from parent folder (experiment_name)
    if not model:
        # try parent name (experiment dir -> may contain model subdir)
        model = metrics_path.parent.parent.name if metrics_path.parent.parent else experiment_name

    # Extract metrics values
    bleu = None
    chrf = None
    # common keys
    if isinstance(metrics, dict):
        bleu = metrics.get('bleu') or metrics.get('BLEU')
        chrf = metrics.get('chrf') or metrics.get('chrF') or metrics.get('ChrF')
        # some formats may nest corpus_metrics
        if (bleu is None or chrf is None) and 'corpus_metrics' in metrics:
            cm = metrics.get('corpus_metrics', {})
            bleu = bleu or cm.get('bleu')
            chrf = chrf or cm.get('chrf')

    # Build lang_pair string
    lang_pair = None
    if src and tgt:
        lang_pair = f"{src}→{tgt}"
    elif metrics and isinstance(metrics, dict) and 'language_pair' in metrics:
        lang_pair = metrics.get('language_pair')

    return {
        'experiment': experiment_name,
        'model': model,
        'lang_pair': lang_pair,
        'source_lang': src,
        'target_lang': tgt,
        'bleu': bleu,
        'chrf': chrf,
        # 'metrics_file': str(metrics_path)
    }


def aggregate(results_dir: Path, out_file: Path, filter_models=None):
    rows = []
    results_dir = results_dir.resolve()
    # Expect structure results/<experiment>/<predictions|metrics>/<files>
    for experiment_dir in sorted(results_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue
        metrics_dir = experiment_dir / 'metrics'
        if not metrics_dir.exists():
            # also allow nested model folders
            # search for any metrics json recursively
            metrics_files = list(experiment_dir.rglob('*_metrics.json'))
        else:
            metrics_files = list(metrics_dir.glob('*.json'))

        for mf in metrics_files:
            metrics = parse_metrics_file(mf)
            if metrics is None:
                continue
            rec = extract_record(mf, metrics, experiment_dir.name)
            if filter_models and rec['model']:
                # model names may contain slashes; check if any filter substring matches
                if not any(fm in rec['model'] or fm == rec['model'] for fm in filter_models):
                    continue
            rows.append(rec)

    # Write CSV
    fieldnames = ['experiment','model','lang_pair','source_lang','target_lang','bleu','chrf']
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_file}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate metrics JSON into a single CSV')
    parser.add_argument('--results-dir', default='../results', help='Path to results directory')
    parser.add_argument('--out-file', default='all_results.csv', help='Output CSV file path')
    parser.add_argument('--models', nargs='+', help='Optional list of model ids to include')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_file = Path(args.out_file)
    filter_models = args.models

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    aggregate(results_dir, out_file, filter_models)


if __name__ == '__main__':
    main()
