# HGT Detection with Hidden Markov Models

Three-state HMM pipeline for detecting horizontally transferred genes in bacterial genomes, using composition and context features. This beta keeps optimized3 development code and the older demo paths.

## Overview
- States: Host, Ameliorated, Foreign
- Emissions: Gaussian (GC/GC3, RSCU, AA composition) + Bernoulli (GC-shift flag, mobility/effector keywords)
- Transition bias: Host -> Ameliorated -> Foreign with high self-transition
- Outputs: per-gene probabilities, foreign flag, and benchmark plots

## Installation
```bash
pip install -r requirements.txt
cd miniproject-beta
```

## Quick Start (demo)
```bash
python run_demo.py
```
Demo inputs live in `data/fake/demo`; outputs go to `results/fake/demo/latest`.

## Project Structure (beta)
```
miniproject-beta/
├── src/                         # Code (optimized2 + optimized3 variants)
├── data/
│   ├── raw/                     # Real CDS inputs
│   ├── processed/               # Cleaned real data (host/donor, truth labels)
│   └── fake/demo/               # Demo synthetic data
├── results/                     # Demo and real outputs
├── docs/documentation.md        # Technical overview
├── tests/                       # Unit/sanity tests
└── run_demo.py                  # Demo pipeline
```

## Real-data workflow (beta)
```bash
python data/scripts/host_cds_filtered.py
python data/scripts/donor_cds_filtered.py   # writes data/processed/hgt_truth.tsv

python src/main_optimized3.py train \
    --host data/processed/host/host_core_train.fasta \
    --foreign data/processed/donor/foreign_train.fasta \
    --output results/real/model_v3.pkl \
    --bw_iters 10 \
    --context_weight 0.5

python src/main_optimized3.py predict \
    --input data/processed/host_test_with_hgt.fasta \
    --model results/real/model_v3.pkl \
    --output results/real/predictions_v3.tsv \
    --foreign_threshold 0.5

python src/benchmark_optimized2.py \
    --predictions results/real/predictions_v3.tsv \
    --truth data/processed/hgt_truth.tsv \
    --output results/real \
    --fasta data/processed/host_test_with_hgt.fasta
```

Notes (beta):
- Train on original genomes only (no HGT tags); `main_optimized3.py` enforces this.
- `simulate` supports `--phylo_scenario {near,mid,far}` for biologically plausible donor GC targets.
