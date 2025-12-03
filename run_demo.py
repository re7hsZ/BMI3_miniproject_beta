"""
Complete HGT Detection Pipeline Demo (synthetic only).

This script keeps synthetic inputs under data/fake/demo and writes
all outputs to results/fake/demo/latest so they stay separate from
real data and results.
"""

import sys
import subprocess
from pathlib import Path

FAKE_DATA_DIR = Path("data/fake/demo")
FAKE_RESULTS_DIR = Path("results/fake/demo/latest")
BENCHMARK_SCRIPT = Path("src/benchmark_optimized2.py")


def run_command(cmd, description):
    """Execute a shell command and handle errors."""
    print(f"\n{'=' * 60}")
    print(description)
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)
    return result


def prepare_demo_dirs():
    """Ensure synthetic data/results folders exist and are clean."""
    FAKE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # wipe previous demo outputs to avoid mixing runs
    for pattern in ("*.png", "*.tsv", "*.pkl"):
        for path in FAKE_RESULTS_DIR.glob(pattern):
            path.unlink()


def main():
    """Run the complete HGT detection pipeline on synthetic data."""
    print("\nHGT detection demo (synthetic) starting…")
    prepare_demo_dirs()

    # Step 1: Generate training data using the optimized simulator
    print("\n[Step 1/5] Generating synthetic training data...")
    print("  - Creating host genes (GC=0.4, n=200)")
    print("  - Creating foreign genes (GC=0.6, n=200)")

    from src.simulator_optimized2 import generate_dataset
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio import SeqIO

    # Generate data
    host_train = generate_dataset(200, 0, 0.4, 0.0)
    foreign_train = generate_dataset(0, 200, 0.0, 0.6)

    # Save training data
    def save_genes(genes, filename):
        records = []
        for gid, seq, _ in genes:
            records.append(SeqRecord(Seq(seq), id=gid, description=""))
        SeqIO.write(records, filename, "fasta")
        print(f"  Saved {len(records)} sequences to {filename}")

    train_host = FAKE_DATA_DIR / "train_host.fasta"
    train_foreign = FAKE_DATA_DIR / "train_foreign.fasta"
    save_genes(host_train, train_host)
    save_genes(foreign_train, train_foreign)

    # Generate simulation source data
    host_sim = generate_dataset(1000, 0, 0.4, 0.0)
    foreign_sim = generate_dataset(0, 100, 0.0, 0.6)
    sim_host = FAKE_DATA_DIR / "sim_host.fasta"
    sim_foreign = FAKE_DATA_DIR / "sim_foreign.fasta"
    save_genes(host_sim, sim_host)
    save_genes(foreign_sim, sim_foreign)

    test_genome = FAKE_DATA_DIR / "test_genome.fasta"
    model_path = FAKE_RESULTS_DIR / "model.pkl"
    predictions_path = FAKE_RESULTS_DIR / "predictions.tsv"

    # Step 2: Simulate genome
    run_command(
        f"python src/main.py simulate --host {sim_host} --foreign {sim_foreign} --output {test_genome} --islands 10",
        "[Step 2/5] Simulating genome with foreign gene islands",
    )

    # Step 3: Train model
    run_command(
        f"python src/main.py train --host {train_host} --foreign {train_foreign} --output {model_path}",
        "[Step 3/5] Training 3-state HMM model",
    )

    # Step 4: Predict
    run_command(
        f"python src/main.py predict --input {test_genome} --model {model_path} --output {predictions_path}",
        "[Step 4/5] Predicting foreign genes",
    )

    # Step 5: Benchmark
    run_command(
        f"python {BENCHMARK_SCRIPT} --predictions {predictions_path} --output {FAKE_RESULTS_DIR} --fasta {test_genome}",
        "[Step 5/5] Generating benchmark visualizations",
    )

    print("\nPipeline completed.")
    print(f"  Synthetic inputs: {FAKE_DATA_DIR}")
    print(f"  Synthetic outputs: {FAKE_RESULTS_DIR}")
    print("Check the prediction file for per-gene HGT probabilities.")
    print("View benchmark PNGs for model performance assessment.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
