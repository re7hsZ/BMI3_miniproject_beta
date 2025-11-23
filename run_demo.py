"""
Complete HGT Detection Pipeline Demo

This script demonstrates the full workflow:
1. Generate synthetic training data
2. Simulate a genome with foreign islands
3. Train the 3-state HMM model  
4. Predict foreign genes
5. Generate benchmark visualizations
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Execute a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)
    return result

def main():
    """Run the complete HGT detection pipeline"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     HGT Detection with 3-State Hidden Markov Model          ║
║     Legionella pneumophila Foreign Gene Detection           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Generate training data using python directly
    print("\n[Step 1/5] Generating synthetic training data...")
    print("  - Creating host genes (GC=0.4, n=200)")
    print("  - Creating foreign genes (GC=0.6, n=200)")
    
    from src.simulator import generate_dataset
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio import SeqIO
    
    # Generate data
    host_train = generate_dataset(200, 0, 0.4, 0.0)
    foreign_train = generate_dataset(0, 200, 0.0, 0.6)
    
    # Save training data
    def save_genes(genes, filename):
        records = []
        for item in genes:
            # item is (gene_id, sequence, label)
            # We only need id and sequence for FASTA
            gid = item[0]
            seq = item[1]
            records.append(SeqRecord(Seq(seq), id=gid, description=""))
        SeqIO.write(records, filename, "fasta")
        print(f"  ✓ Saved {len(records)} sequences to {filename}")
    
    save_genes(host_train, 'data/train_host.fasta')
    save_genes(foreign_train, 'data/train_foreign.fasta')
    
    # Generate simulation source data  
    host_sim = generate_dataset(1000, 0, 0.4, 0.0)
    foreign_sim = generate_dataset(0, 100, 0.0, 0.6)
    save_genes(host_sim, 'data/sim_host.fasta')
    save_genes(foreign_sim, 'data/sim_foreign.fasta')
    
    # Step 2: Simulate genome
    run_command(
        "python src/main.py simulate --host data/sim_host.fasta --foreign data/sim_foreign.fasta --output data/test_genome.fasta --islands 10",
        "[Step 2/5] Simulating genome with foreign gene islands"
    )
    
    # Step 3: Train model
    run_command(
        "python src/main.py train --host data/train_host.fasta --foreign data/train_foreign.fasta --output data/model.pkl",
        "[Step 3/5] Training 3-state HMM model"
    )
    
    # Step 4: Predict
    run_command(
        "python src/main.py predict --input data/test_genome.fasta --model data/model.pkl --output results/predictions.tsv",
        "[Step 4/5] Predicting foreign genes"
    )
    
    # Step 5: Benchmark
    run_command(
        "python src/benchmark.py --predictions results/predictions.tsv --output results/benchmark --fasta data/test_genome.fasta",
        "[Step 5/5] Generating benchmark visualizations"
    )
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    PIPELINE COMPLETED                        ║
╚══════════════════════════════════════════════════════════════╝

Results saved to:
  - results/predictions.tsv         (Prediction file)
  - results/benchmark_*.png         (Visualizations)
  
Check the prediction file for per-gene HGT probabilities.
View the visualizations for model performance assessment.
    """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
