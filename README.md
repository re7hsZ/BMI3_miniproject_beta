# HGT Detection with Hidden Markov Models

A three-state Hidden Markov Model (HMM) framework for detecting horizontally transferred genes in bacterial genomes, with application to *Legionella pneumophila*.

## Overview

This project implements a novel HMM-based approach to identify foreign genes in bacterial genomes by modeling:
- **3 States**: Host-like, Ameliorated (partially assimilated), and Foreign-like genes
- **Dual-channel emissions**: Sequence composition (Gaussian) + contextual features (Bernoulli)  
- **Gene contiguity**: Transition probabilities favor Host ↔ Ameliorated ↔ Foreign to model gradual assimilation

## Key Features

✓ Original HMM implementation (Forward-Backward, Viterbi algorithms coded from scratch)  
✓ Rich feature engineering (GC/GC3, RSCU, amino acid usage, local GC shift, mobility markers)  
✓ Simulated genome generator for validation  
✓ Comprehensive benchmarking with visualizations  
✓ Generalizable to other bacterial genomes

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Navigate to project directory
cd mini_project
```

## Quick Start

Run the full pipeline demo:
```bash
python run_demo.py
```

This will:
1. Generate synthetic training data
2. Simulate a genome with foreign gene islands
3. Train the 3-state HMM
4. Predict foreign genes
5. Generate benchmark visualizations

## Project Structure

```
mini_project/
├── src/
│   ├── features.py      # Feature extraction (GC, RSCU, AA, context features)
│   ├── hmm.py           # 3-state HMM implementation
│   ├── simulator.py     # Genome simulation with ameliorated genes
│   ├── main.py          # CLI for train/predict/simulate
│   └── benchmark.py     # Evaluation and visualization
├── data/                # Generated data and results
├── docs/
│   └── documentation.md # Detailed technical documentation
├── tests/
│   └── test_core.py     # Unit tests
├── run_demo.py          # Full pipeline demo
└── README.md            # This file
```

## Usage

### Training a Model

```bash
python src/main.py train \
    --host data/host_genes.fasta \
    --foreign data/foreign_genes.fasta \
    --output models/hmm_model.pkl
```

### Predicting Foreign Genes

```bash
python src/main.py predict \
    --input data/genome.fasta \
    --model models/hmm_model.pkl \
    --output results/predictions.tsv
```

### Simulating Test Genomes

```bash
python src/main.py simulate \
    --host data/host_genes.fasta \
    --foreign data/foreign_genes.fasta \
    --output data/simulated_genome.fasta \
    --islands 10
```

### Benchmarking

```bash
python src/benchmark.py \
    --predictions results/predictions.tsv \
    --output results/benchmark \
    --fasta data/genome.fasta
```

## Model Architecture

### Three-State HMM

- **State 0 (Host)**: Native genes with host-like composition
- **State 1 (Ameliorated)**: Partially assimilated foreign genes  
- **State 2 (Foreign)**: Recently acquired genes

### Transition Topology

The model favors gradual assimilation:
- High self-transition probabilities (island contiguity)
- Preferred path: Host ↔ Ameliorated ↔ Foreign
- Low probability for direct Host ↔ Foreign jumps

### Dual-Channel Emissions

**Composition Channel (Gaussian)**:
- GC content
- GC3 (GC content at 3rd codon position)
- RSCU (Relative Synonymous Codon Usage)
- Amino acid composition

**Context Channel (Bernoulli)**:
- Local GC shift (deviation from genomic mean)
- Mobility markers (integrase, transposase keywords)
- Effector flags (ankyrin, F-box domain keywords)

Combined as: P(x|state) = P_gaussian(composition) × P_bernoulli(context)^w

## Output

### Prediction File Format

TSV file with columns:
- `GeneID`: Gene identifier
- `State`: Predicted state (0=Host, 1=Ameliorated, 2=Foreign)
- `Prob_Host`: Posterior probability of Host state
- `Prob_Ameliorated`: Posterior probability of Ameliorated state  
- `Prob_Foreign`: Posterior probability of Foreign state

### Benchmark Visualizations

- `confusion_matrix.png`: 3×3 confusion matrix
- `roc_curve.png`: ROC curve for Host vs Non-Host classification
- `heatmap.png`: Posterior probabilities along genome
- `barplot.png`: Top foreign gene candidates

## Customization

### Adjust HMM Parameters

Edit transition probabilities in `src/hmm.py`:
```python
self.A = np.array([
    [0.95, 0.04, 0.01],  # From Host
    [0.10, 0.80, 0.10],  # From Ameliorated
    [0.01, 0.04, 0.95]   # From Foreign
])
```

### Tune Context Weight

In `src/hmm.py`, adjust the balance between composition and context:
```python
self.context_weight = 1.0  # Higher = more weight on context features
```

## Algorithm Details

### Forward-Backward Algorithm

Computes posterior probabilities for each gene using log-space arithmetic for numerical stability.

### Viterbi Algorithm

Finds the most likely state sequence using dynamic programming.

### Supervised Initialization

- Host parameters estimated from conserved genes
- Foreign parameters from known horizontally transferred genes
- Ameliorated parameters initialized as intermediate values

## Performance

On simulated data with 1000 host genes and 10 foreign islands:
- Typical ROC AUC: 0.85-0.95
- Runtime: ~10-30 seconds for full pipeline

## Limitations

- Requires labeled training data (host and foreign genes)
- Context features depend on annotation quality
- Performance degrades for highly ameliorated genes

## Future Extensions

- Unsupervised training via Baum-Welch EM
- Integration with phylogenetic signals
- GPU acceleration for large genomes
- Web server for easy access

## References

1. Gogarten & Townsend (2005). Horizontal gene transfer, genome innovation and evolution. *Nature Rev Microbiol* 3:679-687.
2. Gomez-Valero & Buchrieser (2019). Intracellular parasitism and *Legionella* evolution. *Genes Immun* 20:394-402.
3. Juhas et al. (2009). Genomic islands: tools of bacterial HGT. *FEMS Microbiol Rev* 33:376-393.

## License

This project is developed for educational purposes as part of a bioinformatics course project.

## Contact

For questions or issues, please refer to the documentation in `docs/documentation.md`.
