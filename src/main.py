import argparse
import os
import numpy as np
from Bio import SeqIO
from features_optimized2 import extract_features
from hmm_optimized2 import HMM
import pickle
from sklearn.preprocessing import StandardScaler

def load_sequences(fasta_file):
    """
    Load sequences from a FASTA file.
    Returns list of (header, sequence) tuples.
    """
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append((record.description, str(record.seq)))
    return sequences

def train_model(host_file, foreign_file, output_model):
    """
    Train HMM model using supervised initialization.
    """
    print(f"Loading host sequences from {host_file}...")
    host_data = load_sequences(host_file)
    print(f"Loading foreign sequences from {foreign_file}...")
    foreign_data = load_sequences(foreign_file)
    
    # Calculate genomic GC from host data for context features
    host_seqs = [s for _, s in host_data]
    host_gcs = [(s.count('G') + s.count('C'))/len(s) for s in host_seqs]
    genomic_gc = sum(host_gcs) / len(host_gcs)
    print(f"Estimated Genomic GC: {genomic_gc:.4f}")
    
    print("Extracting features...")
    # extract_features returns (comp, context)
    host_feats = [extract_features(s, h, genomic_gc) for h, s in host_data]
    foreign_feats = [extract_features(s, h, genomic_gc) for h, s in foreign_data]
    
    # Unpack
    host_comp = np.array([f[0] for f in host_feats])
    host_context = np.array([f[1] for f in host_feats])
    
    foreign_comp = np.array([f[0] for f in foreign_feats])
    foreign_context = np.array([f[1] for f in foreign_feats])
    
    print(f"Host composition shape: {host_comp.shape}")
    print(f"Host context shape: {host_context.shape}")
    
    # Scale composition features ONLY
    print("Scaling composition features...")
    scaler = StandardScaler()
    all_comp = np.vstack([host_comp, foreign_comp])
    scaler.fit(all_comp)
    
    host_comp_scaled = scaler.transform(host_comp)
    foreign_comp_scaled = scaler.transform(foreign_comp)
    
    print("Initializing HMM (3 States)...")
    hmm = HMM(n_states=3)
    hmm.train_supervised(host_comp_scaled, host_context, foreign_comp_scaled, foreign_context)
    
    print(f"Saving model to {output_model}...")
    with open(output_model, 'wb') as f:
        pickle.dump({'model': hmm, 'scaler': scaler, 'genomic_gc': genomic_gc}, f)
    print("Done.")

def simulate_genome(host_fasta, foreign_fasta, output_fasta, num_islands=5):
    """
    Simulate a genome using real sequences.
    """
    from simulator_optimized2 import create_genome_from_fasta
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
    
    print(f"Simulating genome using host={host_fasta} and foreign={foreign_fasta}...")
    # Returns (gene_id, sequence, label, context_features)
    genome = create_genome_from_fasta(host_fasta, foreign_fasta, num_islands=num_islands)
    
    records = []
    for gene_id, seq, label, context in genome:
        # Encode label in ID
        prefix = "host" if label == 0 else ("ameliorated" if label == 1 else "foreign")
        new_id = f"{prefix}_{gene_id}"
        
        # Encode context in description for feature extraction
        # context = [HighGCShift, Mobility, Effector]
        desc_parts = []
        if context[1] == 1.0:
            desc_parts.append("transposase")
        if context[2] == 1.0:
            desc_parts.append("ankyrin") # Example effector
            
        description = " ".join(desc_parts)
        
        records.append(SeqRecord(Seq(seq), id=new_id, description=description))
        
    print(f"Saving simulated genome to {output_fasta}...")
    SeqIO.write(records, output_fasta, "fasta")
    print("Done.")

def predict(input_file, model_file, output_file):
    """
    Predict foreign genes in input file using trained model.
    """
    print(f"Loading model from {model_file}...")
    with open(model_file, 'rb') as f:
        saved_data = pickle.load(f)
        hmm = saved_data['model']
        scaler = saved_data['scaler']
        genomic_gc = saved_data.get('genomic_gc', 0.5)
        
    print(f"Loading input sequences from {input_file}...")
    records = list(SeqIO.parse(input_file, "fasta"))
    # Need headers for context features
    data = [(r.description, str(r.seq)) for r in records]
    ids = [r.id for r in records]
    
    print("Extracting features...")
    features = [extract_features(s, h, genomic_gc) for h, s in data]
    
    comp_features = np.array([f[0] for f in features])
    context_features = np.array([f[1] for f in features])
    
    print("Scaling composition features...")
    comp_scaled = scaler.transform(comp_features)
    
    # Prepare observations list of tuples
    observations = []
    for i in range(len(comp_scaled)):
        observations.append((comp_scaled[i], context_features[i]))
    
    print("Running Viterbi decoding...")
    path = hmm.viterbi(observations)
    
    print("Calculating posterior probabilities...")
    posteriors = hmm.posterior(observations)
    
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        # Header for 3 states
        f.write("GeneID\tState\tProb_Host\tProb_Ameliorated\tProb_Foreign\n")
        for i, gene_id in enumerate(ids):
            state = path[i]
            probs = posteriors[i]
            # probs has 3 values now
            f.write(f"{gene_id}\t{state}\t{probs[0]:.4f}\t{probs[1]:.4f}\t{probs[2]:.4f}\n")
            
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="HGT Detection with HMM")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the HMM model')
    train_parser.add_argument('--host', required=True, help='FASTA file of host genes')
    train_parser.add_argument('--foreign', required=True, help='FASTA file of foreign genes')
    train_parser.add_argument('--output', required=True, help='Output model file (pickle)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict HGT genes')
    predict_parser.add_argument('--input', required=True, help='Input FASTA file to analyze')
    predict_parser.add_argument('--model', required=True, help='Trained model file (pickle)')
    predict_parser.add_argument('--output', required=True, help='Output results file (TSV)')
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Simulate genome from FASTA files')
    sim_parser.add_argument('--host', required=True, help='Host FASTA file')
    sim_parser.add_argument('--foreign', required=True, help='Foreign FASTA file')
    sim_parser.add_argument('--output', required=True, help='Output FASTA file')
    sim_parser.add_argument('--islands', type=int, default=5, help='Number of islands')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.host, args.foreign, args.output)
    elif args.command == 'predict':
        predict(args.input, args.model, args.output)
    elif args.command == 'simulate':
        simulate_genome(args.host, args.foreign, args.output, args.islands)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
