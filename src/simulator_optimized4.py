import random
from collections import Counter
from Bio import SeqIO

# Standard genetic code dictionary (same as features.py)
GENETIC_CODE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

AMINO_ACIDS = sorted(list(set(GENETIC_CODE.values()) - {'_'}))
SYNONYMOUS_CODONS = {}
for codon, aa in GENETIC_CODE.items():
    if aa not in SYNONYMOUS_CODONS:
        SYNONYMOUS_CODONS[aa] = []
    SYNONYMOUS_CODONS[aa].append(codon)

# Modification 1: The former weighting mechanisms of 1.0 / (dist + 0.1) is relatively weak, giving
# bad codons too much chance to be selected. Here we use random.choices with a stronger bias 
# towards codons that match the target GC content.

def generate_random_gene(length_aa, gc_target=0.5):
    sequence = []
    # Generate random amino acid sequence
    aa_seq = [random.choice(AMINO_ACIDS) for _ in range(length_aa)]
    
    for aa in aa_seq:
        codons = SYNONYMOUS_CODONS[aa]
        # Calculate GC content of each synonymous codon
        codon_gcs = [(c.count('G') + c.count('C')) / 3.0 for c in codons]
            
        ## Improved Weighting: Exponential bias strongly favors the closest match
        weights = [1.0 / (abs(c_gc - gc_target) + 0.01)**2 for c_gc in codon_gcs]
            
        selected_codon = random.choices(codons, weights=weights, k=1)[0]
        sequence.append(selected_codon)
        
    # Add a stop codon
    stop_codons = ['TAA', 'TAG', 'TGA']
    sequence.append(random.choice(stop_codons))
    
    return "".join(sequence)

def generate_dataset(num_host, num_foreign, host_gc, foreign_gc, min_len=100, max_len=500):
    """
    Generates a dataset of host and foreign genes.
    
    Args:
        num_host (int): Number of host genes.
        num_foreign (int): Number of foreign genes.
        host_gc (float): Target GC for host.
        foreign_gc (float): Target GC for foreign.
        
    Returns:
        list: List of tuples (gene_id, sequence, label). Label 0=Host, 1=Foreign.
    """
    data = []
    
    for i in range(num_host):
        length = random.randint(min_len, max_len)
        seq = generate_random_gene(length, host_gc)
        data.append((f"host_{i}", seq, 0))
        
    for i in range(num_foreign):
        length = random.randint(min_len, max_len)
        seq = generate_random_gene(length, foreign_gc)
        data.append((f"foreign_{i}", seq, 1))
        
    return data

# Modification 2: Move these probablities outside of the function for more flexible modification
# when different simulation parameters are desired.

Feature_probs = {
    'HighGCShift': {0: 0.05, 1: 0.40, 2: 0.80},
    'Mobility':    {0: 0.01, 1: 0.30, 2: 0.60},
    'Effector':    {0: 0.01, 1: 0.30, 2: 0.50 }
}

def simulate_context_features(label, probs=Feature_probs):
    """
    Simulate context features based on label.
    Label: 0=Host, 1=Ameliorated, 2=Foreign
    Features: [HighGCShift, Mobility, Effector]
    """
    # Probabilities of 1 for each feature given state
    # Host: Low prob
    # Foreign: High prob
    # Ameliorated: Intermediate
    
    p_values  = [
        probs['HighGCShift'][label],
        probs['Mobility'][label],
        probs['Effector'][label]
    ]

    features = [1.0 if random.random() < p else 0.0 for p in p_values]
    return features

# Modification 3: The former code flips bases randomly, changing the amino acid sequence and 
# destory the protein's functions. Here we implement a more biologically relevant mutation that uses
# synonymous codons (codons that code for the same amino acid) to preserve the proteins while shifting GC content.

def mutate_sequence(sequence, target_gc, mutation_rate=0.05):
    """
    Mutate sequence to shift GC content towards target using synonymous codons.
    """
    # Split sequence into codons
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    new_sequence = []

    current_gc = (sequence.count('G') + sequence.count('C')) / len(sequence)
    need_higher_gc = target_gc > current_gc

    for codon in codons:
        # Protect incomplete codons or stop codons
        if len(codon) < 3 or GENETIC_CODE.get(codon, '_') == '_':
            new_sequence.append(codon)
            continue

        # Probabilistically mutate codon
        if random.random() < mutation_rate:
            aa = GENETIC_CODE[codon]
            candidate_codons = SYNONYMOUS_CODONS[aa]

            # Filter candidate codons based on GC content
            if need_higher_gc:
                # Prefer codons with higher GC content
                current_codon_gc = (codon.count('G') + codon.count('C'))
                better_candidates = [c for c in candidate_codons if (c.count('G') + c.count('C')) > current_codon_gc]

                # Prefer codons with lower GC content
            else:
                current_codon_gc = (codon.count('G') + codon.count('C'))
                better_candidates = [c for c in candidate_codons if (c.count('G') + c.count('C')) < current_codon_gc]

            if better_candidates:
                new_sequence.append(random.choice(better_candidates))
            else:
                new_sequence.append(codon)
        else:
            new_sequence.append(codon)
    
    return "".join(new_sequence)

# Modification 4: The former function assumes that all input files are perfect, but if user
# input files are empty or with invalid DNA characters, it could lead to errors.
# Here we implement a little check when parsing fasta files.

def create_genome_from_fasta(host_fasta, foreign_fasta, num_islands=5, avg_island_size=5):
    """
    Create a simulated genome using sequences from FASTA files.
    Inserts foreign gene islands into host genome.
    Also simulates 'Ameliorated' genes.
    
    Returns:
        list of tuples: (gene_id, sequence, label, context_features)
        label: 0=Host, 1=Ameliorated, 2=Foreign
    """
    host_genes = []
    for record in SeqIO.parse(host_fasta, "fasta"):
        seq_str = str(record.seq).upper()
        # Basic validation of sequence (valid nucleotides)
        if set(seq_str) - {'A', 'T', 'G', 'C', 'N'}:
            print(f"Warning: Invalid characters found in host gene {record.id}. Skipping.")
            continue
        host_genes.append((record.id, str(record.seq)))
        
    foreign_genes = []
    for record in SeqIO.parse(foreign_fasta, "fasta"):
        foreign_genes.append((record.id, str(record.seq)))
        
    if not host_genes or not foreign_genes:
        raise ValueError("Host or Foreign FASTA file is empty.")
        
    # Calculate Host Mean GC
    host_gcs = [(seq.count('G') + seq.count('C'))/len(seq) for _, seq in host_genes]
    host_mean_gc = sum(host_gcs) / len(host_gcs)
        
    # Start with all host genes
    # Label 0 = Host
    genome = []
    for gid, seq in host_genes:
        context = simulate_context_features(0)
        genome.append((gid, seq, 0, context))
    
    # Insert islands
    foreign_pool = foreign_genes[:]
    random.shuffle(foreign_pool)
    
    for _ in range(num_islands):
        if not foreign_pool:
            break
            
        size = max(1, int(random.gauss(avg_island_size, 2)))
        
        # Decide if this island is "Ameliorated" or "Foreign"
        # Let's say 30% islands are ameliorated
        is_ameliorated = random.random() < 0.3
        label = 1 if is_ameliorated else 2
        
        island = []
        for _ in range(size):
            if foreign_pool:
                gid, seq = foreign_pool.pop()
                
                if is_ameliorated:
                    # Mutate sequence towards host GC
                    seq = mutate_sequence(seq, host_mean_gc, mutation_rate=0.1)
                    
                context = simulate_context_features(label)
                island.append((gid, seq, label, context))
        
        # Insert at random position
        pos = random.randint(0, len(genome))
        genome[pos:pos] = island
        
    return genome
