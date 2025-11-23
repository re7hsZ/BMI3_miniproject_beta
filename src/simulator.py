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

def generate_random_gene(length_aa, gc_target=0.5):
    """
    Generates a random gene sequence with a target GC content.
    This is a simplified simulation where we try to pick codons that match the GC target.
    
    Args:
        length_aa (int): Length of the protein (number of codons).
        gc_target (float): Target GC content (0.0 to 1.0).
        
    Returns:
        str: DNA sequence.
    """
    sequence = []
    # Generate random amino acid sequence
    aa_seq = [random.choice(AMINO_ACIDS) for _ in range(length_aa)]
    
    for aa in aa_seq:
        codons = SYNONYMOUS_CODONS[aa]
        # Calculate GC content of each synonymous codon
        codon_gcs = []
        for c in codons:
            g = c.count('G')
            c_count = c.count('C')
            codon_gcs.append((g + c_count) / 3.0)
            
        # Select codon based on which one is closest to target GC?
        # Or probabilistically? Let's do a weighted choice to bias towards target GC.
        # Simple approach: Calculate weight = 1 / (|codon_gc - target_gc| + epsilon)
        weights = []
        for c_gc in codon_gcs:
            dist = abs(c_gc - gc_target)
            weights.append(1.0 / (dist + 0.1))
            
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

def simulate_context_features(label):
    """
    Simulate context features based on label.
    Label: 0=Host, 1=Ameliorated, 2=Foreign
    Features: [HighGCShift, Mobility, Effector]
    """
    # Probabilities of 1 for each feature given state
    # Host: Low prob
    # Foreign: High prob
    # Ameliorated: Intermediate
    
    probs = {
        0: [0.05, 0.01, 0.01], # Host
        1: [0.40, 0.30, 0.30], # Ameliorated
        2: [0.80, 0.60, 0.50]  # Foreign
    }
    
    p = probs.get(label, [0, 0, 0])
    features = [1.0 if random.random() < pi else 0.0 for pi in p]
    return features

def mutate_sequence(sequence, target_gc, mutation_rate=0.05):
    """
    Mutate sequence to shift GC content towards target.
    """
    seq_list = list(sequence)
    current_gc = (sequence.count('G') + sequence.count('C')) / len(sequence)
    
    # Determine if we need to increase or decrease GC
    increase_gc = target_gc > current_gc
    
    for i in range(len(seq_list)):
        if random.random() < mutation_rate:
            base = seq_list[i]
            if increase_gc:
                # Mutate AT -> GC
                if base in ['A', 'T']:
                    seq_list[i] = random.choice(['G', 'C'])
            else:
                # Mutate GC -> AT
                if base in ['G', 'C']:
                    seq_list[i] = random.choice(['A', 'T'])
                    
    return "".join(seq_list)

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
