import random
from collections import Counter
from Bio import SeqIO

# Standard genetic code dictionary (same as feature extractors)
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
    SYNONYMOUS_CODONS.setdefault(aa, []).append(codon)

# Phylogenetic distance presets with real examples for biological sense.
# Host: Legionella pneumophila (reference)
PHYLO_SCENARIOS = {
    "near": {"gc_target": 0.42, "example": "Coxiella burnetii (Gammaproteobacteria, closer GC)"},
    "mid":  {"gc_target": 0.55, "example": "Vibrio cholerae / Yersinia (moderate distance)"},
    "far":  {"gc_target": 0.65, "example": "Pseudomonas / Mycobacterium (GC-rich, distant)"},
}


def generate_random_gene(length_aa, gc_target=0.5):
    sequence = []
    aa_seq = [random.choice(AMINO_ACIDS) for _ in range(length_aa)]
    for aa in aa_seq:
        codons = SYNONYMOUS_CODONS[aa]
        codon_gcs = [(c.count('G') + c.count('C')) / 3.0 for c in codons]
        weights = [1.0 / (abs(c_gc - gc_target) + 0.01) ** 2 for c_gc in codon_gcs]
        selected_codon = random.choices(codons, weights=weights, k=1)[0]
        sequence.append(selected_codon)
    stop_codons = ['TAA', 'TAG', 'TGA']
    sequence.append(random.choice(stop_codons))
    return "".join(sequence)


def generate_dataset(num_host, num_foreign, host_gc, foreign_gc, min_len=100, max_len=500):
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


Feature_probs = {
    'HighGCShift': {0: 0.05, 1: 0.40, 2: 0.80},
    'Mobility':    {0: 0.01, 1: 0.30, 2: 0.60},
    'Effector':    {0: 0.01, 1: 0.30, 2: 0.50 }
}


def simulate_context_features(label, probs=Feature_probs):
    p_values  = [
        probs['HighGCShift'][label],
        probs['Mobility'][label],
        probs['Effector'][label]
    ]
    return [1.0 if random.random() < p else 0.0 for p in p_values]


def mutate_sequence(sequence, target_gc, mutation_rate=0.05):
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    new_sequence = []
    current_gc = (sequence.count('G') + sequence.count('C')) / len(sequence)
    need_higher_gc = target_gc > current_gc
    for codon in codons:
        if len(codon) < 3 or GENETIC_CODE.get(codon, '_') == '_':
            new_sequence.append(codon)
            continue
        if random.random() < mutation_rate:
            aa = GENETIC_CODE[codon]
            candidate_codons = SYNONYMOUS_CODONS[aa]
            current_codon_gc = (codon.count('G') + codon.count('C'))
            if need_higher_gc:
                better_candidates = [c for c in candidate_codons if (c.count('G') + c.count('C')) > current_codon_gc]
            else:
                better_candidates = [c for c in candidate_codons if (c.count('G') + c.count('C')) < current_codon_gc]
            if better_candidates:
                new_sequence.append(random.choice(better_candidates))
            else:
                new_sequence.append(codon)
        else:
            new_sequence.append(codon)
    return "".join(new_sequence)


def create_genome_from_fasta(host_fasta, foreign_fasta, num_islands=5, avg_island_size=5, phylo_scenario="mid"):
    """
    Create simulated genome by inserting foreign/ameliorated islands.
    - Training should use original genomes (host/foreign FASTA as-is).
    - Testing uses this function to inject foreign segments.
    phylo_scenario controls GC target to emulate biological distances.
    """
    if phylo_scenario not in PHYLO_SCENARIOS:
        raise ValueError(f"phylo_scenario must be one of {list(PHYLO_SCENARIOS.keys())}")
    foreign_gc_target = PHYLO_SCENARIOS[phylo_scenario]["gc_target"]

    host_genes = []
    for record in SeqIO.parse(host_fasta, "fasta"):
        seq_str = str(record.seq).upper()
        if set(seq_str) - {'A', 'T', 'G', 'C', 'N'}:
            print(f"Warning: Invalid characters found in host gene {record.id}. Skipping.")
            continue
        host_genes.append((record.id, str(record.seq)))

    foreign_genes = []
    for record in SeqIO.parse(foreign_fasta, "fasta"):
        foreign_genes.append((record.id, str(record.seq)))

    if not host_genes or not foreign_genes:
        raise ValueError("Host or Foreign FASTA file is empty.")

    host_gcs = [(seq.count('G') + seq.count('C'))/len(seq) for _, seq in host_genes]
    host_mean_gc = sum(host_gcs) / len(host_gcs)

    genome = []
    for gid, seq in host_genes:
        context = simulate_context_features(0)
        genome.append((gid, seq, 0, context))

    foreign_pool = foreign_genes[:]
    random.shuffle(foreign_pool)

    for _ in range(num_islands):
        if not foreign_pool:
            break
        size = max(1, int(random.gauss(avg_island_size, 2)))
        is_ameliorated = random.random() < 0.3
        label = 1 if is_ameliorated else 2
        island = []
        for _ in range(size):
            if foreign_pool:
                gid, seq = foreign_pool.pop()
                target_gc = host_mean_gc if is_ameliorated else foreign_gc_target
                if is_ameliorated:
                    seq = mutate_sequence(seq, host_mean_gc, mutation_rate=0.1)
                else:
                    seq = mutate_sequence(seq, target_gc, mutation_rate=0.15)
                context = simulate_context_features(label)
                island.append((gid, seq, label, context))
        pos = random.randint(0, len(genome))
        genome[pos:pos] = island
    return genome
