import numpy as np
from collections import Counter

# Standard genetic code dictionary
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

# Synonymous codons grouping
SYNONYMOUS_CODONS = {}
for codon, aa in GENETIC_CODE.items():
    if aa not in SYNONYMOUS_CODONS:
        SYNONYMOUS_CODONS[aa] = []
    SYNONYMOUS_CODONS[aa].append(codon)

def calculate_gc_content(sequence):
    """
    Calculates the GC content of a DNA sequence.
    
    Args:
        sequence (str): DNA sequence.
        
    Returns:
        float: GC content percentage (0-100).
    """
    if not sequence:
        return 0.0
    sequence = sequence.upper()
    g = sequence.count('G')
    c = sequence.count('C')
    return (g + c) / len(sequence) * 100.0

def calculate_gc3_content(sequence):
    """
    Calculates the GC content at the 3rd codon position.
    
    Args:
        sequence (str): Coding DNA sequence (length should be multiple of 3).
        
    Returns:
        float: GC3 content percentage (0-100).
    """
    if not sequence:
        return 0.0
    sequence = sequence.upper()
    # Extract 3rd positions
    third_positions = sequence[2::3]
    if not third_positions:
        return 0.0
    g = third_positions.count('G')
    c = third_positions.count('C')
    return (g + c) / len(third_positions) * 100.0

def calculate_rscu(sequence):
    """
    Calculates Relative Synonymous Codon Usage (RSCU) for a sequence.
    
    Args:
        sequence (str): Coding DNA sequence.
        
    Returns:
        dict: RSCU value for each codon.
    """
    sequence = sequence.upper()
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3) if len(sequence[i:i+3]) == 3]
    codon_counts = Counter(codons)
    
    rscu = {}
    for aa, synonymous_codons in SYNONYMOUS_CODONS.items():
        if aa == '_': continue # Skip stop codons for RSCU typically
        
        total_count = sum(codon_counts[c] for c in synonymous_codons)
        num_codons = len(synonymous_codons)
        
        if total_count == 0:
            # If amino acid is not present, RSCU is usually undefined or 0, 
            # but for HMM features we might want a neutral value or 0.
            # Let's set to 1/num_codons or just 0? 
            # Standard definition: if AA not present, RSCU is not calculable. 
            # However, for feature vectors, we need a value. 
            # Let's assign 0 for now, or maybe 1.0 (neutral).
            # Actually, if AA is missing, it provides no information about bias.
            # Let's stick to 0.0 but be careful in downstream analysis.
            for c in synonymous_codons:
                rscu[c] = 0.0
        else:
            for c in synonymous_codons:
                # RSCU_ij = X_ij / ( (1/n_i) * sum(X_ij) )
                observed = codon_counts[c]
                expected = total_count / num_codons
                rscu[c] = observed / expected
                
    return rscu

def calculate_amino_acid_composition(sequence):
    """
    Calculates Amino Acid composition.
    
    Args:
        sequence (str): Coding DNA sequence.
        
    Returns:
        dict: Frequency of each amino acid.
    """
    sequence = sequence.upper()
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3) if len(sequence[i:i+3]) == 3]
    
    aa_counts = Counter()
    total_aa = 0
    for codon in codons:
        if codon in GENETIC_CODE:
            aa = GENETIC_CODE[codon]
            if aa != '_': # Exclude stop codons from composition
                aa_counts[aa] += 1
                total_aa += 1
            
    aa_freq = {}
    all_aas = sorted([aa for aa in SYNONYMOUS_CODONS.keys() if aa != '_'])
    
    for aa in all_aas:
        if total_aa > 0:
            aa_freq[aa] = aa_counts[aa] / total_aa
        else:
            aa_freq[aa] = 0.0
            
    return aa_freq

def calculate_local_gc_shift(sequence, genomic_gc):
    """
    Calculates the absolute difference between sequence GC and genomic mean GC.
    """
    seq_gc = calculate_gc_content(sequence) / 100.0 # Normalize to 0-1
    return abs(seq_gc - genomic_gc)

def extract_context_features(header, sequence, genomic_gc=0.5):
    """
    Extracts context features:
    1. Local GC Shift (High shift > threshold? Or just the value? Let's use the value for Bernoulli? 
       Wait, Bernoulli is for binary features. 
       The proposal says "context channel encodes sparse indicators... local GC shift...".
       Maybe we bin the GC shift? Or treat it as a continuous feature in the Gaussian channel?
       "The composition channel is modeled by a multivariate Gaussian; the context channel modulates likelihood via a low dimensional Bernoulli component".
       So context features MUST be binary.
       Let's define "High GC Shift" as binary flag: |GC_gene - GC_genome| > 0.15 (heuristic).
    2. Mobility keywords in header.
    3. Effector keywords in header.
    
    Returns:
        list: [is_high_gc_shift, is_mobility, is_effector] (values 0 or 1)
    """
    # 1. High GC Shift
    gc_shift = calculate_local_gc_shift(sequence, genomic_gc)
    is_high_gc_shift = 1.0 if gc_shift > 0.15 else 0.0
    
    # 2. Mobility
    mobility_keywords = ['transposase', 'integrase', 'conjugal', 'plasmid', 'phage', 'mobile']
    is_mobility = 0.0
    if header:
        header_lower = header.lower()
        if any(k in header_lower for k in mobility_keywords):
            is_mobility = 1.0
            
    # 3. Effector (Eukaryotic-like)
    effector_keywords = ['ankyrin', 'f-box', 'u-box', 'sel1', 'legionella effector']
    is_effector = 0.0
    if header:
        header_lower = header.lower()
        if any(k in header_lower for k in effector_keywords):
            is_effector = 1.0
            
    return [is_high_gc_shift, is_mobility, is_effector]

def extract_features(sequence, header=None, genomic_gc=0.5):
    """
    Extracts feature vectors for a sequence.
    
    Returns:
        tuple: (composition_features, context_features)
        composition_features: [GC, GC3, RSCU..., AA...] (Continuous)
        context_features: [HighGCShift, Mobility, Effector] (Binary)
    """
    # Composition Features
    gc = calculate_gc_content(sequence)
    gc3 = calculate_gc3_content(sequence)
    rscu = calculate_rscu(sequence)
    aa = calculate_amino_acid_composition(sequence)
    
    # Flatten to vector
    rscu_vals = [rscu.get(c, 0.0) for c in sorted(rscu.keys())]
    aa_vals = [aa.get(a, 0.0) for a in sorted(aa.keys())]
    
    composition_features = [gc, gc3] + rscu_vals + aa_vals
    
    # Context Features
    context_features = extract_context_features(header, sequence, genomic_gc)
    
    return composition_features, context_features
