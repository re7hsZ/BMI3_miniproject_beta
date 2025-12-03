# feature_pipeline.py
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
# Build SYNONYMOUS_CODONS mapping
SYNONYMOUS_CODONS = {}
for codon, aa in GENETIC_CODE.items():
    SYNONYMOUS_CODONS.setdefault(aa, []).append(codon)

# FIXED CODON LIST (exclude stop codons '_')
CODON_LIST = sorted([c for c, a in GENETIC_CODE.items() if a != '_'])
# FIXED AA LIST (exclude stop)
AA_LIST = sorted([aa for aa in SYNONYMOUS_CODONS.keys() if aa != '_'])

def _iter_codons(sequence):
    seq = sequence.upper()
    for i in range(0, len(seq), 3):
        codon = seq[i:i+3]
        if len(codon) == 3:
            yield codon

def calculate_gc_content(sequence):
    if not sequence:
        return 0.0
    seq = sequence.upper()
    g = seq.count('G'); c = seq.count('C')
    return (g + c) / len(seq)  # return fraction 0-1

def calculate_gc3_content(sequence):
    if not sequence:
        return 0.0
    seq = sequence.upper()
    third = seq[2::3]
    if not third:
        return 0.0
    g = third.count('G'); c = third.count('C')
    return (g + c) / len(third)

def calculate_rscu_dict(sequence):
    """
    Return dict mapping codon->RSCU (for fixed CODON_LIST order).
    If an amino acid is absent, return 0.0 for its codons.
    """
    seq = sequence.upper()
    codons = [c for c in _iter_codons(seq)]
    counts = Counter(codons)
    rscu = {}
    for aa, syn_codons in SYNONYMOUS_CODONS.items():
        if aa == '_': 
            for c in syn_codons:
                rscu[c] = 0.0
            continue
        total = sum(counts[c] for c in syn_codons)
        n = len(syn_codons)
        if total == 0:
            for c in syn_codons:
                rscu[c] = 0.0
        else:
            expected = total / n
            for c in syn_codons:
                rscu[c] = (counts[c] / expected) if expected > 0 else 0.0
    # ensure all codons present (esp. non-synonym mapping order)
    for c in CODON_LIST:
        rscu.setdefault(c, 0.0)
    return rscu

def calculate_aa_composition(sequence):
    seq = sequence.upper()
    codons = [c for c in _iter_codons(seq)]
    aa_counts = Counter()
    total = 0
    for codon in codons:
        aa = GENETIC_CODE.get(codon)
        if aa and aa != '_':
            aa_counts[aa] += 1
            total += 1
    freq = {}
    for aa in AA_LIST:
        freq[aa] = (aa_counts[aa] / total) if total > 0 else 0.0
    return freq

def calculate_local_gc_shift(sequence, genomic_gc):
    seq_gc = calculate_gc_content(sequence)
    return abs(seq_gc - genomic_gc)

def extract_context_binary(header, sequence, genomic_gc=0.5):
    gc_shift = calculate_local_gc_shift(sequence, genomic_gc)
    is_high_gc_shift = 1 if gc_shift > 0.15 else 0

    mobility_keywords = ['transposase', 'integrase', 'conjugal', 'plasmid', 'phage', 'mobile']
    effector_keywords = ['ankyrin', 'f-box', 'u-box', 'sel1', 'legionella effector']

    header_lower = (header or "").lower()
    is_mobility = 1 if any(k in header_lower for k in mobility_keywords) else 0
    is_effector = 1 if any(k in header_lower for k in effector_keywords) else 0

    return np.array([is_high_gc_shift, is_mobility, is_effector], dtype=int)

# --- Backward Compatibility for main.py ---
def extract_features(sequence, header=None, genomic_gc=0.5):
    """
    Extracts feature vectors for a sequence.
    Compatible with original features.py interface.
    
    Returns:
        tuple: (composition_features, context_features)
        composition_features: [GC, GC3, RSCU..., AA...] (Continuous)
        context_features: [HighGCShift, Mobility, Effector] (Binary)
    """
    # Composition Features
    gc = calculate_gc_content(sequence)
    gc3 = calculate_gc3_content(sequence)
    rscu = calculate_rscu_dict(sequence)
    aa = calculate_aa_composition(sequence)
    
    # Flatten to vector (ensure sorted order as in original)
    rscu_vals = [rscu.get(c, 0.0) for c in CODON_LIST]
    aa_vals = [aa.get(a, 0.0) for a in AA_LIST]
    
    composition_features = [gc, gc3] + rscu_vals + aa_vals
    
    # Context Features
    context_features = extract_context_binary(header, sequence, genomic_gc).tolist()
    
    return composition_features, context_features


class FeaturePipeline:
    def __init__(self, n_pca_components=3):
        self.n_pca = n_pca_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_pca)
        # codon and aa lists are fixed above
        self.codon_list = CODON_LIST
        self.aa_list = AA_LIST
        self.fitted = False
        self.genomic_gc = None

    def _compose_continuous_vector(self, seq):
        # GC and GC3 as fractions 0-1
        gc = calculate_gc_content(seq)
        gc3 = calculate_gc3_content(seq)
        rscu = calculate_rscu_dict(seq)
        aa = calculate_aa_composition(seq)

        rscu_vals = [rscu[c] for c in self.codon_list]
        aa_vals = [aa[a] for a in self.aa_list]
        vec = np.array([gc, gc3] + rscu_vals + aa_vals, dtype=float)
        return vec

     ### MODIFIED: FeaturePipeline.fit - 增加输入检查、自动调整 PCA 维度、友好报错
    def fit(self, sequences, headers=None):
        """
        Fit the scaler and PCA using a list of sequences.
        headers optional (for genomic_gc calc if provided).
        """
        if sequences is None or len(sequences) == 0:
            raise ValueError("FeaturePipeline.fit: empty sequences list")

        if headers is None:
            headers = [None] * len(sequences)
        # estimate genomic GC as mean of gene-level GC (if not empty)
        gcs = [calculate_gc_content(s) for s in sequences if s]
        self.genomic_gc = (np.mean(gcs) if gcs else 0.5)

        cont_feats = []
        for s in sequences:
            cont_feats.append(self._compose_continuous_vector(s))
        cont_feats = np.vstack(cont_feats)  # shape (n_samples, n_features)

        # --- MODIFIED: 防护 PCA 维度不能超过样本数或特征数
        n_samples, n_features = cont_feats.shape
        max_allowed = min(n_samples, n_features)
        if self.n_pca > max_allowed:
            # shrink n_pca to feasible value and re-create PCA object
            new_n = max_allowed
            print(f"[FeaturePipeline] Warning: requested n_pca={self.n_pca} reduced to {new_n} (n_samples={n_samples}, n_features={n_features})")
            self.n_pca = new_n
            self.pca = PCA(n_components=self.n_pca)

        # Standardize then PCA
        cont_scaled = self.scaler.fit_transform(cont_feats)
        self.pca.fit(cont_scaled)
        self.fitted = True
        return self

     ### MODIFIED: FeaturePipeline.transform - 处理未 fit、空输入并保证返回 shape 正确
    def transform(self, sequences, headers=None):
        """
        Transform sequences -> (continuous_lowd_array, context_array)
        continuous_lowd_array: shape (n_samples, n_pca)
        context_array: shape (n_samples, 3)
        """
        if not self.fitted:
            raise RuntimeError("FeaturePipeline not fitted. Call fit(...) first.")
        if sequences is None or len(sequences) == 0:
            # 返回空数组，保持列数一致
            empty_cont = np.empty((0, self.pca.n_components_ if hasattr(self.pca, 'n_components_') else self.n_pca))
            empty_ctx = np.empty((0, 3), dtype=int)
            return empty_cont, empty_ctx

        if headers is None:
            headers = [None] * len(sequences)
        cont_feats = np.vstack([self._compose_continuous_vector(s) for s in sequences])
        cont_scaled = self.scaler.transform(cont_feats)
        cont_pca = self.pca.transform(cont_scaled)

        # context features use self.genomic_gc
        ctx = np.vstack([extract_context_binary(h, s, genomic_gc=self.genomic_gc) for s, h in zip(sequences, headers)])
        return cont_pca, ctx

    def fit_transform(self, sequences, headers=None):
        self.fit(sequences, headers=headers)
        return self.transform(sequences, headers=headers)

# Example quick test
if __name__ == "__main__":
    seqs = [
        "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG",  # example
        "ATGAAATTTGGGCCCAAATGCTAG"
    ]
    headers = [">gene1", ">gene2"]
    
    # Test backward compatibility
    print("Testing extract_features...")
    f1, c1 = extract_features(seqs[0], headers[0])
    print("Features len:", len(f1))
    print("Context:", c1)
    
    # Test Pipeline
    print("\nTesting FeaturePipeline...")
    fp = FeaturePipeline(n_pca_components=3)
    cont, ctx = fp.fit_transform(seqs, headers)
    print("continuous (PCA):", cont.shape); print(cont)
    print("context:", ctx.shape); print(ctx)
