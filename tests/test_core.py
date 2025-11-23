import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from features import calculate_gc_content, calculate_rscu
from hmm import HMM

class TestFeatures(unittest.TestCase):
    def test_gc_content(self):
        seq = "GCGC"
        self.assertEqual(calculate_gc_content(seq), 100.0)
        seq = "ATAT"
        self.assertEqual(calculate_gc_content(seq), 0.0)
        seq = "ATGC"
        self.assertEqual(calculate_gc_content(seq), 50.0)

    def test_rscu(self):
        # Test simple sequence: ATG (M) -> only 1 codon for M
        seq = "ATG"
        rscu = calculate_rscu(seq)
        self.assertAlmostEqual(rscu['ATG'], 1.0)
        
        # Test sequence with bias: TTT (F), TTC (F). If we have TTT TTT, RSCU(TTT)=2, RSCU(TTC)=0
        seq = "TTTTTT"
        rscu = calculate_rscu(seq)
        # Expected: 2 codons total for F. Observed TTT=2. Expected TTT = 1. RSCU = 2/1 = 2.
        self.assertAlmostEqual(rscu['TTT'], 2.0)
        self.assertAlmostEqual(rscu['TTC'], 0.0)

class TestHMM(unittest.TestCase):
    def test_initialization(self):
        hmm = HMM(n_states=3)
        self.assertEqual(hmm.n_states, 3)
        self.assertTrue(np.allclose(np.sum(hmm.A, axis=1), 1.0))
        self.assertTrue(np.allclose(np.sum(hmm.pi), 1.0))

    def test_viterbi_shape(self):
        hmm = HMM(n_states=3)
        # Fake data: 10 samples, 5 features
        data = np.random.rand(10, 5)
        hmm._init_emissions(data)
        
        path = hmm.viterbi(data)
        self.assertEqual(len(path), 10)
        self.assertTrue(np.all(path >= 0))
        self.assertTrue(np.all(path < 3))

if __name__ == '__main__':
    unittest.main()
