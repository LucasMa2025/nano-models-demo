"""
Unit tests for Innovation Detector.
"""

import unittest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.innovation_detector import InnovationDetector, InnovationDetectionResult
from core.models import KVEntry


class TestInnovationDetector(unittest.TestCase):
    """Tests for InnovationDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_dim = 256
        self.detector = InnovationDetector(
            hidden_dim=self.hidden_dim,
            num_subspaces=32,
            innovation_threshold=0.7,
            projection_weight=0.4,
            structure_weight=0.4,
            coverage_weight=0.2,
        )
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.hidden_dim, self.hidden_dim)
        self.assertEqual(self.detector.tau, 0.7)
        self.assertFalse(self.detector.subspace_initialized)
    
    def test_detect_without_initialization(self):
        """Test detection before subspace initialization."""
        hidden_states = np.random.randn(1, 10, self.hidden_dim)
        attention_weights = np.random.rand(1, 8, 10, 10)
        
        result = self.detector.detect(
            hidden_states=hidden_states,
            attention_weights=attention_weights,
            kv_store=[],
        )
        
        self.assertIsInstance(result, InnovationDetectionResult)
        self.assertIsInstance(result.is_innovation, bool)
        self.assertGreaterEqual(result.score, 0)
        self.assertLessEqual(result.score, 1)
    
    def test_update_semantic_subspaces(self):
        """Test semantic subspace learning."""
        # Generate training data
        for _ in range(150):
            hidden_states = np.random.randn(1, 10, self.hidden_dim)
            self.detector.update_semantic_subspaces(hidden_states, is_low_entropy=True)
        
        self.assertTrue(self.detector.subspace_initialized)
    
    def test_projection_error(self):
        """Test projection error computation."""
        # Initialize subspaces first
        for _ in range(150):
            hidden_states = np.random.randn(1, 10, self.hidden_dim)
            self.detector.update_semantic_subspaces(hidden_states, is_low_entropy=True)
        
        # Test with in-distribution data
        in_dist = np.random.randn(1, 10, self.hidden_dim)
        error_in = self.detector.compute_projection_error(in_dist)
        
        # Test with out-of-distribution data (larger magnitude)
        out_dist = np.random.randn(1, 10, self.hidden_dim) * 10
        error_out = self.detector.compute_projection_error(out_dist)
        
        self.assertIsInstance(error_in, float)
        self.assertIsInstance(error_out, float)
    
    def test_structure_novelty(self):
        """Test attention pattern anomaly detection."""
        # Add reference patterns
        for _ in range(10):
            pattern = np.random.rand(1, 8, 10, 10)
            self.detector.update_reference_patterns(pattern)
        
        self.assertEqual(len(self.detector.reference_patterns), 10)
        
        # Test novelty computation
        test_pattern = np.random.rand(1, 8, 10, 10)
        novelty = self.detector.compute_structure_novelty(test_pattern)
        
        self.assertIsInstance(novelty, float)
        self.assertGreaterEqual(novelty, 0)
        self.assertLessEqual(novelty, 1)
    
    def test_kv_coverage(self):
        """Test KV coverage computation."""
        # Create KV entries
        kv_entries = [
            KVEntry(
                key=np.random.randn(self.hidden_dim),
                value=np.random.randn(self.hidden_dim),
            )
            for _ in range(5)
        ]
        
        # Test with matching query
        query = kv_entries[0].key.copy()
        coverage = self.detector.compute_kv_coverage(query, kv_entries)
        self.assertAlmostEqual(coverage, 1.0, places=5)
        
        # Test with random query
        random_query = np.random.randn(self.hidden_dim)
        coverage = self.detector.compute_kv_coverage(random_query, kv_entries)
        self.assertIsInstance(coverage, float)
    
    def test_pattern_decay(self):
        """Test reference pattern decay mechanism."""
        # Add patterns
        for _ in range(10):
            pattern = np.random.rand(1, 8, 10, 10)
            self.detector.update_reference_patterns(pattern)
        
        initial_weights = self.detector.pattern_weights.copy()
        
        # Add more patterns to trigger decay
        for _ in range(5):
            pattern = np.random.rand(1, 8, 10, 10)
            self.detector.update_reference_patterns(pattern)
        
        # Check that older patterns have decayed
        # The first pattern should have lower weight
        self.assertLess(self.detector.pattern_weights[0], initial_weights[0])
    
    def test_statistics(self):
        """Test statistics collection."""
        # Run some detections
        for _ in range(10):
            hidden_states = np.random.randn(1, 10, self.hidden_dim)
            attention_weights = np.random.rand(1, 8, 10, 10)
            self.detector.detect(hidden_states, attention_weights, [])
        
        stats = self.detector.get_statistics()
        
        self.assertEqual(stats["detection_count"], 10)
        self.assertIn("innovation_rate", stats)
        self.assertIn("threshold", stats)
    
    def test_threshold_update(self):
        """Test threshold update."""
        self.detector.set_threshold(0.8)
        self.assertEqual(self.detector.tau, 0.8)
    
    def test_weight_update(self):
        """Test weight update."""
        self.detector.set_weights(0.5, 0.3, 0.2)
        self.assertAlmostEqual(self.detector.w1, 0.5, places=5)
        self.assertAlmostEqual(self.detector.w2, 0.3, places=5)
        self.assertAlmostEqual(self.detector.w3, 0.2, places=5)


class TestInnovationDetectionResult(unittest.TestCase):
    """Tests for InnovationDetectionResult class."""
    
    def test_creation(self):
        """Test result creation."""
        result = InnovationDetectionResult(
            is_innovation=True,
            score=0.85,
            projection_error=0.4,
            structure_novelty=0.3,
            kv_coverage=0.2,
        )
        
        self.assertTrue(result.is_innovation)
        self.assertEqual(result.score, 0.85)


if __name__ == "__main__":
    unittest.main()
