"""
Unit tests for Conflict-Aware Fusion.
"""

import unittest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.fusion import ConflictAwareFusion, OutputFusion, FusionResult


class TestConflictAwareFusion(unittest.TestCase):
    """Tests for ConflictAwareFusion class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fusion = ConflictAwareFusion(
            direction_weight=0.7,
            base_conflict_threshold=0.3,
        )
        self.hidden_dim = 256
    
    def test_single_nano_fusion(self):
        """Test fusion with single Nano Model."""
        nano_outputs = {
            "nano_001": np.random.randn(self.hidden_dim),
        }
        kv_hit_scores = {"nano_001": 0.8}
        
        result = self.fusion.fuse(nano_outputs, kv_hit_scores)
        
        self.assertEqual(result.fusion_method, "single")
        self.assertEqual(result.fusion_weights["nano_001"], 1.0)
        self.assertFalse(result.conflict_detected)
    
    def test_weighted_average_fusion(self):
        """Test weighted average fusion (low conflict)."""
        # Create similar outputs (low conflict)
        base = np.random.randn(self.hidden_dim)
        nano_outputs = {
            "nano_001": base + np.random.randn(self.hidden_dim) * 0.1,
            "nano_002": base + np.random.randn(self.hidden_dim) * 0.1,
        }
        kv_hit_scores = {"nano_001": 0.8, "nano_002": 0.6}
        
        result = self.fusion.fuse(nano_outputs, kv_hit_scores)
        
        self.assertEqual(result.fusion_method, "weighted_average")
        self.assertAlmostEqual(sum(result.fusion_weights.values()), 1.0, places=5)
    
    def test_winner_takes_all_fusion(self):
        """Test winner-takes-all fusion (high conflict)."""
        # Create conflicting outputs
        nano_outputs = {
            "nano_001": np.random.randn(self.hidden_dim),
            "nano_002": -np.random.randn(self.hidden_dim),  # Opposite direction
        }
        kv_hit_scores = {"nano_001": 0.8, "nano_002": 0.6}
        
        # Force high conflict by using very different outputs
        nano_outputs["nano_001"] = np.ones(self.hidden_dim)
        nano_outputs["nano_002"] = -np.ones(self.hidden_dim)
        
        result = self.fusion.fuse(nano_outputs, kv_hit_scores)
        
        # Should detect conflict
        self.assertTrue(result.conflict_detected)
        self.assertEqual(result.fusion_method, "winner_takes_all")
        self.assertIsNotNone(result.winner_id)
    
    def test_conflict_detection(self):
        """Test conflict detection mechanism."""
        # Similar outputs (low conflict)
        similar_outputs = {
            "nano_001": np.ones(self.hidden_dim),
            "nano_002": np.ones(self.hidden_dim) * 1.1,
        }
        
        matrix, max_conflict = self.fusion._detect_conflicts(similar_outputs)
        self.assertLess(max_conflict, 0.3)
        
        # Different outputs (high conflict)
        different_outputs = {
            "nano_001": np.ones(self.hidden_dim),
            "nano_002": -np.ones(self.hidden_dim),
        }
        
        matrix, max_conflict = self.fusion._detect_conflicts(different_outputs)
        self.assertGreater(max_conflict, 0.5)
    
    def test_adaptive_threshold(self):
        """Test adaptive threshold computation."""
        # Initial threshold
        threshold = self.fusion._compute_adaptive_threshold()
        self.assertEqual(threshold, self.fusion.base_threshold)
        
        # Add conflict history
        for _ in range(20):
            self.fusion.conflict_history.append(0.5)  # High conflicts
        
        threshold = self.fusion._compute_adaptive_threshold()
        self.assertGreater(threshold, self.fusion.base_threshold)
    
    def test_statistics(self):
        """Test statistics collection."""
        # Run some fusions
        for _ in range(5):
            nano_outputs = {
                "nano_001": np.random.randn(self.hidden_dim),
                "nano_002": np.random.randn(self.hidden_dim),
            }
            kv_hit_scores = {"nano_001": 0.8, "nano_002": 0.6}
            self.fusion.fuse(nano_outputs, kv_hit_scores)
        
        stats = self.fusion.get_statistics()
        
        self.assertEqual(stats["fusion_count"], 5)
        self.assertIn("conflict_rate", stats)
        self.assertIn("avg_conflict", stats)
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        # Run some fusions
        nano_outputs = {"nano_001": np.random.randn(self.hidden_dim)}
        self.fusion.fuse(nano_outputs, {"nano_001": 0.8})
        
        self.fusion.reset_statistics()
        
        stats = self.fusion.get_statistics()
        self.assertEqual(stats["fusion_count"], 0)


class TestOutputFusion(unittest.TestCase):
    """Tests for OutputFusion class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fusion = OutputFusion()
        self.hidden_dim = 256
    
    def test_base_only_fusion(self):
        """Test fusion with base output only."""
        base_output = np.random.randn(self.hidden_dim)
        
        final, diagnostics = self.fusion.fuse(
            base_output=base_output,
            aga_output=None,
            nano_output=None,
        )
        
        np.testing.assert_array_equal(final, base_output)
        self.assertEqual(diagnostics["contributions"]["base"], 1.0)
    
    def test_aga_fusion(self):
        """Test fusion with AGA output."""
        base_output = np.random.randn(self.hidden_dim)
        aga_output = np.random.randn(self.hidden_dim)
        
        final, diagnostics = self.fusion.fuse(
            base_output=base_output,
            aga_output=aga_output,
            nano_output=None,
            alpha=0.5,
        )
        
        expected = base_output + 0.5 * (aga_output - base_output)
        np.testing.assert_array_almost_equal(final, expected)
        self.assertEqual(diagnostics["alpha"], 0.5)
    
    def test_nano_fusion(self):
        """Test fusion with Nano output."""
        base_output = np.random.randn(self.hidden_dim)
        nano_output = np.random.randn(self.hidden_dim)
        
        final, diagnostics = self.fusion.fuse(
            base_output=base_output,
            aga_output=None,
            nano_output=nano_output,
            gamma=0.3,
        )
        
        expected = base_output + 0.3 * nano_output
        np.testing.assert_array_almost_equal(final, expected)
        self.assertEqual(diagnostics["gamma"], 0.3)
    
    def test_full_fusion(self):
        """Test fusion with all components."""
        base_output = np.random.randn(self.hidden_dim)
        aga_output = np.random.randn(self.hidden_dim)
        nano_output = np.random.randn(self.hidden_dim)
        
        final, diagnostics = self.fusion.fuse(
            base_output=base_output,
            aga_output=aga_output,
            nano_output=nano_output,
            alpha=0.4,
            gamma=0.2,
        )
        
        expected = base_output + 0.4 * (aga_output - base_output) + 0.2 * nano_output
        np.testing.assert_array_almost_equal(final, expected)


if __name__ == "__main__":
    unittest.main()
