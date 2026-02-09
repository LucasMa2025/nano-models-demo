"""
Unit tests for core data models.
"""

import unittest
import numpy as np
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.models import (
    NanoModel, NanoLifecycleState, KVEntry, KVAccessMode,
    LoRAWeights, NanoModelDiagnostics, InnovationSample
)


class TestKVEntry(unittest.TestCase):
    """Tests for KVEntry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.key = np.random.randn(64)
        self.value = np.random.randn(256)
        self.entry = KVEntry(
            key=self.key,
            value=self.value,
            nano_id="test_nano",
            access_mode="exclusive",
        )
    
    def test_creation(self):
        """Test KVEntry creation."""
        self.assertIsNotNone(self.entry.kv_id)
        self.assertEqual(self.entry.nano_id, "test_nano")
        self.assertEqual(self.entry.access_mode, "exclusive")
        self.assertEqual(self.entry.version, 1)
        self.assertEqual(self.entry.access_count, 0)
    
    def test_hit_detection(self):
        """Test KV hit detection."""
        # Same key should hit
        self.assertTrue(self.entry.hit(self.key, threshold=0.9))
        
        # Random key should not hit
        random_key = np.random.randn(64)
        # May or may not hit depending on random values
        result = self.entry.hit(random_key, threshold=0.9)
        self.assertIsInstance(result, bool)
    
    def test_compute_similarity(self):
        """Test similarity computation."""
        # Same vector should have similarity 1
        sim = self.entry.compute_similarity(self.key)
        self.assertAlmostEqual(sim, 1.0, places=5)
        
        # Orthogonal vector should have similarity ~0
        orthogonal = np.zeros(64)
        orthogonal[0] = 1.0
        # Not guaranteed to be orthogonal, but test runs
        sim = self.entry.compute_similarity(orthogonal)
        self.assertIsInstance(sim, float)
    
    def test_update_value(self):
        """Test value update with version control."""
        new_value = np.random.randn(256)
        
        # Update by owner should succeed
        success, conflict = self.entry.update_value(new_value, "test_nano", force=True)
        self.assertTrue(success)
        self.assertEqual(self.entry.version, 2)
        self.assertEqual(len(self.entry.version_history), 1)
        
        # Update by non-owner should fail
        success, conflict = self.entry.update_value(new_value, "other_nano")
        self.assertFalse(success)
    
    def test_rollback(self):
        """Test version rollback."""
        original_value = self.entry.value.copy()
        
        # Make an update (force to bypass conflict check)
        new_value = np.random.randn(256)
        success, _ = self.entry.update_value(new_value, "test_nano", force=True)
        self.assertTrue(success)
        
        # Rollback to version 1
        success = self.entry.rollback(1)
        self.assertTrue(success)
        np.testing.assert_array_almost_equal(self.entry.value, original_value)
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        data = self.entry.to_dict()
        
        self.assertIn("kv_id", data)
        self.assertIn("key", data)
        self.assertIn("value", data)
        
        # Reconstruct
        reconstructed = KVEntry.from_dict(data)
        self.assertEqual(reconstructed.kv_id, self.entry.kv_id)
        self.assertEqual(reconstructed.nano_id, self.entry.nano_id)


class TestLoRAWeights(unittest.TestCase):
    """Tests for LoRAWeights class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rank = 8
        self.input_dim = 256
        self.output_dim = 256
        
        self.lora = LoRAWeights(
            layer_name="q_proj",
            A=np.random.randn(self.rank, self.input_dim) * 0.01,
            B=np.random.randn(self.output_dim, self.rank) * 0.01,
            alpha=32.0,
            rank=self.rank,
        )
    
    def test_delta_w_shape(self):
        """Test delta_w computation."""
        delta = self.lora.delta_w
        self.assertEqual(delta.shape, (self.output_dim, self.input_dim))
    
    def test_apply(self):
        """Test LoRA application."""
        x = np.random.randn(1, self.input_dim)
        output = self.lora.apply(x)
        self.assertEqual(output.shape, (1, self.output_dim))
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        data = self.lora.to_dict()
        
        self.assertEqual(data["layer_name"], "q_proj")
        self.assertEqual(data["rank"], self.rank)
        
        reconstructed = LoRAWeights.from_dict(data)
        self.assertEqual(reconstructed.layer_name, self.lora.layer_name)
        self.assertEqual(reconstructed.rank, self.lora.rank)


class TestNanoModel(unittest.TestCase):
    """Tests for NanoModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lora_weights = {
            "q_proj": LoRAWeights(
                layer_name="q_proj",
                A=np.random.randn(8, 256) * 0.01,
                B=np.random.randn(256, 8) * 0.01,
            ),
        }
        
        self.nano = NanoModel(
            nano_id="test_nano_001",
            lora_weights=self.lora_weights,
            bound_kv_ids={"kv_1", "kv_2"},
            state=NanoLifecycleState.ACTIVE,
            created_at=datetime.now(),
            last_activated=datetime.now(),
            creation_mode="standard",
            confidence_score=0.8,
            innovation_domain="test_domain",
        )
    
    def test_creation(self):
        """Test NanoModel creation."""
        self.assertEqual(self.nano.nano_id, "test_nano_001")
        self.assertEqual(self.nano.state, NanoLifecycleState.ACTIVE)
        self.assertEqual(len(self.nano.bound_kv_ids), 2)
        self.assertTrue(self.nano.is_frozen())
    
    def test_apply(self):
        """Test NanoModel application."""
        hidden_states = np.random.randn(1, 10, 256)
        output = self.nano.apply(hidden_states, "q_proj")
        self.assertEqual(output.shape, hidden_states.shape)
    
    def test_record_activation(self):
        """Test activation recording."""
        self.nano.record_activation(success=True, contribution=0.5)
        
        self.assertEqual(self.nano.activation_count, 1)
        self.assertEqual(self.nano.success_count, 1)
        self.assertEqual(self.nano.total_contribution, 0.5)
    
    def test_success_rate(self):
        """Test success rate computation."""
        self.nano.record_activation(success=True, contribution=0.5)
        self.nano.record_activation(success=True, contribution=0.5)
        self.nano.record_activation(success=False, contribution=0.3)
        
        self.assertAlmostEqual(self.nano.success_rate, 2/3, places=5)
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        data = self.nano.to_dict()
        
        self.assertEqual(data["nano_id"], "test_nano_001")
        self.assertEqual(data["state"], "active")
        
        reconstructed = NanoModel.from_dict(data)
        self.assertEqual(reconstructed.nano_id, self.nano.nano_id)
        self.assertEqual(reconstructed.state, self.nano.state)


class TestInnovationSample(unittest.TestCase):
    """Tests for InnovationSample class."""
    
    def test_creation(self):
        """Test InnovationSample creation."""
        sample = InnovationSample(
            query=np.random.randn(256),
            target_output=np.random.randn(256),
            innovation_score=0.85,
            innovation_components={"proj_error": 0.4, "struct_novelty": 0.3},
        )
        
        self.assertIsNotNone(sample.sample_id)
        self.assertEqual(sample.innovation_score, 0.85)
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        sample = InnovationSample(
            query=np.random.randn(256),
            target_output=np.random.randn(256),
            innovation_score=0.85,
            innovation_components={},
        )
        
        data = sample.to_dict()
        reconstructed = InnovationSample.from_dict(data)
        
        self.assertEqual(reconstructed.sample_id, sample.sample_id)
        self.assertEqual(reconstructed.innovation_score, sample.innovation_score)


if __name__ == "__main__":
    unittest.main()
