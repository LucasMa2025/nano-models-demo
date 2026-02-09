"""
Unit tests for Nano Model Factory.
"""

import unittest
import numpy as np
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.factory import NanoModelFactory, CreationMode, CreationConfig
from core.registry import NanoRegistry
from core.models import InnovationSample, NanoLifecycleState


class TestCreationConfig(unittest.TestCase):
    """Tests for CreationConfig class."""
    
    def test_emergency_config(self):
        """Test emergency mode configuration."""
        config = CreationConfig.for_mode(CreationMode.EMERGENCY)
        
        self.assertEqual(config.mode, CreationMode.EMERGENCY)
        self.assertEqual(config.min_samples, 1)
        self.assertEqual(config.max_samples, 5)
        self.assertEqual(config.lora_rank, 4)
        self.assertEqual(config.initial_state, NanoLifecycleState.TRIAL)
    
    def test_standard_config(self):
        """Test standard mode configuration."""
        config = CreationConfig.for_mode(CreationMode.STANDARD)
        
        self.assertEqual(config.mode, CreationMode.STANDARD)
        self.assertEqual(config.min_samples, 20)
        self.assertEqual(config.lora_rank, 8)
        self.assertEqual(config.initial_state, NanoLifecycleState.ACTIVE)


class TestNanoModelFactory(unittest.TestCase):
    """Tests for NanoModelFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = NanoRegistry()
        self.factory = NanoModelFactory(
            registry=self.registry,
            hidden_dim=256,
            default_lora_rank=8,
            buffer_threshold=10,
            auto_create=False,  # Disable auto-create for testing
        )
    
    def _create_sample(self, innovation_score: float = 0.8) -> InnovationSample:
        """Helper to create a test sample."""
        return InnovationSample(
            query=np.random.randn(256),
            target_output=np.random.randn(256),
            innovation_score=innovation_score,
            innovation_components={},
        )
    
    def test_collect_sample(self):
        """Test sample collection."""
        self.factory.collect_sample(
            query=np.random.randn(256),
            target_output=np.random.randn(256),
            innovation_score=0.85,
        )
        
        self.assertEqual(len(self.factory.sample_buffer), 1)
    
    def test_determine_creation_mode(self):
        """Test creation mode determination."""
        # Emergency mode
        mode = self.factory.determine_creation_mode(
            num_samples=3,
            avg_innovation_score=0.95,
            sample_diversity=0.5,
            risk_tolerance=0.8,
        )
        self.assertEqual(mode, CreationMode.EMERGENCY)
        
        # Few-shot mode
        mode = self.factory.determine_creation_mode(
            num_samples=10,
            avg_innovation_score=0.85,
            sample_diversity=0.8,
            risk_tolerance=0.5,
        )
        self.assertEqual(mode, CreationMode.FEW_SHOT)
        
        # Confident mode
        mode = self.factory.determine_creation_mode(
            num_samples=100,
            avg_innovation_score=0.7,
            sample_diversity=0.5,
            risk_tolerance=0.5,
        )
        self.assertEqual(mode, CreationMode.CONFIDENT)
    
    def test_compute_sample_diversity(self):
        """Test sample diversity computation."""
        # Similar samples (low diversity)
        base = np.random.randn(256)
        similar_samples = [
            InnovationSample(
                query=base + np.random.randn(256) * 0.01,
                target_output=np.random.randn(256),
                innovation_score=0.8,
                innovation_components={},
            )
            for _ in range(5)
        ]
        
        diversity = self.factory.compute_sample_diversity(similar_samples)
        self.assertLess(diversity, 0.5)
        
        # Diverse samples
        diverse_samples = [
            InnovationSample(
                query=np.random.randn(256),
                target_output=np.random.randn(256),
                innovation_score=0.8,
                innovation_components={},
            )
            for _ in range(5)
        ]
        
        diversity = self.factory.compute_sample_diversity(diverse_samples)
        self.assertGreater(diversity, 0.3)
    
    def test_create_from_buffer(self):
        """Test Nano Model creation from buffer."""
        # Add samples to buffer
        for _ in range(15):
            self.factory.sample_buffer.append(self._create_sample())
        
        # Create Nano
        nano = self.factory.create_from_buffer(
            force_mode=CreationMode.FEW_SHOT,
            innovation_domain="test_domain",
        )
        
        self.assertIsNotNone(nano)
        self.assertEqual(nano.innovation_domain, "test_domain")
        self.assertEqual(len(self.registry.nanos), 1)
    
    def test_create_emergency(self):
        """Test emergency Nano Model creation."""
        samples = [self._create_sample(0.95) for _ in range(3)]
        
        nano = self.factory.create_emergency(samples, "emergency_domain")
        
        self.assertIsNotNone(nano)
        self.assertEqual(nano.state, NanoLifecycleState.TRIAL)
        self.assertEqual(nano.creation_mode, "emergency")
    
    def test_lora_training(self):
        """Test LoRA weight training."""
        samples = [self._create_sample() for _ in range(10)]
        config = CreationConfig.for_mode(CreationMode.FEW_SHOT)
        
        lora_weights = self.factory._train_lora(samples, config)
        
        self.assertIn("q_proj", lora_weights)
        self.assertEqual(lora_weights["q_proj"].rank, config.lora_rank)
    
    def test_confidence_computation(self):
        """Test confidence score computation."""
        samples = [self._create_sample() for _ in range(20)]
        config = CreationConfig.for_mode(CreationMode.STANDARD)
        
        confidence = self.factory._compute_confidence(samples, config)
        
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
    
    def test_buffer_status(self):
        """Test buffer status reporting."""
        # Empty buffer
        status = self.factory.get_buffer_status()
        self.assertEqual(status["buffer_size"], 0)
        self.assertFalse(status["ready_for_creation"])
        
        # Add samples
        for _ in range(15):
            self.factory.sample_buffer.append(self._create_sample())
        
        status = self.factory.get_buffer_status()
        self.assertEqual(status["buffer_size"], 15)
        self.assertTrue(status["ready_for_creation"])
    
    def test_statistics(self):
        """Test statistics collection."""
        stats = self.factory.get_statistics()
        
        self.assertEqual(stats["total_created"], 0)
        self.assertIn("buffer_size", stats)
    
    def test_clear_buffer(self):
        """Test buffer clearing."""
        for _ in range(5):
            self.factory.sample_buffer.append(self._create_sample())
        
        self.factory.clear_buffer()
        
        self.assertEqual(len(self.factory.sample_buffer), 0)


if __name__ == "__main__":
    unittest.main()
