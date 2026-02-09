"""
Unit tests for the integrated Nano Model System.

Note: These tests use a simplified approach due to the complex
import structure. For full integration testing, run main.py.
"""

import unittest
import numpy as np
import tempfile
import os
import sys

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

from core.models import NanoModel, NanoLifecycleState, KVEntry, LoRAWeights
from core.innovation_detector import InnovationDetector
from core.registry import NanoRegistry
from core.factory import NanoModelFactory
from core.fusion import ConflictAwareFusion, OutputFusion
from storage.kv_store import HierarchicalKVStore


class TestIntegratedComponents(unittest.TestCase):
    """Tests for integrated component behavior."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_dim = 128
        
        # Initialize components
        self.detector = InnovationDetector(
            hidden_dim=self.hidden_dim,
            innovation_threshold=0.7,
        )
        
        self.registry = NanoRegistry(
            dormant_threshold_days=7,
            deprecate_threshold_days=30,
            max_active_nanos=10,
        )
        
        self.factory = NanoModelFactory(
            registry=self.registry,
            hidden_dim=self.hidden_dim,
            buffer_threshold=5,
            auto_create=False,
        )
        
        self.kv_store = HierarchicalKVStore()
        
        self.fusion = ConflictAwareFusion()
        self.output_fusion = OutputFusion()
    
    def test_detector_registry_integration(self):
        """Test innovation detector with registry."""
        # Build semantic subspaces
        for _ in range(150):
            hidden_states = np.random.randn(1, 10, self.hidden_dim)
            self.detector.update_semantic_subspaces(hidden_states, is_low_entropy=True)
        
        # Create and register a Nano
        nano = self._create_test_nano("nano_001")
        self.registry.register(nano)
        
        # Inject KV for the Nano
        key = np.random.randn(self.hidden_dim)
        self.kv_store.insert_exclusive(key, np.random.randn(self.hidden_dim), "nano_001")
        
        # Run detection
        hidden_states = np.random.randn(1, 10, self.hidden_dim)
        attention_weights = np.random.rand(1, 8, 10, 10)
        
        all_kvs = list(self.kv_store.exclusive_store.entries.values())
        result = self.detector.detect(hidden_states, attention_weights, all_kvs)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result.is_innovation, bool)
    
    def test_factory_registry_integration(self):
        """Test factory creating and registering Nanos."""
        # Collect samples
        for _ in range(10):
            self.factory.collect_sample(
                query=np.random.randn(self.hidden_dim),
                target_output=np.random.randn(self.hidden_dim),
                innovation_score=0.85,
            )
        
        # Create Nano from buffer
        nano = self.factory.create_from_buffer(innovation_domain="test")
        
        self.assertIsNotNone(nano)
        self.assertEqual(len(self.registry.nanos), 1)
    
    def test_kv_store_fusion_integration(self):
        """Test KV store with fusion."""
        # Create Nanos
        nano1 = self._create_test_nano("nano_001")
        nano2 = self._create_test_nano("nano_002")
        
        self.registry.register(nano1)
        self.registry.register(nano2)
        
        # Create outputs
        nano_outputs = {
            "nano_001": np.random.randn(self.hidden_dim),
            "nano_002": np.random.randn(self.hidden_dim),
        }
        kv_hit_scores = {"nano_001": 0.8, "nano_002": 0.6}
        
        # Fuse
        result = self.fusion.fuse(nano_outputs, kv_hit_scores)
        
        self.assertIsNotNone(result.fused_output)
        self.assertEqual(result.fused_output.shape, (self.hidden_dim,))
    
    def test_full_pipeline_simulation(self):
        """Test simulated full pipeline."""
        # 1. Build semantic subspaces
        for _ in range(150):
            hidden_states = np.random.randn(1, 10, self.hidden_dim)
            self.detector.update_semantic_subspaces(hidden_states, is_low_entropy=True)
        
        # 2. Inject global knowledge
        for i in range(5):
            self.kv_store.insert_global(
                np.random.randn(self.hidden_dim),
                np.random.randn(self.hidden_dim),
            )
        
        # 3. Simulate queries
        innovation_count = 0
        for _ in range(20):
            query = np.random.randn(self.hidden_dim)
            hidden_states = query.reshape(1, 1, -1)
            attention_weights = np.random.rand(1, 8, 1, 1)
            
            all_kvs = list(self.kv_store.global_store.entries.values())
            result = self.detector.detect(hidden_states, attention_weights, all_kvs)
            
            if result.is_innovation:
                innovation_count += 1
                # Collect sample
                self.factory.collect_sample(
                    query=query,
                    target_output=np.random.randn(self.hidden_dim),
                    innovation_score=result.score,
                )
        
        # 4. Check statistics
        stats = self.detector.get_statistics()
        self.assertEqual(stats["detection_count"], 20)
    
    def _create_test_nano(self, nano_id: str) -> NanoModel:
        """Helper to create a test Nano Model."""
        return NanoModel(
            nano_id=nano_id,
            lora_weights={
                "q_proj": LoRAWeights(
                    layer_name="q_proj",
                    A=np.random.randn(8, self.hidden_dim) * 0.01,
                    B=np.random.randn(self.hidden_dim, 8) * 0.01,
                ),
            },
            bound_kv_ids={f"{nano_id}_kv_1"},
            state=NanoLifecycleState.ACTIVE,
            created_at=__import__('datetime').datetime.now(),
            last_activated=__import__('datetime').datetime.now(),
        )


class TestOutputFusionIntegration(unittest.TestCase):
    """Tests for output fusion integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_dim = 128
        self.output_fusion = OutputFusion()
    
    def test_full_output_fusion(self):
        """Test complete output fusion."""
        base_output = np.random.randn(self.hidden_dim)
        aga_output = np.random.randn(self.hidden_dim)
        nano_output = np.random.randn(self.hidden_dim)
        
        final, diagnostics = self.output_fusion.fuse(
            base_output=base_output,
            aga_output=aga_output,
            nano_output=nano_output,
            alpha=0.3,
            gamma=0.2,
        )
        
        self.assertEqual(final.shape, (self.hidden_dim,))
        self.assertIn("alpha", diagnostics)
        self.assertIn("gamma", diagnostics)


class TestLifecycleIntegration(unittest.TestCase):
    """Tests for lifecycle management integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = NanoRegistry(
            dormant_threshold_days=7,
            deprecate_threshold_days=30,
            trial_validation_count=5,
            trial_success_threshold=0.6,
        )
    
    def test_trial_validation_flow(self):
        """Test TRIAL → ACTIVE validation flow."""
        from datetime import datetime
        
        nano = NanoModel(
            nano_id="trial_nano",
            lora_weights={},
            bound_kv_ids=set(),
            state=NanoLifecycleState.TRIAL,
            created_at=datetime.now(),
            last_activated=datetime.now(),
        )
        
        self.registry.register(nano)
        
        # Record successful activations
        for _ in range(5):
            self.registry.record_activation_result("trial_nano", success=True, contribution=0.5)
        
        # Should transition to ACTIVE
        retrieved = self.registry.get("trial_nano")
        self.assertEqual(retrieved.state, NanoLifecycleState.ACTIVE)
    
    def test_trial_failure_flow(self):
        """Test TRIAL → DEPRECATED failure flow."""
        from datetime import datetime
        
        nano = NanoModel(
            nano_id="failing_nano",
            lora_weights={},
            bound_kv_ids=set(),
            state=NanoLifecycleState.TRIAL,
            created_at=datetime.now(),
            last_activated=datetime.now(),
        )
        
        self.registry.register(nano)
        
        # Record failed activations
        for _ in range(5):
            self.registry.record_activation_result("failing_nano", success=False, contribution=0.1)
        
        # Should transition to DEPRECATED
        retrieved = self.registry.get("failing_nano")
        self.assertEqual(retrieved.state, NanoLifecycleState.DEPRECATED)


if __name__ == "__main__":
    unittest.main()
