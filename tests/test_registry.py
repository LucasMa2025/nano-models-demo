"""
Unit tests for Nano Model Registry.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.registry import NanoRegistry
from core.models import NanoModel, NanoLifecycleState, KVEntry, LoRAWeights


class TestNanoRegistry(unittest.TestCase):
    """Tests for NanoRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = NanoRegistry(
            dormant_threshold_days=7,
            deprecate_threshold_days=30,
            trial_validation_count=5,
            trial_success_threshold=0.6,
            max_active_nanos=10,
        )
    
    def _create_nano(self, nano_id: str, state: NanoLifecycleState = NanoLifecycleState.ACTIVE) -> NanoModel:
        """Helper to create a test Nano Model."""
        return NanoModel(
            nano_id=nano_id,
            lora_weights={
                "q_proj": LoRAWeights(
                    layer_name="q_proj",
                    A=np.random.randn(8, 256) * 0.01,
                    B=np.random.randn(256, 8) * 0.01,
                ),
            },
            bound_kv_ids={f"{nano_id}_kv_1"},
            state=state,
            created_at=datetime.now(),
            last_activated=datetime.now(),
        )
    
    def test_register(self):
        """Test Nano Model registration."""
        nano = self._create_nano("nano_001")
        success = self.registry.register(nano)
        
        self.assertTrue(success)
        self.assertEqual(len(self.registry.nanos), 1)
        self.assertEqual(self.registry.total_registered, 1)
    
    def test_register_duplicate(self):
        """Test duplicate registration fails."""
        nano = self._create_nano("nano_001")
        self.registry.register(nano)
        
        # Try to register again
        success = self.registry.register(nano)
        self.assertFalse(success)
    
    def test_unregister(self):
        """Test Nano Model unregistration."""
        nano = self._create_nano("nano_001")
        self.registry.register(nano)
        
        success = self.registry.unregister("nano_001")
        self.assertTrue(success)
        self.assertEqual(len(self.registry.nanos), 0)
        self.assertEqual(self.registry.total_destroyed, 1)
    
    def test_get(self):
        """Test getting a Nano Model."""
        nano = self._create_nano("nano_001")
        self.registry.register(nano)
        
        retrieved = self.registry.get("nano_001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.nano_id, "nano_001")
        
        # Non-existent
        retrieved = self.registry.get("nano_999")
        self.assertIsNone(retrieved)
    
    def test_select(self):
        """Test Nano Model selection based on KV hit."""
        # Create and register Nano
        nano = self._create_nano("nano_001")
        self.registry.register(nano)
        
        # Create KV entries
        kv_entries = [
            KVEntry(
                key=np.random.randn(256),
                value=np.random.randn(256),
                nano_id="nano_001",
            ),
        ]
        
        # Select with matching query
        query = kv_entries[0].key.copy()
        selected = self.registry.select(query, kv_entries, hit_threshold=0.5)
        
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][0].nano_id, "nano_001")
    
    def test_select_excludes_dormant(self):
        """Test that dormant Nanos are excluded from selection."""
        nano = self._create_nano("nano_001", NanoLifecycleState.DORMANT)
        self.registry.register(nano)
        
        kv_entries = [
            KVEntry(
                key=np.random.randn(256),
                value=np.random.randn(256),
                nano_id="nano_001",
            ),
        ]
        
        query = kv_entries[0].key.copy()
        selected = self.registry.select(query, kv_entries, include_trial=False)
        
        self.assertEqual(len(selected), 0)
    
    def test_record_activation_result(self):
        """Test recording activation results."""
        nano = self._create_nano("nano_001")
        self.registry.register(nano)
        
        self.registry.record_activation_result("nano_001", success=True, contribution=0.5)
        
        retrieved = self.registry.get("nano_001")
        self.assertEqual(retrieved.success_count, 1)
        self.assertEqual(retrieved.total_contribution, 0.5)
    
    def test_trial_to_active_transition(self):
        """Test TRIAL → ACTIVE transition."""
        nano = self._create_nano("nano_001", NanoLifecycleState.TRIAL)
        self.registry.register(nano)
        
        # Record enough successful activations
        for _ in range(5):
            self.registry.record_activation_result("nano_001", success=True, contribution=0.5)
        
        retrieved = self.registry.get("nano_001")
        self.assertEqual(retrieved.state, NanoLifecycleState.ACTIVE)
    
    def test_trial_to_deprecated_transition(self):
        """Test TRIAL → DEPRECATED transition on validation failure."""
        nano = self._create_nano("nano_001", NanoLifecycleState.TRIAL)
        self.registry.register(nano)
        
        # Record mostly failed activations
        for _ in range(5):
            self.registry.record_activation_result("nano_001", success=False, contribution=0.1)
        
        retrieved = self.registry.get("nano_001")
        self.assertEqual(retrieved.state, NanoLifecycleState.DEPRECATED)
    
    def test_lifecycle_update(self):
        """Test lifecycle state updates."""
        nano = self._create_nano("nano_001")
        # Set last_activated to past
        nano.last_activated = datetime.now() - timedelta(days=10)
        self.registry.register(nano)
        
        self.registry.update_lifecycle()
        
        retrieved = self.registry.get("nano_001")
        self.assertEqual(retrieved.state, NanoLifecycleState.DORMANT)
    
    def test_reactivate(self):
        """Test reactivating a dormant Nano."""
        nano = self._create_nano("nano_001", NanoLifecycleState.DORMANT)
        self.registry.register(nano)
        
        success = self.registry.reactivate("nano_001")
        self.assertTrue(success)
        
        retrieved = self.registry.get("nano_001")
        self.assertEqual(retrieved.state, NanoLifecycleState.ACTIVE)
    
    def test_max_capacity_eviction(self):
        """Test LRU eviction when capacity exceeded."""
        # Register max_active Nanos
        for i in range(10):
            nano = self._create_nano(f"nano_{i:03d}")
            self.registry.register(nano)
        
        self.assertEqual(len(self.registry.nanos), 10)
        
        # Register one more
        nano = self._create_nano("nano_010")
        self.registry.register(nano)
        
        # Should still be at max
        self.assertEqual(len(self.registry.nanos), 10)
        # First one should be evicted
        self.assertIsNone(self.registry.get("nano_000"))
    
    def test_statistics(self):
        """Test statistics collection."""
        nano = self._create_nano("nano_001")
        self.registry.register(nano)
        
        stats = self.registry.get_statistics()
        
        self.assertEqual(stats["total_registered"], 1)
        self.assertEqual(stats["current_count"], 1)
        self.assertIn("state_distribution", stats)
    
    def test_export_import_state(self):
        """Test state export and import."""
        nano = self._create_nano("nano_001")
        self.registry.register(nano)
        
        # Export
        state = self.registry.export_state()
        
        # Create new registry and import
        new_registry = NanoRegistry()
        new_registry.import_state(state)
        
        self.assertEqual(len(new_registry.nanos), 1)
        self.assertIsNotNone(new_registry.get("nano_001"))


if __name__ == "__main__":
    unittest.main()
