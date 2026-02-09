"""
Unit tests for KV Store.
"""

import unittest
import numpy as np

import sys
import os

# Add parent directory to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

# Import modules directly
from core.models import KVEntry
from storage.kv_store import VersionedKVStore, HierarchicalKVStore


class TestVersionedKVStore(unittest.TestCase):
    """Tests for VersionedKVStore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.store = VersionedKVStore(max_entries=100)
        self.hidden_dim = 256
    
    def test_insert(self):
        """Test KV entry insertion."""
        key = np.random.randn(self.hidden_dim)
        value = np.random.randn(self.hidden_dim)
        
        kv_id = self.store.insert(key, value, nano_id="nano_001")
        
        self.assertIsNotNone(kv_id)
        self.assertEqual(len(self.store.entries), 1)
        self.assertEqual(self.store.total_inserts, 1)
    
    def test_get(self):
        """Test KV entry retrieval."""
        key = np.random.randn(self.hidden_dim)
        value = np.random.randn(self.hidden_dim)
        
        kv_id = self.store.insert(key, value)
        entry = self.store.get(kv_id)
        
        self.assertIsNotNone(entry)
        np.testing.assert_array_equal(entry.key, key)
    
    def test_query(self):
        """Test KV query by similarity."""
        # Insert entries
        keys = [np.random.randn(self.hidden_dim) for _ in range(5)]
        for key in keys:
            self.store.insert(key, np.random.randn(self.hidden_dim))
        
        # Query with first key
        results = self.store.query(keys[0], threshold=0.5)
        
        self.assertGreater(len(results), 0)
        self.assertGreater(results[0][1], 0.5)  # Similarity > threshold
    
    def test_update(self):
        """Test KV entry update."""
        key = np.random.randn(self.hidden_dim)
        value = np.random.randn(self.hidden_dim)
        
        kv_id = self.store.insert(key, value, nano_id="nano_001", access_mode="exclusive")
        
        # Update by owner (force to bypass conflict check)
        new_value = np.random.randn(self.hidden_dim)
        success, conflict = self.store.update(kv_id, new_value, "nano_001", force=True)
        
        self.assertTrue(success)
        self.assertEqual(self.store.total_updates, 1)
    
    def test_update_access_control(self):
        """Test update access control."""
        key = np.random.randn(self.hidden_dim)
        value = np.random.randn(self.hidden_dim)
        
        kv_id = self.store.insert(key, value, nano_id="nano_001", access_mode="exclusive")
        
        # Update by non-owner should fail
        new_value = np.random.randn(self.hidden_dim)
        success, conflict = self.store.update(kv_id, new_value, "nano_002")
        
        self.assertFalse(success)
    
    def test_delete(self):
        """Test KV entry deletion."""
        key = np.random.randn(self.hidden_dim)
        value = np.random.randn(self.hidden_dim)
        
        kv_id = self.store.insert(key, value)
        success = self.store.delete(kv_id)
        
        self.assertTrue(success)
        self.assertEqual(len(self.store.entries), 0)
    
    def test_rollback(self):
        """Test version rollback."""
        key = np.random.randn(self.hidden_dim)
        value = np.random.randn(self.hidden_dim)
        
        kv_id = self.store.insert(key, value, nano_id="nano_001", access_mode="exclusive")
        
        # Update (force to bypass conflict check)
        new_value = np.random.randn(self.hidden_dim)
        success, _ = self.store.update(kv_id, new_value, "nano_001", force=True)
        self.assertTrue(success)
        
        # Rollback
        success = self.store.rollback(kv_id, 1)
        
        self.assertTrue(success)
        entry = self.store.get(kv_id)
        np.testing.assert_array_almost_equal(entry.value, value)
    
    def test_lru_eviction(self):
        """Test LRU eviction when capacity exceeded."""
        # Fill store
        for i in range(100):
            self.store.insert(
                np.random.randn(self.hidden_dim),
                np.random.randn(self.hidden_dim),
            )
        
        self.assertEqual(len(self.store.entries), 100)
        
        # Insert one more
        self.store.insert(
            np.random.randn(self.hidden_dim),
            np.random.randn(self.hidden_dim),
        )
        
        # Should still be at max
        self.assertEqual(len(self.store.entries), 100)
    
    def test_statistics(self):
        """Test statistics collection."""
        self.store.insert(
            np.random.randn(self.hidden_dim),
            np.random.randn(self.hidden_dim),
        )
        
        stats = self.store.get_statistics()
        
        self.assertEqual(stats["total_entries"], 1)
        self.assertEqual(stats["total_inserts"], 1)
        self.assertIn("mode_distribution", stats)
    
    def test_export_import_state(self):
        """Test state export and import."""
        # Add entries
        for _ in range(5):
            self.store.insert(
                np.random.randn(self.hidden_dim),
                np.random.randn(self.hidden_dim),
            )
        
        # Export
        state = self.store.export_state()
        
        # Create new store and import
        new_store = VersionedKVStore()
        new_store.import_state(state)
        
        self.assertEqual(len(new_store.entries), 5)


class TestHierarchicalKVStore(unittest.TestCase):
    """Tests for HierarchicalKVStore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.store = HierarchicalKVStore(max_entries_per_tier=100)
        self.hidden_dim = 256
    
    def test_insert_global(self):
        """Test global KV insertion."""
        key = np.random.randn(self.hidden_dim)
        value = np.random.randn(self.hidden_dim)
        
        kv_id = self.store.insert_global(key, value)
        
        self.assertIsNotNone(kv_id)
        self.assertEqual(len(self.store.global_store.entries), 1)
    
    def test_insert_shared(self):
        """Test shared KV insertion."""
        key = np.random.randn(self.hidden_dim)
        value = np.random.randn(self.hidden_dim)
        
        kv_id = self.store.insert_shared(key, value, {"nano_001", "nano_002"})
        
        self.assertIsNotNone(kv_id)
        self.assertIn(kv_id, self.store.shared_access)
        self.assertEqual(self.store.shared_access[kv_id], {"nano_001", "nano_002"})
    
    def test_insert_exclusive(self):
        """Test exclusive KV insertion."""
        key = np.random.randn(self.hidden_dim)
        value = np.random.randn(self.hidden_dim)
        
        kv_id = self.store.insert_exclusive(key, value, "nano_001")
        
        self.assertIsNotNone(kv_id)
        self.assertEqual(len(self.store.exclusive_store.entries), 1)
    
    def test_query_priority(self):
        """Test query with tier priority."""
        key = np.random.randn(self.hidden_dim)
        value = np.random.randn(self.hidden_dim)
        
        # Insert in all tiers
        self.store.insert_global(key.copy(), value)
        self.store.insert_shared(key.copy(), value, {"nano_001"})
        self.store.insert_exclusive(key.copy(), value, "nano_001")
        
        # Query
        results = self.store.query(key, "nano_001", threshold=0.5)
        
        # Should return results from all tiers
        self.assertGreater(len(results), 0)
        
        # First result should be exclusive (highest priority)
        self.assertEqual(results[0][2], "exclusive")
    
    def test_get_all_for_nano(self):
        """Test getting all entries for a Nano Model."""
        # Insert entries
        self.store.insert_global(np.random.randn(self.hidden_dim), np.random.randn(self.hidden_dim))
        self.store.insert_shared(np.random.randn(self.hidden_dim), np.random.randn(self.hidden_dim), {"nano_001"})
        self.store.insert_exclusive(np.random.randn(self.hidden_dim), np.random.randn(self.hidden_dim), "nano_001")
        
        entries = self.store.get_all_for_nano("nano_001")
        
        # Should include global, shared (with access), and exclusive
        self.assertGreaterEqual(len(entries), 3)
    
    def test_statistics(self):
        """Test hierarchical statistics."""
        self.store.insert_global(np.random.randn(self.hidden_dim), np.random.randn(self.hidden_dim))
        self.store.insert_exclusive(np.random.randn(self.hidden_dim), np.random.randn(self.hidden_dim), "nano_001")
        
        stats = self.store.get_statistics()
        
        self.assertIn("global", stats)
        self.assertIn("shared", stats)
        self.assertIn("exclusive", stats)
    
    def test_export_import_state(self):
        """Test state export and import."""
        # Add entries
        self.store.insert_global(np.random.randn(self.hidden_dim), np.random.randn(self.hidden_dim))
        self.store.insert_shared(np.random.randn(self.hidden_dim), np.random.randn(self.hidden_dim), {"nano_001"})
        
        # Export
        state = self.store.export_state()
        
        # Create new store and import
        new_store = HierarchicalKVStore()
        new_store.import_state(state)
        
        self.assertEqual(len(new_store.global_store.entries), 1)
        self.assertEqual(len(new_store.shared_store.entries), 1)


if __name__ == "__main__":
    unittest.main()
