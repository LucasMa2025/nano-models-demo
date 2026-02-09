"""
Versioned KV Store
==================

Implements the hierarchical KV storage system with:
- Global KV: Read-only, accessible by all Nano Models
- Shared KV: Collaborative write with version control
- Exclusive KV: Single Nano Model read-write

Features:
- Version control for rollback
- Conflict detection
- Write locking for shared KV
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
import numpy as np
import logging
import json

try:
    from ..core.models import KVEntry, KVAccessMode
except ImportError:
    from core.models import KVEntry, KVAccessMode

logger = logging.getLogger(__name__)


class VersionedKVStore:
    """
    Versioned Key-Value store with Nano Model binding.
    
    Supports:
    - Version control for rollback
    - Write locking for shared KV
    - Conflict detection via semantic similarity
    - Access mode enforcement (exclusive/shared/global)
    """
    
    def __init__(self, max_entries: int = 10000):
        """
        Initialize the KV Store.
        
        Args:
            max_entries: Maximum number of KV entries
        """
        self.entries: Dict[str, KVEntry] = {}
        self.max_entries = max_entries
        
        # Statistics
        self.total_inserts = 0
        self.total_updates = 0
        self.total_hits = 0
        self.total_conflicts = 0
        
        logger.info(f"VersionedKVStore initialized: max_entries={max_entries}")
    
    def insert(
        self,
        key: np.ndarray,
        value: np.ndarray,
        nano_id: Optional[str] = None,
        access_mode: str = "exclusive",
        kv_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Insert a new KV entry.
        
        Args:
            key: Key embedding
            value: Value embedding
            nano_id: Bound Nano Model ID (None for global)
            access_mode: Access mode (exclusive/shared/global)
            kv_id: Optional custom KV ID
            metadata: Additional metadata
            
        Returns:
            KV entry ID
        """
        # Enforce capacity
        if len(self.entries) >= self.max_entries:
            self._evict_lru()
        
        entry = KVEntry(
            key=key,
            value=value,
            nano_id=nano_id,
            access_mode=access_mode,
            kv_id=kv_id or "",
            metadata=metadata or {},
        )
        
        self.entries[entry.kv_id] = entry
        self.total_inserts += 1
        
        logger.debug(f"Inserted KV entry {entry.kv_id} (mode={access_mode})")
        
        return entry.kv_id
    
    def get(self, kv_id: str) -> Optional[KVEntry]:
        """Get a KV entry by ID."""
        entry = self.entries.get(kv_id)
        if entry:
            entry.access_count += 1
        return entry
    
    def query(
        self,
        query_embedding: np.ndarray,
        nano_id: Optional[str] = None,
        threshold: float = 0.5,
        top_k: int = 10,
    ) -> List[Tuple[KVEntry, float]]:
        """
        Query KV entries by similarity.
        
        Args:
            query_embedding: Query embedding
            nano_id: Nano Model ID for access filtering
            threshold: Minimum similarity threshold
            top_k: Maximum number of results
            
        Returns:
            List of (KVEntry, similarity) tuples
        """
        results = []
        
        for entry in self.entries.values():
            # Check access permissions
            if not self._can_access(entry, nano_id):
                continue
            
            similarity = entry.compute_similarity(query_embedding)
            
            if similarity >= threshold:
                results.append((entry, similarity))
                entry.access_count += 1
                self.total_hits += 1
        
        # Sort by similarity (descending) and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def update(
        self,
        kv_id: str,
        new_value: np.ndarray,
        nano_id: str,
        force: bool = False,
    ) -> Tuple[bool, float]:
        """
        Update a KV entry value.
        
        Args:
            kv_id: KV entry ID
            new_value: New value embedding
            nano_id: Nano Model ID requesting update
            force: Force update even with high conflict
            
        Returns:
            Tuple of (success, conflict_score)
        """
        entry = self.entries.get(kv_id)
        if entry is None:
            return False, 1.0
        
        success, conflict = entry.update_value(new_value, nano_id, force)
        
        if success:
            self.total_updates += 1
        if conflict > 0.5:
            self.total_conflicts += 1
        
        return success, conflict
    
    def delete(self, kv_id: str) -> bool:
        """Delete a KV entry."""
        if kv_id in self.entries:
            del self.entries[kv_id]
            return True
        return False
    
    def rollback(self, kv_id: str, target_version: int) -> bool:
        """Rollback a KV entry to a specific version."""
        entry = self.entries.get(kv_id)
        if entry:
            return entry.rollback(target_version)
        return False
    
    def acquire_lock(self, kv_id: str, nano_id: str) -> bool:
        """Acquire write lock for a shared KV entry."""
        entry = self.entries.get(kv_id)
        if entry:
            return entry.acquire_lock(nano_id)
        return False
    
    def release_lock(self, kv_id: str, nano_id: str) -> bool:
        """Release write lock for a shared KV entry."""
        entry = self.entries.get(kv_id)
        if entry:
            return entry.release_lock(nano_id)
        return False
    
    def _can_access(self, entry: KVEntry, nano_id: Optional[str]) -> bool:
        """Check if a Nano Model can access an entry."""
        if entry.access_mode == "global":
            return True
        if entry.access_mode == "exclusive":
            return entry.nano_id == nano_id
        if entry.access_mode == "shared":
            return True  # Shared entries accessible to all
        return False
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.entries:
            return
        
        # Find entry with lowest access count
        lru_id = min(self.entries.keys(), key=lambda k: self.entries[k].access_count)
        del self.entries[lru_id]
        logger.debug(f"Evicted LRU KV entry {lru_id}")
    
    def get_entries_for_nano(self, nano_id: str) -> List[KVEntry]:
        """Get all entries bound to a Nano Model."""
        return [e for e in self.entries.values() if e.nano_id == nano_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        mode_counts = {"exclusive": 0, "shared": 0, "global": 0}
        for entry in self.entries.values():
            mode_counts[entry.access_mode] = mode_counts.get(entry.access_mode, 0) + 1
        
        return {
            "total_entries": len(self.entries),
            "max_entries": self.max_entries,
            "total_inserts": self.total_inserts,
            "total_updates": self.total_updates,
            "total_hits": self.total_hits,
            "total_conflicts": self.total_conflicts,
            "mode_distribution": mode_counts,
        }
    
    def export_state(self) -> Dict[str, Any]:
        """Export store state for persistence."""
        return {
            "entries": {
                kv_id: entry.to_dict()
                for kv_id, entry in self.entries.items()
            },
            "statistics": {
                "total_inserts": self.total_inserts,
                "total_updates": self.total_updates,
                "total_hits": self.total_hits,
                "total_conflicts": self.total_conflicts,
            },
        }
    
    def import_state(self, state: Dict[str, Any]):
        """Import store state from persistence."""
        self.entries.clear()
        
        for kv_id, entry_data in state.get("entries", {}).items():
            entry = KVEntry.from_dict(entry_data)
            self.entries[kv_id] = entry
        
        stats = state.get("statistics", {})
        self.total_inserts = stats.get("total_inserts", len(self.entries))
        self.total_updates = stats.get("total_updates", 0)
        self.total_hits = stats.get("total_hits", 0)
        self.total_conflicts = stats.get("total_conflicts", 0)
        
        logger.info(f"Imported {len(self.entries)} KV entries from state")


class HierarchicalKVStore:
    """
    Hierarchical KV Store implementing the three-tier binding system.
    
    Tiers:
    - Global KV: Accessible by all Nano Models (read-only)
    - Shared KV: Accessible by designated Nano Models (collaborative write)
    - Exclusive KV: Bound to single Nano Model (read-write)
    
    Access Priority: Exclusive > Shared > Global
    """
    
    def __init__(self, max_entries_per_tier: int = 5000):
        """
        Initialize the Hierarchical KV Store.
        
        Args:
            max_entries_per_tier: Maximum entries per tier
        """
        self.global_store = VersionedKVStore(max_entries_per_tier)
        self.shared_store = VersionedKVStore(max_entries_per_tier)
        self.exclusive_store = VersionedKVStore(max_entries_per_tier)
        
        # Shared KV access mapping: kv_id -> set of nano_ids
        self.shared_access: Dict[str, Set[str]] = {}
        
        logger.info(f"HierarchicalKVStore initialized: max_per_tier={max_entries_per_tier}")
    
    def insert_global(
        self,
        key: np.ndarray,
        value: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a global KV entry (read-only for all)."""
        return self.global_store.insert(
            key=key,
            value=value,
            nano_id=None,
            access_mode="global",
            metadata=metadata,
        )
    
    def insert_shared(
        self,
        key: np.ndarray,
        value: np.ndarray,
        nano_ids: Set[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a shared KV entry (collaborative write for designated Nanos)."""
        kv_id = self.shared_store.insert(
            key=key,
            value=value,
            nano_id=None,
            access_mode="shared",
            metadata=metadata,
        )
        self.shared_access[kv_id] = nano_ids.copy()
        return kv_id
    
    def insert_exclusive(
        self,
        key: np.ndarray,
        value: np.ndarray,
        nano_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert an exclusive KV entry (single Nano read-write)."""
        return self.exclusive_store.insert(
            key=key,
            value=value,
            nano_id=nano_id,
            access_mode="exclusive",
            metadata=metadata,
        )
    
    def query(
        self,
        query_embedding: np.ndarray,
        nano_id: str,
        threshold: float = 0.5,
        top_k: int = 10,
    ) -> List[Tuple[KVEntry, float, str]]:
        """
        Query across all tiers with priority ordering.
        
        Args:
            query_embedding: Query embedding
            nano_id: Nano Model ID
            threshold: Minimum similarity threshold
            top_k: Maximum results
            
        Returns:
            List of (KVEntry, similarity, tier) tuples
        """
        results = []
        
        # Query exclusive (highest priority)
        exclusive_results = self.exclusive_store.query(
            query_embedding, nano_id, threshold, top_k
        )
        for entry, sim in exclusive_results:
            results.append((entry, sim, "exclusive"))
        
        # Query shared (medium priority)
        shared_results = self.shared_store.query(
            query_embedding, None, threshold, top_k
        )
        for entry, sim in shared_results:
            # Check if nano has access
            if entry.kv_id in self.shared_access:
                if nano_id in self.shared_access[entry.kv_id]:
                    results.append((entry, sim, "shared"))
        
        # Query global (lowest priority)
        global_results = self.global_store.query(
            query_embedding, None, threshold, top_k
        )
        for entry, sim in global_results:
            results.append((entry, sim, "global"))
        
        # Sort by (tier_priority, similarity)
        tier_priority = {"exclusive": 0, "shared": 1, "global": 2}
        results.sort(key=lambda x: (tier_priority[x[2]], -x[1]))
        
        return results[:top_k]
    
    def get_all_for_nano(self, nano_id: str) -> List[KVEntry]:
        """Get all KV entries accessible by a Nano Model."""
        entries = []
        
        # Exclusive entries
        entries.extend(self.exclusive_store.get_entries_for_nano(nano_id))
        
        # Shared entries
        for kv_id, nano_ids in self.shared_access.items():
            if nano_id in nano_ids:
                entry = self.shared_store.get(kv_id)
                if entry:
                    entries.append(entry)
        
        # Global entries
        entries.extend(self.global_store.entries.values())
        
        return entries
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hierarchical store statistics."""
        return {
            "global": self.global_store.get_statistics(),
            "shared": self.shared_store.get_statistics(),
            "exclusive": self.exclusive_store.get_statistics(),
            "shared_access_mappings": len(self.shared_access),
        }
    
    def export_state(self) -> Dict[str, Any]:
        """Export complete state for persistence."""
        return {
            "global": self.global_store.export_state(),
            "shared": self.shared_store.export_state(),
            "exclusive": self.exclusive_store.export_state(),
            "shared_access": {
                kv_id: list(nano_ids)
                for kv_id, nano_ids in self.shared_access.items()
            },
        }
    
    def import_state(self, state: Dict[str, Any]):
        """Import complete state from persistence."""
        self.global_store.import_state(state.get("global", {}))
        self.shared_store.import_state(state.get("shared", {}))
        self.exclusive_store.import_state(state.get("exclusive", {}))
        
        self.shared_access = {
            kv_id: set(nano_ids)
            for kv_id, nano_ids in state.get("shared_access", {}).items()
        }
        
        logger.info("Imported hierarchical KV store state")
