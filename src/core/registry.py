"""
Nano Model Registry
===================

Manages Nano Model lifecycle including:
- Registration of new Nano Models
- Selection based on KV hit
- Lifecycle state transitions (TRIAL → ACTIVE → DORMANT → DEPRECATED)
- Cleanup of deprecated models
- LRU eviction when capacity exceeded

Implements Iron Law 2 (KV Binding) through exclusive KV access control.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from collections import OrderedDict
import logging
import json

from .models import NanoModel, NanoLifecycleState, KVEntry

logger = logging.getLogger(__name__)


class NanoRegistry:
    """
    Registry for managing Nano Model lifecycle.
    
    Handles:
    - Registration of new Nano Models
    - Selection based on KV hit
    - Lifecycle state transitions
    - Cleanup of deprecated models
    - Capacity management via LRU eviction
    
    Lifecycle State Machine:
    CREATE → [TRIAL] → ACTIVE → DORMANT → DEPRECATED → DESTROY
    """
    
    def __init__(
        self,
        dormant_threshold_days: int = 7,
        deprecate_threshold_days: int = 30,
        trial_validation_count: int = 10,
        trial_success_threshold: float = 0.6,
        max_active_nanos: int = 100,
    ):
        """
        Initialize the Nano Registry.
        
        Args:
            dormant_threshold_days: Days of inactivity before dormancy
            deprecate_threshold_days: Days of dormancy before deprecation
            trial_validation_count: Activations needed to exit TRIAL state
            trial_success_threshold: Success rate needed to exit TRIAL state
            max_active_nanos: Maximum number of active Nano Models
        """
        self.nanos: OrderedDict[str, NanoModel] = OrderedDict()
        self.dormant_threshold = timedelta(days=dormant_threshold_days)
        self.deprecate_threshold = timedelta(days=deprecate_threshold_days)
        self.trial_validation_count = trial_validation_count
        self.trial_success_threshold = trial_success_threshold
        self.max_active = max_active_nanos
        
        # Statistics
        self.total_registered = 0
        self.total_deprecated = 0
        self.total_destroyed = 0
        
        # Event log
        self.event_log: List[Dict[str, Any]] = []
        
        logger.info(
            f"NanoRegistry initialized: max_active={max_active_nanos}, "
            f"dormant_days={dormant_threshold_days}, deprecate_days={deprecate_threshold_days}"
        )
    
    def register(self, nano: NanoModel) -> bool:
        """
        Register a new Nano Model.
        
        Args:
            nano: NanoModel to register
            
        Returns:
            True if registration succeeded, False otherwise
        """
        if nano.nano_id in self.nanos:
            logger.warning(f"Nano Model {nano.nano_id} already registered")
            return False
        
        # Enforce max active limit via LRU eviction
        while len(self.nanos) >= self.max_active:
            self._evict_least_used()
        
        self.nanos[nano.nano_id] = nano
        self.total_registered += 1
        
        self._log_event("register", nano.nano_id, {
            "state": nano.state.value,
            "creation_mode": nano.creation_mode,
            "bound_kv_count": len(nano.bound_kv_ids),
        })
        
        logger.info(
            f"Registered Nano Model {nano.nano_id} "
            f"(state={nano.state.value}, mode={nano.creation_mode})"
        )
        
        return True
    
    def unregister(self, nano_id: str) -> bool:
        """
        Unregister (destroy) a Nano Model.
        
        Args:
            nano_id: ID of Nano Model to remove
            
        Returns:
            True if removal succeeded
        """
        if nano_id not in self.nanos:
            return False
        
        nano = self.nanos.pop(nano_id)
        self.total_destroyed += 1
        
        self._log_event("destroy", nano_id, {
            "final_state": nano.state.value,
            "activation_count": nano.activation_count,
            "success_rate": nano.success_rate,
        })
        
        logger.info(f"Destroyed Nano Model {nano_id}")
        return True
    
    def get(self, nano_id: str) -> Optional[NanoModel]:
        """Get a Nano Model by ID."""
        return self.nanos.get(nano_id)
    
    def select(
        self,
        query_embedding: Any,
        kv_store: List[KVEntry],
        hit_threshold: float = 0.3,
        include_trial: bool = True,
    ) -> List[Tuple[NanoModel, float]]:
        """
        Select Nano Models based on KV hit.
        
        Only active (and optionally trial) Nano Models with bound KV entries
        that match the query are selected.
        
        Args:
            query_embedding: Query embedding for matching
            kv_store: List of KV entries
            hit_threshold: Minimum hit score for selection
            include_trial: Whether to include TRIAL state Nanos
            
        Returns:
            List of (NanoModel, hit_score) tuples, sorted by hit score
        """
        selected = []
        
        valid_states = {NanoLifecycleState.ACTIVE}
        if include_trial:
            valid_states.add(NanoLifecycleState.TRIAL)
        
        for nano in self.nanos.values():
            if nano.state not in valid_states:
                continue
            
            # Check if any bound KV entries hit
            bound_kvs = [kv for kv in kv_store if kv.nano_id == nano.nano_id]
            
            if not bound_kvs:
                continue
            
            # Compute hit score
            hit_count = sum(1 for kv in bound_kvs if kv.hit(query_embedding, hit_threshold))
            hit_score = hit_count / len(bound_kvs)
            
            if hit_score > hit_threshold:
                selected.append((nano, hit_score))
                
                # Update activation stats (but don't record success yet)
                nano.last_activated = datetime.now()
                nano.activation_count += 1
                
                # Move to end of OrderedDict (LRU update)
                self.nanos.move_to_end(nano.nano_id)
        
        # Sort by hit score (descending)
        selected.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Selected {len(selected)} Nano Models for query")
        
        return selected
    
    def record_activation_result(
        self,
        nano_id: str,
        success: bool,
        contribution: float,
    ):
        """
        Record the result of a Nano Model activation.
        
        Args:
            nano_id: ID of the Nano Model
            success: Whether the activation was successful
            contribution: Contribution weight used
        """
        if nano_id not in self.nanos:
            return
        
        nano = self.nanos[nano_id]
        nano.record_activation(success, contribution)
        
        # Check TRIAL → ACTIVE transition
        if nano.state == NanoLifecycleState.TRIAL:
            if nano.activation_count >= self.trial_validation_count:
                if nano.success_rate >= self.trial_success_threshold:
                    self._transition_state(nano, NanoLifecycleState.ACTIVE)
                else:
                    # Failed validation, deprecate
                    self._transition_state(nano, NanoLifecycleState.DEPRECATED)
    
    def update_lifecycle(self):
        """
        Update lifecycle states based on usage patterns.
        
        Called periodically to transition states:
        - ACTIVE → DORMANT: Low usage
        - DORMANT → ACTIVE: Reactivation (handled in select)
        - DORMANT → DEPRECATED: Prolonged inactivity
        - DEPRECATED → DESTROY: After grace period
        """
        now = datetime.now()
        to_destroy = []
        
        for nano in list(self.nanos.values()):
            time_since_activation = now - nano.last_activated
            
            if nano.state == NanoLifecycleState.ACTIVE:
                if time_since_activation > self.dormant_threshold:
                    self._transition_state(nano, NanoLifecycleState.DORMANT)
            
            elif nano.state == NanoLifecycleState.DORMANT:
                if time_since_activation > self.deprecate_threshold:
                    self._transition_state(nano, NanoLifecycleState.DEPRECATED)
            
            elif nano.state == NanoLifecycleState.DEPRECATED:
                # Remove after grace period (2x deprecate threshold)
                if time_since_activation > self.deprecate_threshold * 2:
                    to_destroy.append(nano.nano_id)
        
        # Destroy deprecated Nanos
        for nano_id in to_destroy:
            self.unregister(nano_id)
        
        if to_destroy:
            logger.info(f"Lifecycle update: destroyed {len(to_destroy)} deprecated Nanos")
    
    def reactivate(self, nano_id: str) -> bool:
        """
        Reactivate a dormant Nano Model.
        
        Args:
            nano_id: ID of Nano Model to reactivate
            
        Returns:
            True if reactivation succeeded
        """
        if nano_id not in self.nanos:
            return False
        
        nano = self.nanos[nano_id]
        if nano.state == NanoLifecycleState.DORMANT:
            self._transition_state(nano, NanoLifecycleState.ACTIVE)
            return True
        
        return False
    
    def _transition_state(self, nano: NanoModel, new_state: NanoLifecycleState):
        """Transition a Nano Model to a new state."""
        old_state = nano.state
        nano.state = new_state
        
        if new_state == NanoLifecycleState.DEPRECATED:
            self.total_deprecated += 1
        
        self._log_event("state_transition", nano.nano_id, {
            "from_state": old_state.value,
            "to_state": new_state.value,
            "activation_count": nano.activation_count,
            "success_rate": nano.success_rate,
        })
        
        logger.info(f"Nano {nano.nano_id}: {old_state.value} → {new_state.value}")
    
    def _evict_least_used(self):
        """Evict the least recently used Nano Model."""
        if not self.nanos:
            return
        
        # OrderedDict maintains insertion order; first item is LRU
        lru_nano_id = next(iter(self.nanos))
        lru_nano = self.nanos[lru_nano_id]
        
        # Transition to deprecated first
        if lru_nano.state != NanoLifecycleState.DEPRECATED:
            self._transition_state(lru_nano, NanoLifecycleState.DEPRECATED)
        
        self.unregister(lru_nano_id)
        logger.info(f"Evicted LRU Nano Model {lru_nano_id}")
    
    def _log_event(self, event_type: str, nano_id: str, details: Dict[str, Any]):
        """Log a registry event."""
        self.event_log.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "nano_id": nano_id,
            "details": details,
        })
        
        # Keep only last 1000 events
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-1000:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        state_counts = {}
        for state in NanoLifecycleState:
            state_counts[state.value] = sum(
                1 for n in self.nanos.values() if n.state == state
            )
        
        return {
            "total_registered": self.total_registered,
            "total_deprecated": self.total_deprecated,
            "total_destroyed": self.total_destroyed,
            "current_count": len(self.nanos),
            "max_capacity": self.max_active,
            "state_distribution": state_counts,
            "average_success_rate": sum(n.success_rate for n in self.nanos.values()) / max(len(self.nanos), 1),
            "average_activation_count": sum(n.activation_count for n in self.nanos.values()) / max(len(self.nanos), 1),
        }
    
    def get_all_nanos(self, state_filter: Optional[NanoLifecycleState] = None) -> List[NanoModel]:
        """
        Get all Nano Models, optionally filtered by state.
        
        Args:
            state_filter: Optional state to filter by
            
        Returns:
            List of Nano Models
        """
        if state_filter is None:
            return list(self.nanos.values())
        return [n for n in self.nanos.values() if n.state == state_filter]
    
    def get_bound_kv_ids(self, nano_id: str) -> Set[str]:
        """Get the set of KV IDs bound to a Nano Model."""
        nano = self.nanos.get(nano_id)
        if nano:
            return nano.bound_kv_ids.copy()
        return set()
    
    def export_state(self) -> Dict[str, Any]:
        """Export registry state for persistence."""
        return {
            "nanos": {
                nano_id: nano.to_dict()
                for nano_id, nano in self.nanos.items()
            },
            "statistics": {
                "total_registered": self.total_registered,
                "total_deprecated": self.total_deprecated,
                "total_destroyed": self.total_destroyed,
            },
            "event_log": self.event_log[-100:],  # Last 100 events
        }
    
    def import_state(self, state: Dict[str, Any]):
        """Import registry state from persistence."""
        self.nanos.clear()
        
        for nano_id, nano_data in state.get("nanos", {}).items():
            nano = NanoModel.from_dict(nano_data)
            self.nanos[nano_id] = nano
        
        stats = state.get("statistics", {})
        self.total_registered = stats.get("total_registered", len(self.nanos))
        self.total_deprecated = stats.get("total_deprecated", 0)
        self.total_destroyed = stats.get("total_destroyed", 0)
        
        self.event_log = state.get("event_log", [])
        
        logger.info(f"Imported {len(self.nanos)} Nano Models from state")
