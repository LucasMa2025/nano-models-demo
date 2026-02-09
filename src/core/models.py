"""
Core Data Models for Nano Models Framework
==========================================

Implements:
- NanoLifecycleState: Lifecycle state enumeration
- KVEntry: Versioned key-value entry with Nano Model binding
- NanoModel: Innovation-triggered derivation patch
- NanoModelDiagnostics: Diagnostic information container
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from datetime import datetime
import numpy as np
import json
import hashlib


class NanoLifecycleState(Enum):
    """
    Lifecycle states for Nano Models.
    
    State Transitions:
    CREATE → [TRIAL] → ACTIVE → DORMANT → DEPRECATED → DESTROY
    
    - TRIAL: Low-confidence validation period (for emergency/few-shot creation)
    - ACTIVE: Normal operation state
    - DORMANT: Low usage, can be reactivated
    - DEPRECATED: Marked for removal
    """
    TRIAL = "trial"
    ACTIVE = "active"
    DORMANT = "dormant"
    DEPRECATED = "deprecated"


class KVAccessMode(Enum):
    """Access modes for KV entries."""
    EXCLUSIVE = "exclusive"  # Single Nano Model read-write
    SHARED = "shared"        # Multiple Nano Models read, collaborative write
    GLOBAL = "global"        # All Nano Models read-only


@dataclass
class KVEntry:
    """
    A versioned key-value entry with Nano Model binding.
    
    Supports:
    - Version control for rollback
    - Write locking for shared KV
    - Conflict detection via semantic similarity
    
    Attributes:
        key: Embedding vector for matching queries
        value: Associated value embedding
        nano_id: Bound Nano Model ID (None = global)
        access_mode: Access permission level
        created_at: Creation timestamp
        version: Current version number
        access_count: Number of times accessed
    """
    key: np.ndarray                          # [bottleneck_dim]
    value: np.ndarray                        # [hidden_dim]
    nano_id: Optional[str] = None            # Bound Nano Model ID
    access_mode: str = "exclusive"           # 'exclusive', 'shared', 'global'
    created_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    access_count: int = 0
    kv_id: str = field(default_factory=lambda: "")
    
    # Version history for rollback: List of (version, value, timestamp)
    version_history: List[Tuple[int, np.ndarray, datetime]] = field(
        default_factory=list
    )
    write_lock: Optional[str] = None  # nano_id holding lock
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate KV ID if not provided."""
        if not self.kv_id:
            self.kv_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID based on key content."""
        key_bytes = self.key.tobytes() if isinstance(self.key, np.ndarray) else str(self.key).encode()
        return f"kv_{hashlib.md5(key_bytes).hexdigest()[:12]}"
    
    def hit(self, query: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Check if this KV entry matches the query.
        
        Args:
            query: Query embedding vector
            threshold: Cosine similarity threshold
            
        Returns:
            True if similarity exceeds threshold
        """
        similarity = self.compute_similarity(query)
        return similarity > threshold
    
    def compute_similarity(self, query: np.ndarray) -> float:
        """Compute cosine similarity between query and key."""
        key_flat = self.key.flatten()
        query_flat = query.flatten()
        
        norm_key = np.linalg.norm(key_flat)
        norm_query = np.linalg.norm(query_flat)
        
        if norm_key < 1e-8 or norm_query < 1e-8:
            return 0.0
        
        return float(np.dot(key_flat, query_flat) / (norm_key * norm_query))
    
    def update_value(
        self,
        new_value: np.ndarray,
        nano_id: str,
        force: bool = False,
    ) -> Tuple[bool, float]:
        """
        Version-controlled update with conflict detection.
        
        Args:
            new_value: New value to set
            nano_id: ID of Nano Model requesting update
            force: Force update even with high conflict
            
        Returns:
            Tuple of (success, conflict_score)
        """
        # Check permissions - exclusive mode requires matching nano_id
        if self.access_mode == 'exclusive':
            if self.nano_id is not None and self.nano_id != nano_id:
                return False, 1.0
        if self.access_mode == 'global':
            return False, 1.0  # Global KV is read-only
        
        # Check write lock
        if self.write_lock is not None and self.write_lock != nano_id:
            return False, 1.0
        
        # Detect semantic conflict
        conflict_score = self._compute_conflict(new_value)
        
        if conflict_score > 0.5 and not force:
            return False, conflict_score
        
        # Save to history
        self.version_history.append(
            (self.version, self.value.copy(), datetime.now())
        )
        
        # Update
        self.value = new_value
        self.version += 1
        return True, conflict_score
    
    def _compute_conflict(self, new_value: np.ndarray) -> float:
        """Compute conflict score between current and new value."""
        old_flat = self.value.flatten()
        new_flat = new_value.flatten()
        
        norm_old = np.linalg.norm(old_flat)
        norm_new = np.linalg.norm(new_flat)
        
        if norm_old < 1e-8 or norm_new < 1e-8:
            return 0.0
        
        similarity = np.dot(old_flat, new_flat) / (norm_old * norm_new)
        return float(1 - similarity)
    
    def rollback(self, target_version: int) -> bool:
        """
        Rollback to a specific version.
        
        Args:
            target_version: Version number to rollback to
            
        Returns:
            True if rollback succeeded
        """
        for ver, val, _ in reversed(self.version_history):
            if ver == target_version:
                self.value = val.copy()
                self.version = ver
                return True
        return False
    
    def acquire_lock(self, nano_id: str) -> bool:
        """Acquire write lock for shared KV."""
        if self.access_mode != 'shared':
            return False
        if self.write_lock is None:
            self.write_lock = nano_id
            return True
        return self.write_lock == nano_id
    
    def release_lock(self, nano_id: str) -> bool:
        """Release write lock."""
        if self.write_lock == nano_id:
            self.write_lock = None
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "kv_id": self.kv_id,
            "key": self.key.tolist(),
            "value": self.value.tolist(),
            "nano_id": self.nano_id,
            "access_mode": self.access_mode,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KVEntry":
        """Deserialize from dictionary."""
        return cls(
            kv_id=data["kv_id"],
            key=np.array(data["key"]),
            value=np.array(data["value"]),
            nano_id=data.get("nano_id"),
            access_mode=data.get("access_mode", "exclusive"),
            created_at=datetime.fromisoformat(data["created_at"]),
            version=data.get("version", 1),
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LoRAWeights:
    """
    LoRA weight container for a single layer.
    
    LoRA decomposes weight updates as: ΔW = B @ A
    where B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d, k)
    """
    layer_name: str
    A: np.ndarray  # [r, k] - down projection
    B: np.ndarray  # [d, r] - up projection
    alpha: float = 32.0
    rank: int = 8
    
    @property
    def delta_w(self) -> np.ndarray:
        """Compute the weight delta: ΔW = (alpha/rank) * B @ A"""
        return (self.alpha / self.rank) * (self.B @ self.A)
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply LoRA transformation: x @ ΔW^T"""
        return x @ self.delta_w.T
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "layer_name": self.layer_name,
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "alpha": self.alpha,
            "rank": self.rank,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoRAWeights":
        """Deserialize from dictionary."""
        return cls(
            layer_name=data["layer_name"],
            A=np.array(data["A"]),
            B=np.array(data["B"]),
            alpha=data.get("alpha", 32.0),
            rank=data.get("rank", 8),
        )


@dataclass
class NanoModel:
    """
    A Nano Model: innovation-triggered derivation patch.
    
    Once created, the lora_weights are FROZEN and cannot be modified.
    The bound_kv_ids define which KV entries this Nano Model can access.
    
    Implements Iron Law 1 (Immutability): Weights frozen post-creation.
    Implements Iron Law 2 (KV Binding): Exclusive KV access.
    
    Attributes:
        nano_id: Unique identifier
        lora_weights: Dictionary of layer name -> LoRAWeights
        bound_kv_ids: Set of bound KV entry IDs
        state: Current lifecycle state
        created_at: Creation timestamp
        last_activated: Last activation timestamp
        activation_count: Total activation count
        creation_mode: How the Nano was created (emergency/few_shot/standard/confident)
        confidence_score: Confidence in the Nano Model (0-1)
        innovation_domain: Description of the innovation domain
    """
    nano_id: str
    lora_weights: Dict[str, LoRAWeights]
    bound_kv_ids: Set[str]
    state: NanoLifecycleState
    created_at: datetime
    last_activated: datetime
    activation_count: int = 0
    
    # Creation metadata
    creation_mode: str = "standard"  # emergency, few_shot, standard, confident
    confidence_score: float = 0.8
    innovation_domain: str = ""
    
    # Hyperparameters used during creation
    lora_rank: int = 8
    lora_alpha: float = 32.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "fc1", "fc2"
    ])
    
    # Performance tracking
    success_count: int = 0
    failure_count: int = 0
    total_contribution: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    _frozen: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        """Freeze weights after creation (Iron Law 1)."""
        self._frozen = True
    
    def is_frozen(self) -> bool:
        """Check if the Nano Model is frozen."""
        return self._frozen
    
    def apply(self, hidden_states: np.ndarray, layer_name: str) -> np.ndarray:
        """
        Apply Nano Model's LoRA weights to hidden states.
        
        Args:
            hidden_states: Input hidden states [batch, seq, hidden]
            layer_name: Name of the layer to apply LoRA to
            
        Returns:
            Modified hidden states
        """
        if layer_name not in self.lora_weights:
            return hidden_states
        
        lora = self.lora_weights[layer_name]
        return hidden_states + lora.apply(hidden_states)
    
    def record_activation(self, success: bool, contribution: float):
        """Record an activation event."""
        self.last_activated = datetime.now()
        self.activation_count += 1
        self.total_contribution += contribution
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    @property
    def success_rate(self) -> float:
        """Compute success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Default
        return self.success_count / total
    
    @property
    def average_contribution(self) -> float:
        """Compute average contribution per activation."""
        if self.activation_count == 0:
            return 0.0
        return self.total_contribution / self.activation_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "nano_id": self.nano_id,
            "lora_weights": {
                name: lora.to_dict() 
                for name, lora in self.lora_weights.items()
            },
            "bound_kv_ids": list(self.bound_kv_ids),
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "last_activated": self.last_activated.isoformat(),
            "activation_count": self.activation_count,
            "creation_mode": self.creation_mode,
            "confidence_score": self.confidence_score,
            "innovation_domain": self.innovation_domain,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_contribution": self.total_contribution,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NanoModel":
        """Deserialize from dictionary."""
        return cls(
            nano_id=data["nano_id"],
            lora_weights={
                name: LoRAWeights.from_dict(lora_data)
                for name, lora_data in data["lora_weights"].items()
            },
            bound_kv_ids=set(data["bound_kv_ids"]),
            state=NanoLifecycleState(data["state"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activated=datetime.fromisoformat(data["last_activated"]),
            activation_count=data.get("activation_count", 0),
            creation_mode=data.get("creation_mode", "standard"),
            confidence_score=data.get("confidence_score", 0.8),
            innovation_domain=data.get("innovation_domain", ""),
            lora_rank=data.get("lora_rank", 8),
            lora_alpha=data.get("lora_alpha", 32.0),
            target_modules=data.get("target_modules", ["q_proj", "v_proj", "fc1", "fc2"]),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            total_contribution=data.get("total_contribution", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class NanoModelDiagnostics:
    """
    Diagnostic information from Nano Model inference.
    
    Provides detailed insights into the inference process for
    debugging, monitoring, and feedback collection.
    """
    # Selection info
    selected_nanos: List[str]
    kv_hit_scores: Dict[str, float]
    fusion_weights: Dict[str, float]
    
    # Innovation detection
    innovation_score: float
    innovation_components: Dict[str, float]  # projection_error, structure_novelty, kv_coverage
    
    # Fusion info
    total_nano_contribution: float
    conflict_detected: bool = False
    conflict_scores: Dict[str, float] = field(default_factory=dict)
    fusion_method: str = "weighted_average"  # or "winner_takes_all"
    
    # Performance
    inference_time_ms: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    query_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "selected_nanos": self.selected_nanos,
            "kv_hit_scores": self.kv_hit_scores,
            "fusion_weights": self.fusion_weights,
            "innovation_score": self.innovation_score,
            "innovation_components": self.innovation_components,
            "total_nano_contribution": self.total_nano_contribution,
            "conflict_detected": self.conflict_detected,
            "conflict_scores": self.conflict_scores,
            "fusion_method": self.fusion_method,
            "inference_time_ms": self.inference_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "query_hash": self.query_hash,
        }


@dataclass
class InnovationSample:
    """
    A sample demonstrating an innovation gap.
    
    Used for collecting training data for Nano Model creation.
    """
    query: np.ndarray
    target_output: np.ndarray
    innovation_score: float
    innovation_components: Dict[str, float]
    collected_at: datetime = field(default_factory=datetime.now)
    sample_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.sample_id:
            self.sample_id = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sample_id": self.sample_id,
            "query": self.query.tolist(),
            "target_output": self.target_output.tolist(),
            "innovation_score": self.innovation_score,
            "innovation_components": self.innovation_components,
            "collected_at": self.collected_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InnovationSample":
        """Deserialize from dictionary."""
        return cls(
            sample_id=data["sample_id"],
            query=np.array(data["query"]),
            target_output=np.array(data["target_output"]),
            innovation_score=data["innovation_score"],
            innovation_components=data.get("innovation_components", {}),
            collected_at=datetime.fromisoformat(data["collected_at"]),
            metadata=data.get("metadata", {}),
        )
