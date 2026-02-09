"""
Nano Model Factory
==================

Factory for creating Nano Models via LoRA fine-tuning.

Handles:
- Adaptive sample collection for innovation gaps
- LoRA training with regularization
- KV binding and registration
- Creation mode selection (emergency/few_shot/standard/confident)

Implements Iron Law 3 (Innovation Triggering): Nano Models created only
when genuine innovation gaps are detected.
"""

from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import logging
import hashlib

from .models import (
    NanoModel, NanoLifecycleState, KVEntry, LoRAWeights, InnovationSample
)
from .registry import NanoRegistry

logger = logging.getLogger(__name__)


class CreationMode(Enum):
    """Nano Model creation modes based on sample availability."""
    EMERGENCY = "emergency"    # 1-5 samples, high risk tolerance
    FEW_SHOT = "few_shot"      # 5-20 samples, high diversity
    STANDARD = "standard"      # 20-50 samples, normal
    CONFIDENT = "confident"    # 50-200 samples, high reliability


@dataclass
class CreationConfig:
    """Configuration for Nano Model creation."""
    mode: CreationMode
    min_samples: int
    max_samples: int
    lora_rank: int
    lora_alpha: float
    dropout: float
    regularization_strength: float
    initial_state: NanoLifecycleState
    
    @classmethod
    def for_mode(cls, mode: CreationMode) -> "CreationConfig":
        """Get configuration for a specific creation mode."""
        configs = {
            CreationMode.EMERGENCY: cls(
                mode=CreationMode.EMERGENCY,
                min_samples=1,
                max_samples=5,
                lora_rank=4,
                lora_alpha=16.0,
                dropout=0.1,
                regularization_strength=0.1,
                initial_state=NanoLifecycleState.TRIAL,
            ),
            CreationMode.FEW_SHOT: cls(
                mode=CreationMode.FEW_SHOT,
                min_samples=5,
                max_samples=20,
                lora_rank=4,
                lora_alpha=16.0,
                dropout=0.1,
                regularization_strength=0.05,
                initial_state=NanoLifecycleState.TRIAL,
            ),
            CreationMode.STANDARD: cls(
                mode=CreationMode.STANDARD,
                min_samples=20,
                max_samples=50,
                lora_rank=8,
                lora_alpha=32.0,
                dropout=0.05,
                regularization_strength=0.01,
                initial_state=NanoLifecycleState.ACTIVE,
            ),
            CreationMode.CONFIDENT: cls(
                mode=CreationMode.CONFIDENT,
                min_samples=50,
                max_samples=200,
                lora_rank=8,
                lora_alpha=32.0,
                dropout=0.05,
                regularization_strength=0.005,
                initial_state=NanoLifecycleState.ACTIVE,
            ),
        }
        return configs[mode]


class NanoModelFactory:
    """
    Factory for creating Nano Models via LoRA fine-tuning.
    
    Handles:
    - Sample collection for innovation gaps
    - LoRA training on collected samples
    - KV binding and registration
    
    Adaptive Creation Modes:
    - Emergency: 1-5 samples, high risk tolerance, TRIAL state
    - Few-shot: 5-20 samples, high diversity, TRIAL state
    - Standard: 20-50 samples, normal, ACTIVE state
    - Confident: 50-200 samples, high reliability, ACTIVE state
    """
    
    def __init__(
        self,
        registry: NanoRegistry,
        hidden_dim: int = 4096,
        default_lora_rank: int = 8,
        default_lora_alpha: float = 32.0,
        target_modules: Optional[List[str]] = None,
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        buffer_threshold: int = 50,
        auto_create: bool = True,
    ):
        """
        Initialize the Nano Model Factory.
        
        Args:
            registry: NanoRegistry for registration
            hidden_dim: Hidden dimension of the model
            default_lora_rank: Default LoRA rank
            default_lora_alpha: Default LoRA alpha
            target_modules: Modules to apply LoRA to
            learning_rate: Learning rate for LoRA training
            num_epochs: Number of training epochs
            buffer_threshold: Samples needed for standard creation
            auto_create: Whether to auto-create when buffer is full
        """
        self.registry = registry
        self.hidden_dim = hidden_dim
        self.default_lora_rank = default_lora_rank
        self.default_lora_alpha = default_lora_alpha
        self.target_modules = target_modules or ["q_proj", "v_proj", "fc1", "fc2"]
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.buffer_threshold = buffer_threshold
        self.auto_create = auto_create
        
        # Sample buffer for innovation gaps
        self.sample_buffer: List[InnovationSample] = []
        
        # Creation statistics
        self.total_created = 0
        self.creation_history: List[Dict[str, Any]] = []
        
        logger.info(
            f"NanoModelFactory initialized: "
            f"hidden_dim={hidden_dim}, lora_rank={default_lora_rank}"
        )
    
    def collect_sample(
        self,
        query: np.ndarray,
        target_output: np.ndarray,
        innovation_score: float,
        innovation_components: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Collect a sample demonstrating an innovation gap.
        
        Samples are buffered until threshold is reached.
        
        Args:
            query: Query embedding
            target_output: Target output embedding
            innovation_score: Innovation score from detector
            innovation_components: Component scores (proj_error, struct_novelty, kv_coverage)
            metadata: Additional metadata
            
        Returns:
            True if a Nano Model was created
        """
        sample = InnovationSample(
            query=query,
            target_output=target_output,
            innovation_score=innovation_score,
            innovation_components=innovation_components or {},
            metadata=metadata or {},
        )
        
        self.sample_buffer.append(sample)
        
        logger.debug(
            f"Collected sample {sample.sample_id}, "
            f"buffer size: {len(self.sample_buffer)}"
        )
        
        # Check if we should create a new Nano Model
        if self.auto_create and len(self.sample_buffer) >= self.buffer_threshold:
            nano = self.create_from_buffer()
            return nano is not None
        
        return False
    
    def determine_creation_mode(
        self,
        num_samples: int,
        avg_innovation_score: float,
        sample_diversity: float,
        risk_tolerance: float = 0.5,
    ) -> CreationMode:
        """
        Determine the appropriate creation mode based on conditions.
        
        Args:
            num_samples: Number of available samples
            avg_innovation_score: Average innovation score
            sample_diversity: Diversity of samples (0-1)
            risk_tolerance: Risk tolerance level (0-1)
            
        Returns:
            Appropriate CreationMode
        """
        # Emergency mode: very few samples but high innovation score
        if num_samples <= 5 and avg_innovation_score > 0.9 and risk_tolerance > 0.7:
            return CreationMode.EMERGENCY
        
        # Few-shot mode: limited samples but high diversity
        if num_samples <= 20 and (avg_innovation_score > 0.8 or sample_diversity > 0.7):
            return CreationMode.FEW_SHOT
        
        # Confident mode: many samples
        if num_samples >= 50:
            return CreationMode.CONFIDENT
        
        # Standard mode: default
        return CreationMode.STANDARD
    
    def compute_sample_diversity(self, samples: List[InnovationSample]) -> float:
        """
        Compute diversity of samples based on query embeddings.
        
        Args:
            samples: List of innovation samples
            
        Returns:
            Diversity score in [0, 1]
        """
        if len(samples) < 2:
            return 0.0
        
        # Compute pairwise cosine distances
        queries = np.array([s.query.flatten() for s in samples])
        
        # Normalize
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        queries_normalized = queries / norms
        
        # Compute similarity matrix
        similarity_matrix = queries_normalized @ queries_normalized.T
        
        # Average off-diagonal similarity
        n = len(samples)
        mask = ~np.eye(n, dtype=bool)
        avg_similarity = similarity_matrix[mask].mean()
        
        # Diversity = 1 - similarity
        return float(1 - avg_similarity)
    
    def create_from_buffer(
        self,
        force_mode: Optional[CreationMode] = None,
        innovation_domain: str = "",
    ) -> Optional[NanoModel]:
        """
        Create a Nano Model from buffered samples.
        
        Args:
            force_mode: Force a specific creation mode
            innovation_domain: Description of the innovation domain
            
        Returns:
            Created NanoModel or None if creation failed
        """
        if not self.sample_buffer:
            logger.warning("No samples in buffer for Nano creation")
            return None
        
        # Determine creation mode
        avg_score = np.mean([s.innovation_score for s in self.sample_buffer])
        diversity = self.compute_sample_diversity(self.sample_buffer)
        
        mode = force_mode or self.determine_creation_mode(
            num_samples=len(self.sample_buffer),
            avg_innovation_score=avg_score,
            sample_diversity=diversity,
        )
        
        config = CreationConfig.for_mode(mode)
        
        # Select samples based on mode
        num_samples = min(len(self.sample_buffer), config.max_samples)
        samples = self.sample_buffer[:num_samples]
        self.sample_buffer = self.sample_buffer[num_samples:]
        
        logger.info(
            f"Creating Nano Model: mode={mode.value}, "
            f"samples={len(samples)}, diversity={diversity:.3f}"
        )
        
        # Create Nano Model
        nano = self._create_nano(samples, config, innovation_domain)
        
        if nano:
            # Register
            success = self.registry.register(nano)
            if success:
                self.total_created += 1
                self._log_creation(nano, config, samples)
                return nano
            else:
                logger.error(f"Failed to register Nano Model {nano.nano_id}")
        
        return None
    
    def create_emergency(
        self,
        samples: List[InnovationSample],
        innovation_domain: str = "",
    ) -> Optional[NanoModel]:
        """
        Create a Nano Model in emergency mode (1-5 samples).
        
        Args:
            samples: Innovation samples (1-5)
            innovation_domain: Description of the innovation domain
            
        Returns:
            Created NanoModel or None
        """
        if not 1 <= len(samples) <= 5:
            logger.warning(f"Emergency mode requires 1-5 samples, got {len(samples)}")
            return None
        
        config = CreationConfig.for_mode(CreationMode.EMERGENCY)
        nano = self._create_nano(samples, config, innovation_domain)
        
        if nano and self.registry.register(nano):
            self.total_created += 1
            self._log_creation(nano, config, samples)
            return nano
        
        return None
    
    def _create_nano(
        self,
        samples: List[InnovationSample],
        config: CreationConfig,
        innovation_domain: str,
    ) -> Optional[NanoModel]:
        """
        Internal method to create a Nano Model.
        
        Args:
            samples: Training samples
            config: Creation configuration
            innovation_domain: Domain description
            
        Returns:
            Created NanoModel or None
        """
        try:
            # Generate unique ID
            nano_id = self._generate_nano_id(samples)
            
            # Train LoRA weights
            lora_weights = self._train_lora(samples, config)
            
            # Create KV entries and get IDs
            kv_ids = self._create_bound_kvs(samples, nano_id)
            
            # Compute confidence score
            confidence = self._compute_confidence(samples, config)
            
            # Create Nano Model
            nano = NanoModel(
                nano_id=nano_id,
                lora_weights=lora_weights,
                bound_kv_ids=kv_ids,
                state=config.initial_state,
                created_at=datetime.now(),
                last_activated=datetime.now(),
                creation_mode=config.mode.value,
                confidence_score=confidence,
                innovation_domain=innovation_domain,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=self.target_modules,
                metadata={
                    "num_samples": len(samples),
                    "avg_innovation_score": np.mean([s.innovation_score for s in samples]),
                    "sample_diversity": self.compute_sample_diversity(samples),
                },
            )
            
            logger.info(
                f"Created Nano Model {nano_id}: "
                f"mode={config.mode.value}, confidence={confidence:.3f}"
            )
            
            return nano
            
        except Exception as e:
            logger.error(f"Failed to create Nano Model: {e}")
            return None
    
    def _generate_nano_id(self, samples: List[InnovationSample]) -> str:
        """Generate unique Nano Model ID."""
        # Hash based on sample content and timestamp
        content = "".join(s.sample_id for s in samples)
        content += datetime.now().isoformat()
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"nano_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash_val}"
    
    def _train_lora(
        self,
        samples: List[InnovationSample],
        config: CreationConfig,
    ) -> Dict[str, LoRAWeights]:
        """
        Train LoRA weights on innovation samples.
        
        This is a simplified simulation of LoRA training.
        In production, this would use actual gradient descent.
        
        Args:
            samples: Training samples
            config: Creation configuration
            
        Returns:
            Dictionary of layer name -> LoRAWeights
        """
        lora_weights = {}
        
        # Stack queries and targets
        queries = np.array([s.query.flatten() for s in samples])
        targets = np.array([s.target_output.flatten() for s in samples])
        
        # Determine dimensions
        input_dim = queries.shape[1] if queries.ndim > 1 else self.hidden_dim
        output_dim = targets.shape[1] if targets.ndim > 1 else self.hidden_dim
        
        for module_name in self.target_modules:
            # Initialize LoRA matrices
            # A: [r, input_dim] - down projection
            # B: [output_dim, r] - up projection
            
            r = config.lora_rank
            
            # Xavier initialization
            A = np.random.randn(r, input_dim) * np.sqrt(2.0 / (r + input_dim))
            B = np.zeros((output_dim, r))  # Initialize B to zero (standard LoRA)
            
            # Simulate training: compute pseudo-gradient update
            # In real implementation, this would be actual gradient descent
            for epoch in range(self.num_epochs):
                for query, target in zip(queries, targets):
                    # Forward pass simulation
                    query_flat = query.flatten()[:input_dim]
                    target_flat = target.flatten()[:output_dim]
                    
                    # Compute delta (simplified)
                    h = A @ query_flat  # [r]
                    output = B @ h  # [output_dim]
                    
                    # Error
                    error = target_flat - output
                    
                    # Gradient update (simplified)
                    grad_B = np.outer(error, h) * self.lr
                    grad_A = np.outer(B.T @ error, query_flat) * self.lr
                    
                    # Apply regularization
                    grad_B -= config.regularization_strength * B
                    grad_A -= config.regularization_strength * A
                    
                    # Update
                    B += grad_B
                    A += grad_A
            
            # Apply dropout (zero out some weights)
            if config.dropout > 0:
                mask_A = np.random.random(A.shape) > config.dropout
                mask_B = np.random.random(B.shape) > config.dropout
                A = A * mask_A / (1 - config.dropout)
                B = B * mask_B / (1 - config.dropout)
            
            lora_weights[module_name] = LoRAWeights(
                layer_name=module_name,
                A=A,
                B=B,
                alpha=config.lora_alpha,
                rank=config.lora_rank,
            )
        
        return lora_weights
    
    def _create_bound_kvs(
        self,
        samples: List[InnovationSample],
        nano_id: str,
    ) -> Set[str]:
        """
        Create KV entry IDs bound to the new Nano Model.
        
        Note: Actual KV entries are created separately in the KV store.
        This method just generates the binding IDs.
        
        Args:
            samples: Training samples
            nano_id: ID of the Nano Model
            
        Returns:
            Set of KV entry IDs
        """
        kv_ids = set()
        
        for i, sample in enumerate(samples):
            kv_id = f"{nano_id}_kv_{i}"
            kv_ids.add(kv_id)
        
        return kv_ids
    
    def _compute_confidence(
        self,
        samples: List[InnovationSample],
        config: CreationConfig,
    ) -> float:
        """
        Compute confidence score for the Nano Model.
        
        Based on:
        - Number of samples
        - Sample diversity
        - Average innovation score consistency
        
        Args:
            samples: Training samples
            config: Creation configuration
            
        Returns:
            Confidence score in [0, 1]
        """
        # Base confidence from sample count
        sample_factor = min(len(samples) / config.max_samples, 1.0)
        
        # Diversity factor
        diversity = self.compute_sample_diversity(samples)
        
        # Consistency factor (low variance in innovation scores)
        scores = [s.innovation_score for s in samples]
        if len(scores) > 1:
            consistency = 1 - min(np.std(scores), 0.5) * 2
        else:
            consistency = 0.5
        
        # Weighted combination
        confidence = 0.4 * sample_factor + 0.3 * diversity + 0.3 * consistency
        
        # Adjust for creation mode
        mode_factors = {
            CreationMode.EMERGENCY: 0.6,
            CreationMode.FEW_SHOT: 0.7,
            CreationMode.STANDARD: 0.85,
            CreationMode.CONFIDENT: 1.0,
        }
        
        return confidence * mode_factors[config.mode]
    
    def _log_creation(
        self,
        nano: NanoModel,
        config: CreationConfig,
        samples: List[InnovationSample],
    ):
        """Log Nano Model creation event."""
        self.creation_history.append({
            "timestamp": datetime.now().isoformat(),
            "nano_id": nano.nano_id,
            "mode": config.mode.value,
            "num_samples": len(samples),
            "confidence": nano.confidence_score,
            "innovation_domain": nano.innovation_domain,
            "avg_innovation_score": np.mean([s.innovation_score for s in samples]),
        })
        
        # Keep only last 100 entries
        if len(self.creation_history) > 100:
            self.creation_history = self.creation_history[-100:]
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status."""
        if not self.sample_buffer:
            return {
                "buffer_size": 0,
                "threshold": self.buffer_threshold,
                "ready_for_creation": False,
            }
        
        return {
            "buffer_size": len(self.sample_buffer),
            "threshold": self.buffer_threshold,
            "ready_for_creation": len(self.sample_buffer) >= self.buffer_threshold,
            "avg_innovation_score": np.mean([s.innovation_score for s in self.sample_buffer]),
            "sample_diversity": self.compute_sample_diversity(self.sample_buffer),
            "oldest_sample": self.sample_buffer[0].collected_at.isoformat(),
            "newest_sample": self.sample_buffer[-1].collected_at.isoformat(),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get factory statistics."""
        return {
            "total_created": self.total_created,
            "buffer_size": len(self.sample_buffer),
            "buffer_threshold": self.buffer_threshold,
            "creation_history_size": len(self.creation_history),
            "recent_creations": self.creation_history[-5:] if self.creation_history else [],
        }
    
    def clear_buffer(self):
        """Clear the sample buffer."""
        count = len(self.sample_buffer)
        self.sample_buffer.clear()
        logger.info(f"Cleared {count} samples from buffer")
