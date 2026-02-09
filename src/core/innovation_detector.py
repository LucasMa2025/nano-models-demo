"""
Innovation Detection Mechanism
==============================

Implements the three-stage innovation detection mechanism:
1. Representation Space Projection Error
2. Attention Pattern Anomaly Detection (MMD-based)
3. Knowledge Coverage Check

Avoids circular dependency by not requiring generation for consistency checking.

Key Features:
- Online learning of semantic subspaces via Incremental PCA
- Reference pattern maintenance with forgetting mechanism
- Configurable weights for each detection stage
"""

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
import logging

from .models import KVEntry

logger = logging.getLogger(__name__)


@dataclass
class InnovationDetectionResult:
    """Result of innovation detection."""
    is_innovation: bool
    score: float
    projection_error: float
    structure_novelty: float
    kv_coverage: float
    details: Dict[str, Any] = field(default_factory=dict)


class InnovationDetector:
    """
    Improved innovation detector avoiding circular dependency.
    
    Three-stage detection (no generation required):
    1. Representation projection error (outside known semantic spaces)
    2. Attention pattern anomaly (novel reasoning structure needed)
    3. Knowledge coverage (existing KV insufficient)
    
    Key improvement: Online learning of semantic subspaces and
    reference patterns with forgetting mechanism.
    
    Mathematical Foundation:
    InnovationScore(q) = w1 * ProjectionError(h_q) 
                       + w2 * StructureNovelty(A_q) 
                       + w3 * (1 - KVCoverage(q))
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        num_subspaces: int = 64,
        projection_weight: float = 0.4,
        structure_weight: float = 0.4,
        coverage_weight: float = 0.2,
        innovation_threshold: float = 0.7,
        buffer_size: int = 10000,
        pattern_decay: float = 0.95,
        max_patterns: int = 200,
        min_samples_for_pca: int = 100,
    ):
        """
        Initialize the Innovation Detector.
        
        Args:
            hidden_dim: Dimension of hidden states
            num_subspaces: Number of PCA components for semantic subspaces
            projection_weight: Weight for projection error (w1)
            structure_weight: Weight for structure novelty (w2)
            coverage_weight: Weight for KV coverage (w3)
            innovation_threshold: Threshold for innovation detection (τ)
            buffer_size: Maximum size of representation buffer
            pattern_decay: Exponential decay factor for reference patterns
            max_patterns: Maximum number of reference patterns to maintain
            min_samples_for_pca: Minimum samples before PCA initialization
        """
        self.hidden_dim = hidden_dim
        self.num_subspaces = num_subspaces
        self.w1 = projection_weight
        self.w2 = structure_weight
        self.w3 = coverage_weight
        self.tau = innovation_threshold
        self.pattern_decay = pattern_decay
        self.max_patterns = max_patterns
        self.min_samples_for_pca = min_samples_for_pca
        
        # Online learning components
        self.representation_buffer: deque = deque(maxlen=buffer_size)
        self.incremental_pca = IncrementalPCA(n_components=min(num_subspaces, hidden_dim))
        self.subspace_initialized = False
        self.pca_sample_count = 0
        
        # Reference patterns with weights (for decay)
        self.reference_patterns: List[np.ndarray] = []
        self.pattern_weights: List[float] = []
        
        # Statistics
        self.detection_count = 0
        self.innovation_count = 0
        self.last_detection_time: Optional[datetime] = None
        
        logger.info(
            f"InnovationDetector initialized: "
            f"threshold={innovation_threshold}, "
            f"weights=({projection_weight}, {structure_weight}, {coverage_weight})"
        )
    
    def detect(
        self,
        hidden_states: np.ndarray,
        attention_weights: Optional[np.ndarray],
        kv_store: List[KVEntry],
    ) -> InnovationDetectionResult:
        """
        Detect if an innovation gap exists.
        
        Args:
            hidden_states: Hidden states from base model [batch, seq, hidden] or [seq, hidden]
            attention_weights: Attention weights [batch, heads, seq, seq] or None
            kv_store: List of KV entries to check coverage
            
        Returns:
            InnovationDetectionResult with detection outcome and scores
        """
        self.detection_count += 1
        self.last_detection_time = datetime.now()
        
        # Ensure proper shape
        if hidden_states.ndim == 2:
            hidden_states = hidden_states[np.newaxis, ...]
        
        # Stage 1: Projection Error
        proj_error = self.compute_projection_error(hidden_states)
        
        # Stage 2: Structure Novelty
        if attention_weights is not None:
            struct_novelty = self.compute_structure_novelty(attention_weights)
        else:
            struct_novelty = 0.5  # Default when attention not available
        
        # Stage 3: KV Coverage
        query_embedding = hidden_states.mean(axis=(0, 1))  # [hidden]
        kv_coverage = self.compute_kv_coverage(query_embedding, kv_store)
        
        # Combined score
        score = (
            self.w1 * proj_error +
            self.w2 * struct_novelty +
            self.w3 * (1 - kv_coverage)
        )
        
        is_innovation = score > self.tau
        
        if is_innovation:
            self.innovation_count += 1
        
        result = InnovationDetectionResult(
            is_innovation=is_innovation,
            score=score,
            projection_error=proj_error,
            structure_novelty=struct_novelty,
            kv_coverage=kv_coverage,
            details={
                "threshold": self.tau,
                "weights": {"w1": self.w1, "w2": self.w2, "w3": self.w3},
                "subspace_initialized": self.subspace_initialized,
                "num_reference_patterns": len(self.reference_patterns),
            }
        )
        
        logger.debug(
            f"Innovation detection: score={score:.4f}, "
            f"is_innovation={is_innovation}, "
            f"components=(proj={proj_error:.4f}, struct={struct_novelty:.4f}, cov={kv_coverage:.4f})"
        )
        
        return result
    
    def update_semantic_subspaces(
        self,
        hidden_states: np.ndarray,
        is_low_entropy: bool = True,
    ):
        """
        Online update of semantic subspaces using Incremental PCA.
        Only update with confident (low-entropy) representations.
        
        Theory: Low-entropy outputs represent known concepts,
        forming the semantic manifold we want to learn.
        
        Args:
            hidden_states: Hidden states to learn from [batch, seq, hidden]
            is_low_entropy: Whether the output was confident (low entropy)
        """
        if not is_low_entropy:
            return  # Only learn from confident outputs
        
        # Ensure proper shape
        if hidden_states.ndim == 2:
            hidden_states = hidden_states[np.newaxis, ...]
        
        # Average over sequence dimension
        h = hidden_states.mean(axis=1)  # [batch, hidden]
        
        # Add to buffer
        for sample in h:
            self.representation_buffer.append(sample)
        
        self.pca_sample_count += len(h)
        
        # Incremental PCA update every min_samples_for_pca samples
        if len(self.representation_buffer) >= self.min_samples_for_pca:
            batch = np.array(list(self.representation_buffer)[-self.min_samples_for_pca:])
            
            try:
                self.incremental_pca.partial_fit(batch)
                self.subspace_initialized = True
                logger.debug(
                    f"Updated semantic subspaces with {len(batch)} samples, "
                    f"total={self.pca_sample_count}"
                )
            except Exception as e:
                logger.warning(f"Failed to update PCA: {e}")
    
    def compute_projection_error(self, hidden_states: np.ndarray) -> float:
        """
        Compute projection error using learned PCA subspaces.
        
        High error = query outside known semantic manifold.
        
        Formula: ProjectionError(h_q) = ||h_q - Proj_S(h_q)|| / ||h_q||
        
        Args:
            hidden_states: Hidden states [batch, seq, hidden]
            
        Returns:
            Normalized projection error in [0, 1]
        """
        if not self.subspace_initialized:
            return 0.5  # Default before initialization
        
        # Average over batch and sequence
        h = hidden_states.mean(axis=(0, 1))  # [hidden]
        h = h.reshape(1, -1)  # [1, hidden]
        
        try:
            # Project to learned subspace and back
            projected = self.incremental_pca.transform(h)
            reconstructed = self.incremental_pca.inverse_transform(projected)
            
            # Reconstruction error (normalized)
            error = np.linalg.norm(h - reconstructed) / (np.linalg.norm(h) + 1e-8)
            return float(min(error, 1.0))
        except Exception as e:
            logger.warning(f"Projection error computation failed: {e}")
            return 0.5
    
    def compute_structure_novelty(self, attention_weights: np.ndarray) -> float:
        """
        Detect attention pattern anomaly using weighted MMD.
        Patterns decay over time to adapt to distribution shift.
        
        Formula: StructureNovelty(A_q) = min_{A_ref} MMD(A_q, A_ref)
        
        Args:
            attention_weights: Attention weights [batch, heads, seq, seq]
            
        Returns:
            Structure novelty score in [0, 1]
        """
        if not self.reference_patterns:
            return 0.5  # Default when no reference patterns
        
        # Flatten attention pattern
        if attention_weights.ndim == 4:
            current = attention_weights.mean(axis=(0, 1)).flatten()  # Average over batch and heads
        elif attention_weights.ndim == 3:
            current = attention_weights.mean(axis=0).flatten()
        else:
            current = attention_weights.flatten()
        
        # Weighted minimum MMD (recent patterns weighted higher)
        weighted_mmds = []
        for ref, weight in zip(self.reference_patterns, self.pattern_weights):
            ref_flat = ref.flatten()
            
            # Ensure same size (truncate or pad)
            min_len = min(len(current), len(ref_flat))
            mmd = self._compute_mmd(current[:min_len], ref_flat[:min_len])
            
            # Weight inversely (higher weight = more important = lower effective MMD)
            weighted_mmds.append(mmd / (weight + 1e-8))
        
        if not weighted_mmds:
            return 0.5
        
        return float(min(min(weighted_mmds), 1.0))
    
    def _compute_mmd(
        self,
        x: np.ndarray,
        y: np.ndarray,
        kernel_bandwidth: float = 1.0,
    ) -> float:
        """
        Compute Maximum Mean Discrepancy with RBF kernel.
        
        MMD measures the distance between two distributions.
        
        Args:
            x: First distribution sample
            y: Second distribution sample
            kernel_bandwidth: RBF kernel bandwidth
            
        Returns:
            MMD value in [0, 1]
        """
        xy_dist = np.sum((x - y) ** 2)
        return float(1 - np.exp(-xy_dist / (2 * kernel_bandwidth ** 2)))
    
    def compute_kv_coverage(
        self,
        query_embedding: np.ndarray,
        kv_store: List[KVEntry],
    ) -> float:
        """
        Compute max similarity to any KV entry.
        
        Formula: KVCoverage(q) = max_{kv} sim(q, kv.key)
        
        Args:
            query_embedding: Query embedding vector
            kv_store: List of KV entries
            
        Returns:
            Maximum similarity score in [0, 1]
        """
        if not kv_store:
            return 0.0
        
        max_sim = 0.0
        for kv in kv_store:
            sim = kv.compute_similarity(query_embedding)
            max_sim = max(max_sim, sim)
        
        return max(0.0, max_sim)
    
    def update_reference_patterns(self, attention_weights: np.ndarray):
        """
        Add successful attention pattern with decay mechanism.
        
        Strategy:
        1. Decay all existing pattern weights
        2. Add new pattern with weight 1.0
        3. Remove patterns with weight < 0.1
        4. Cluster similar patterns to prevent redundancy
        
        Args:
            attention_weights: Attention weights to add as reference
        """
        # Decay existing weights
        self.pattern_weights = [w * self.pattern_decay for w in self.pattern_weights]
        
        # Add new pattern
        if attention_weights.ndim >= 3:
            pattern = attention_weights.mean(axis=tuple(range(attention_weights.ndim - 2)))
        else:
            pattern = attention_weights.copy()
        
        self.reference_patterns.append(pattern)
        self.pattern_weights.append(1.0)
        
        # Remove low-weight patterns
        valid_indices = [i for i, w in enumerate(self.pattern_weights) if w >= 0.1]
        self.reference_patterns = [self.reference_patterns[i] for i in valid_indices]
        self.pattern_weights = [self.pattern_weights[i] for i in valid_indices]
        
        # Cluster if too many patterns
        if len(self.reference_patterns) > self.max_patterns:
            self._cluster_patterns()
        
        logger.debug(f"Updated reference patterns: {len(self.reference_patterns)} patterns")
    
    def _cluster_patterns(self, n_clusters: int = 100):
        """
        Merge similar patterns via clustering.
        
        Args:
            n_clusters: Target number of clusters
        """
        if len(self.reference_patterns) <= n_clusters:
            return
        
        try:
            # Stack and flatten patterns
            patterns = np.array([p.flatten() for p in self.reference_patterns])
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(patterns)
            
            # Keep cluster centroids as new patterns
            new_patterns = []
            new_weights = []
            
            original_shape = self.reference_patterns[0].shape
            
            for i in range(n_clusters):
                mask = labels == i
                if mask.sum() > 0:
                    centroid = patterns[mask].mean(axis=0)
                    new_patterns.append(centroid.reshape(original_shape))
                    # Aggregate weights (max of cluster)
                    cluster_weights = [
                        self.pattern_weights[j] 
                        for j in range(len(labels)) 
                        if labels[j] == i
                    ]
                    new_weights.append(max(cluster_weights))
            
            self.reference_patterns = new_patterns
            self.pattern_weights = new_weights
            
            logger.info(f"Clustered patterns: {len(patterns)} -> {len(new_patterns)}")
            
        except Exception as e:
            logger.warning(f"Pattern clustering failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "detection_count": self.detection_count,
            "innovation_count": self.innovation_count,
            "innovation_rate": self.innovation_count / max(self.detection_count, 1),
            "subspace_initialized": self.subspace_initialized,
            "pca_sample_count": self.pca_sample_count,
            "num_reference_patterns": len(self.reference_patterns),
            "representation_buffer_size": len(self.representation_buffer),
            "last_detection_time": self.last_detection_time.isoformat() if self.last_detection_time else None,
            "threshold": self.tau,
            "weights": {"w1": self.w1, "w2": self.w2, "w3": self.w3},
        }
    
    def reset_statistics(self):
        """Reset detection statistics."""
        self.detection_count = 0
        self.innovation_count = 0
        self.last_detection_time = None
    
    def set_threshold(self, threshold: float):
        """Update innovation threshold."""
        self.tau = threshold
        logger.info(f"Innovation threshold updated to {threshold}")
    
    def set_weights(self, w1: float, w2: float, w3: float):
        """Update detection weights."""
        total = w1 + w2 + w3
        self.w1 = w1 / total
        self.w2 = w2 / total
        self.w3 = w3 / total
        logger.info(f"Detection weights updated: ({self.w1:.2f}, {self.w2:.2f}, {self.w3:.2f})")
