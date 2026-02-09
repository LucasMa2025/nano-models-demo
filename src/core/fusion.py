"""
Conflict-Aware Fusion Mechanism
===============================

Implements the three-step fusion process for multiple Nano Model outputs:
1. Conflict Detection (Enhanced) - Direction + Magnitude
2. Adaptive Conflict Resolution
3. Hierarchical Fusion

Handles scenarios where multiple Nano Models produce conflicting derivations.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np
import logging

from .models import NanoModel

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Result of Nano Model fusion."""
    fused_output: np.ndarray
    fusion_method: str  # "weighted_average" or "winner_takes_all"
    fusion_weights: Dict[str, float]
    conflict_detected: bool
    conflict_scores: Dict[str, float]
    winner_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class ConflictAwareFusion:
    """
    Conflict-aware fusion mechanism for multiple Nano Model outputs.
    
    Three-step fusion process:
    1. Conflict Detection (Enhanced): Combines direction and magnitude
    2. Adaptive Conflict Resolution: Threshold adapts to history
    3. Hierarchical Fusion: Weighted average or winner-takes-all
    
    Conflict Formula:
    Conflict_ij = α * (1 - cos_sim(O_i, O_j)) + (1-α) * |‖O_i‖ - ‖O_j‖| / max(‖O_i‖, ‖O_j‖)
    
    where α = 0.7 balances directional and magnitude conflict.
    """
    
    def __init__(
        self,
        direction_weight: float = 0.7,
        base_conflict_threshold: float = 0.3,
        conflict_history_size: int = 100,
        temperature: float = 1.0,
    ):
        """
        Initialize the Conflict-Aware Fusion mechanism.
        
        Args:
            direction_weight: Weight for directional conflict (α)
            base_conflict_threshold: Base threshold for conflict detection
            conflict_history_size: Size of conflict history for adaptation
            temperature: Temperature for softmax weighting
        """
        self.alpha = direction_weight
        self.base_threshold = base_conflict_threshold
        self.temperature = temperature
        
        # Conflict history for adaptive threshold
        self.conflict_history: deque = deque(maxlen=conflict_history_size)
        
        # Statistics
        self.fusion_count = 0
        self.conflict_count = 0
        self.winner_takes_all_count = 0
        
        logger.info(
            f"ConflictAwareFusion initialized: "
            f"direction_weight={direction_weight}, base_threshold={base_conflict_threshold}"
        )
    
    def fuse(
        self,
        nano_outputs: Dict[str, np.ndarray],
        kv_hit_scores: Dict[str, float],
        base_output: Optional[np.ndarray] = None,
    ) -> FusionResult:
        """
        Fuse multiple Nano Model outputs with conflict awareness.
        
        Args:
            nano_outputs: Dictionary of nano_id -> output array
            kv_hit_scores: Dictionary of nano_id -> KV hit score
            base_output: Base model output for winner selection
            
        Returns:
            FusionResult with fused output and metadata
        """
        self.fusion_count += 1
        
        if not nano_outputs:
            return FusionResult(
                fused_output=np.zeros(1),
                fusion_method="none",
                fusion_weights={},
                conflict_detected=False,
                conflict_scores={},
            )
        
        if len(nano_outputs) == 1:
            nano_id = list(nano_outputs.keys())[0]
            return FusionResult(
                fused_output=nano_outputs[nano_id],
                fusion_method="single",
                fusion_weights={nano_id: 1.0},
                conflict_detected=False,
                conflict_scores={},
            )
        
        # Step 1: Conflict Detection
        conflict_matrix, max_conflict = self._detect_conflicts(nano_outputs)
        
        # Step 2: Adaptive Conflict Resolution
        threshold = self._compute_adaptive_threshold()
        conflict_detected = max_conflict >= threshold
        
        if conflict_detected:
            self.conflict_count += 1
        
        # Record conflict for history
        self.conflict_history.append(max_conflict)
        
        # Step 3: Hierarchical Fusion
        if conflict_detected:
            # Winner-takes-all
            self.winner_takes_all_count += 1
            result = self._winner_takes_all(
                nano_outputs, kv_hit_scores, base_output, conflict_matrix
            )
        else:
            # Weighted average
            result = self._weighted_average(nano_outputs, kv_hit_scores)
        
        # Add conflict info to result
        result.conflict_detected = conflict_detected
        result.conflict_scores = self._format_conflict_scores(conflict_matrix, nano_outputs)
        result.details["max_conflict"] = max_conflict
        result.details["threshold"] = threshold
        
        logger.debug(
            f"Fusion: method={result.fusion_method}, "
            f"conflict={conflict_detected}, max_conflict={max_conflict:.4f}"
        )
        
        return result
    
    def _detect_conflicts(
        self,
        nano_outputs: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        """
        Detect pairwise conflicts between Nano Model outputs.
        
        Enhanced conflict formula:
        Conflict_ij = α * (1 - cos_sim(O_i, O_j)) + (1-α) * magnitude_diff
        
        Args:
            nano_outputs: Dictionary of nano_id -> output array
            
        Returns:
            Tuple of (conflict_matrix, max_conflict)
        """
        nano_ids = list(nano_outputs.keys())
        n = len(nano_ids)
        conflict_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                output_i = nano_outputs[nano_ids[i]].flatten()
                output_j = nano_outputs[nano_ids[j]].flatten()
                
                # Ensure same length
                min_len = min(len(output_i), len(output_j))
                output_i = output_i[:min_len]
                output_j = output_j[:min_len]
                
                # Directional conflict (1 - cosine similarity)
                norm_i = np.linalg.norm(output_i)
                norm_j = np.linalg.norm(output_j)
                
                if norm_i < 1e-8 or norm_j < 1e-8:
                    direction_conflict = 0.5
                else:
                    cos_sim = np.dot(output_i, output_j) / (norm_i * norm_j)
                    direction_conflict = 1 - cos_sim
                
                # Magnitude conflict
                max_norm = max(norm_i, norm_j)
                if max_norm < 1e-8:
                    magnitude_conflict = 0.0
                else:
                    magnitude_conflict = abs(norm_i - norm_j) / max_norm
                
                # Combined conflict
                conflict = (
                    self.alpha * direction_conflict +
                    (1 - self.alpha) * magnitude_conflict
                )
                
                conflict_matrix[i, j] = conflict
                conflict_matrix[j, i] = conflict
        
        max_conflict = conflict_matrix.max() if n > 1 else 0.0
        
        return conflict_matrix, float(max_conflict)
    
    def _compute_adaptive_threshold(self) -> float:
        """
        Compute adaptive conflict threshold based on history.
        
        Formula: τ_conflict = base_threshold + β * RecentConflictRate
        
        Returns:
            Adaptive threshold value
        """
        if not self.conflict_history:
            return self.base_threshold
        
        # Recent conflict rate (proportion above base threshold)
        recent_conflicts = list(self.conflict_history)[-20:]  # Last 20
        conflict_rate = sum(1 for c in recent_conflicts if c > self.base_threshold) / len(recent_conflicts)
        
        # Adaptive threshold
        beta = 0.2  # Adaptation strength
        threshold = self.base_threshold + beta * conflict_rate
        
        return min(threshold, 0.8)  # Cap at 0.8
    
    def _weighted_average(
        self,
        nano_outputs: Dict[str, np.ndarray],
        kv_hit_scores: Dict[str, float],
    ) -> FusionResult:
        """
        Compute weighted average of Nano outputs.
        
        Weights are computed via softmax of KV hit scores.
        
        Args:
            nano_outputs: Dictionary of nano_id -> output array
            kv_hit_scores: Dictionary of nano_id -> KV hit score
            
        Returns:
            FusionResult with weighted average output
        """
        nano_ids = list(nano_outputs.keys())
        
        # Compute softmax weights from KV hit scores
        scores = np.array([kv_hit_scores.get(nid, 0.5) for nid in nano_ids])
        scores = scores / self.temperature
        
        # Softmax
        exp_scores = np.exp(scores - scores.max())  # Numerical stability
        weights = exp_scores / exp_scores.sum()
        
        # Weighted average
        output_shape = nano_outputs[nano_ids[0]].shape
        fused = np.zeros(output_shape)
        
        for nano_id, weight in zip(nano_ids, weights):
            output = nano_outputs[nano_id]
            # Handle shape mismatch
            if output.shape != output_shape:
                output = np.resize(output, output_shape)
            fused += weight * output
        
        fusion_weights = {nid: float(w) for nid, w in zip(nano_ids, weights)}
        
        return FusionResult(
            fused_output=fused,
            fusion_method="weighted_average",
            fusion_weights=fusion_weights,
            conflict_detected=False,
            conflict_scores={},
        )
    
    def _winner_takes_all(
        self,
        nano_outputs: Dict[str, np.ndarray],
        kv_hit_scores: Dict[str, float],
        base_output: Optional[np.ndarray],
        conflict_matrix: np.ndarray,
    ) -> FusionResult:
        """
        Select winner based on alignment with base output.
        
        Formula: winner = argmax_j cos_sim(O_base, O_j)
        
        If base_output is not available, use KV hit scores.
        
        Args:
            nano_outputs: Dictionary of nano_id -> output array
            kv_hit_scores: Dictionary of nano_id -> KV hit score
            base_output: Base model output for comparison
            conflict_matrix: Conflict matrix for logging
            
        Returns:
            FusionResult with winner output
        """
        nano_ids = list(nano_outputs.keys())
        
        if base_output is not None:
            # Select based on alignment with base output
            base_flat = base_output.flatten()
            base_norm = np.linalg.norm(base_flat)
            
            best_score = -1
            winner_id = nano_ids[0]
            
            for nano_id in nano_ids:
                output_flat = nano_outputs[nano_id].flatten()
                
                # Ensure same length
                min_len = min(len(base_flat), len(output_flat))
                
                output_norm = np.linalg.norm(output_flat[:min_len])
                
                if base_norm < 1e-8 or output_norm < 1e-8:
                    score = 0.0
                else:
                    score = np.dot(base_flat[:min_len], output_flat[:min_len]) / (base_norm * output_norm)
                
                if score > best_score:
                    best_score = score
                    winner_id = nano_id
        else:
            # Fall back to KV hit scores
            winner_id = max(nano_ids, key=lambda nid: kv_hit_scores.get(nid, 0))
        
        # Create fusion weights (winner = 1, others = 0)
        fusion_weights = {nid: 1.0 if nid == winner_id else 0.0 for nid in nano_ids}
        
        return FusionResult(
            fused_output=nano_outputs[winner_id],
            fusion_method="winner_takes_all",
            fusion_weights=fusion_weights,
            conflict_detected=True,
            conflict_scores={},
            winner_id=winner_id,
            details={"selection_method": "base_alignment" if base_output is not None else "kv_hit"},
        )
    
    def _format_conflict_scores(
        self,
        conflict_matrix: np.ndarray,
        nano_outputs: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Format conflict matrix as dictionary."""
        nano_ids = list(nano_outputs.keys())
        n = len(nano_ids)
        
        scores = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{nano_ids[i]}_{nano_ids[j]}"
                scores[key] = float(conflict_matrix[i, j])
        
        return scores
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fusion statistics."""
        return {
            "fusion_count": self.fusion_count,
            "conflict_count": self.conflict_count,
            "conflict_rate": self.conflict_count / max(self.fusion_count, 1),
            "winner_takes_all_count": self.winner_takes_all_count,
            "winner_takes_all_rate": self.winner_takes_all_count / max(self.fusion_count, 1),
            "avg_conflict": np.mean(list(self.conflict_history)) if self.conflict_history else 0.0,
            "current_threshold": self._compute_adaptive_threshold(),
        }
    
    def reset_statistics(self):
        """Reset fusion statistics."""
        self.fusion_count = 0
        self.conflict_count = 0
        self.winner_takes_all_count = 0
        self.conflict_history.clear()


class OutputFusion:
    """
    Final output fusion combining base, AGA, and Nano outputs.
    
    Formula: O_final = O_base + α * O_AGA + γ * O_nano
    
    where:
    - α is the AGA gating coefficient (entropy-based)
    - γ is the Nano contribution weight (innovation-score-based)
    """
    
    def __init__(self):
        """Initialize the Output Fusion module."""
        self.fusion_count = 0
    
    def fuse(
        self,
        base_output: np.ndarray,
        aga_output: Optional[np.ndarray],
        nano_output: Optional[np.ndarray],
        alpha: float = 0.0,
        gamma: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fuse base, AGA, and Nano outputs.
        
        Args:
            base_output: Base model output
            aga_output: AGA module output (optional)
            nano_output: Fused Nano Model output (optional)
            alpha: AGA gating coefficient
            gamma: Nano contribution weight
            
        Returns:
            Tuple of (final_output, diagnostics)
        """
        self.fusion_count += 1
        
        final_output = base_output.copy()
        
        contributions = {"base": 1.0}
        
        if aga_output is not None and alpha > 0:
            # Ensure same shape
            if aga_output.shape == base_output.shape:
                final_output = final_output + alpha * (aga_output - base_output)
                contributions["aga"] = alpha
        
        if nano_output is not None and gamma > 0:
            # Ensure same shape
            if nano_output.shape == base_output.shape:
                final_output = final_output + gamma * nano_output
                contributions["nano"] = gamma
        
        diagnostics = {
            "alpha": alpha,
            "gamma": gamma,
            "contributions": contributions,
            "output_norm": float(np.linalg.norm(final_output)),
        }
        
        return final_output, diagnostics
