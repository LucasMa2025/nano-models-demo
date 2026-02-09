"""
Nano Model System
=================

Complete integrated system combining all components:
- Innovation Detection
- Nano Model Registry
- Nano Model Factory
- KV Store
- Fusion Engine
- Feedback Collection

This is the main entry point for using the Nano Models framework.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging
import hashlib

from .core.models import NanoModel, KVEntry, NanoModelDiagnostics, NanoLifecycleState
from .core.innovation_detector import InnovationDetector, InnovationDetectionResult
from .core.registry import NanoRegistry
from .core.factory import NanoModelFactory
from .core.fusion import ConflictAwareFusion, OutputFusion, FusionResult
from .storage.kv_store import HierarchicalKVStore
from .storage.database import DatabaseManager
from .feedback.collector import FeedbackCollector
from .feedback.analyzer import FeedbackAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """Configuration for the Nano Model System."""
    # Model dimensions
    hidden_dim: int = 256
    bottleneck_dim: int = 64
    
    # LoRA configuration
    lora_rank: int = 8
    lora_alpha: float = 32.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "fc1", "fc2"])
    
    # Innovation detection
    innovation_threshold: float = 0.7
    projection_weight: float = 0.4
    structure_weight: float = 0.4
    coverage_weight: float = 0.2
    
    # Fusion
    conflict_threshold: float = 0.3
    direction_weight: float = 0.7
    
    # Lifecycle
    dormant_threshold_days: int = 7
    deprecate_threshold_days: int = 30
    trial_validation_count: int = 10
    trial_success_threshold: float = 0.6
    max_active_nanos: int = 100
    
    # Storage
    max_kv_entries: int = 10000
    db_path: str = "data/nano_models.db"
    
    # Factory
    sample_buffer_threshold: int = 50
    auto_create_nanos: bool = True


@dataclass
class InferenceResult:
    """Result of system inference."""
    output: np.ndarray
    innovation_detected: bool
    innovation_score: float
    nanos_selected: List[str]
    fusion_method: str
    diagnostics: NanoModelDiagnostics
    processing_time_ms: float


class NanoModelSystem:
    """
    Complete Nano Model + AGA integrated system.
    
    This is the main entry point for inference with
    innovation-aware knowledge augmentation.
    
    Architecture:
    1. Base Model (simulated) produces hidden states
    2. Innovation Detector checks for innovation gaps
    3. If innovation detected:
       a. Select matching Nano Models from Registry
       b. If none, collect sample for future creation
       c. Fuse Nano outputs with conflict awareness
    4. Return final output
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the Nano Model System.
        
        Args:
            config: System configuration (uses defaults if None)
        """
        self.config = config or SystemConfig()
        
        # Initialize components
        self._init_components()
        
        # Statistics
        self.total_inferences = 0
        self.innovation_detections = 0
        self.nano_activations = 0
        
        logger.info("NanoModelSystem initialized")
    
    def _init_components(self):
        """Initialize all system components."""
        # Database
        self.db = DatabaseManager(self.config.db_path)
        
        # Innovation Detector
        self.detector = InnovationDetector(
            hidden_dim=self.config.hidden_dim,
            innovation_threshold=self.config.innovation_threshold,
            projection_weight=self.config.projection_weight,
            structure_weight=self.config.structure_weight,
            coverage_weight=self.config.coverage_weight,
        )
        
        # Registry
        self.registry = NanoRegistry(
            dormant_threshold_days=self.config.dormant_threshold_days,
            deprecate_threshold_days=self.config.deprecate_threshold_days,
            trial_validation_count=self.config.trial_validation_count,
            trial_success_threshold=self.config.trial_success_threshold,
            max_active_nanos=self.config.max_active_nanos,
        )
        
        # Factory
        self.factory = NanoModelFactory(
            registry=self.registry,
            hidden_dim=self.config.hidden_dim,
            default_lora_rank=self.config.lora_rank,
            default_lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            buffer_threshold=self.config.sample_buffer_threshold,
            auto_create=self.config.auto_create_nanos,
        )
        
        # KV Store
        self.kv_store = HierarchicalKVStore(
            max_entries_per_tier=self.config.max_kv_entries // 3
        )
        
        # Fusion
        self.nano_fusion = ConflictAwareFusion(
            direction_weight=self.config.direction_weight,
            base_conflict_threshold=self.config.conflict_threshold,
        )
        self.output_fusion = OutputFusion()
        
        # Feedback
        self.feedback_collector = FeedbackCollector(self.db)
        self.feedback_analyzer = FeedbackAnalyzer(self.feedback_collector)
    
    def infer(
        self,
        query: np.ndarray,
        hidden_states: Optional[np.ndarray] = None,
        attention_weights: Optional[np.ndarray] = None,
        base_output: Optional[np.ndarray] = None,
        aga_output: Optional[np.ndarray] = None,
        aga_alpha: float = 0.0,
        return_diagnostics: bool = True,
    ) -> InferenceResult:
        """
        Run inference with Nano Model integration.
        
        Args:
            query: Query embedding [hidden_dim]
            hidden_states: Hidden states [batch, seq, hidden] (simulated if None)
            attention_weights: Attention weights [batch, heads, seq, seq] (simulated if None)
            base_output: Base model output (uses query if None)
            aga_output: AGA module output (optional)
            aga_alpha: AGA gating coefficient
            return_diagnostics: Whether to return detailed diagnostics
            
        Returns:
            InferenceResult with output and metadata
        """
        import time
        start_time = time.time()
        
        self.total_inferences += 1
        
        # Simulate hidden states if not provided
        if hidden_states is None:
            hidden_states = query.reshape(1, 1, -1)
        
        # Simulate attention weights if not provided
        if attention_weights is None:
            seq_len = hidden_states.shape[1]
            attention_weights = np.random.rand(1, 8, seq_len, seq_len)
        
        # Use query as base output if not provided
        if base_output is None:
            base_output = query.copy()
        
        # Get all accessible KV entries
        all_kvs = self._get_all_kvs()
        
        # Step 1: Innovation Detection
        detection_result = self.detector.detect(
            hidden_states=hidden_states,
            attention_weights=attention_weights,
            kv_store=all_kvs,
        )
        
        # Update semantic subspaces with confident outputs
        if not detection_result.is_innovation:
            self.detector.update_semantic_subspaces(hidden_states, is_low_entropy=True)
            self.detector.update_reference_patterns(attention_weights)
        
        # Step 2: Nano Model Selection and Fusion
        selected_nanos = []
        nano_selections = []  # Initialize here to avoid UnboundLocalError
        nano_output = None
        fusion_method = "none"
        fusion_weights = {}
        conflict_detected = False
        conflict_scores = {}
        
        if detection_result.is_innovation:
            self.innovation_detections += 1
            
            # Select Nano Models
            nano_selections = self.registry.select(
                query_embedding=query,
                kv_store=all_kvs,
            )
            
            if nano_selections:
                self.nano_activations += 1
                selected_nanos = [n.nano_id for n, _ in nano_selections]
                
                # Generate Nano outputs
                nano_outputs = {}
                kv_hit_scores = {}
                
                for nano, score in nano_selections:
                    nano_outputs[nano.nano_id] = self._apply_nano(hidden_states, nano)
                    kv_hit_scores[nano.nano_id] = score
                
                # Fuse outputs
                fusion_result = self.nano_fusion.fuse(
                    nano_outputs=nano_outputs,
                    kv_hit_scores=kv_hit_scores,
                    base_output=base_output,
                )
                
                nano_output = fusion_result.fused_output
                fusion_method = fusion_result.fusion_method
                fusion_weights = fusion_result.fusion_weights
                conflict_detected = fusion_result.conflict_detected
                conflict_scores = fusion_result.conflict_scores
                
                # Record activation results
                for nano_id, weight in fusion_weights.items():
                    self.registry.record_activation_result(
                        nano_id=nano_id,
                        success=True,  # Will be updated by feedback
                        contribution=weight,
                    )
            else:
                # Collect sample for future Nano creation
                self.factory.collect_sample(
                    query=query,
                    target_output=base_output,  # Use base as target for now
                    innovation_score=detection_result.score,
                    innovation_components={
                        "projection_error": detection_result.projection_error,
                        "structure_novelty": detection_result.structure_novelty,
                        "kv_coverage": detection_result.kv_coverage,
                    },
                )
        
        # Step 3: Final Output Fusion
        gamma = detection_result.score if selected_nanos else 0.0
        
        final_output, fusion_details = self.output_fusion.fuse(
            base_output=base_output,
            aga_output=aga_output,
            nano_output=nano_output,
            alpha=aga_alpha,
            gamma=gamma,
        )
        
        # Compute processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Create diagnostics
        kv_scores = {}
        if nano_selections:
            for nano, score in nano_selections:
                kv_scores[nano.nano_id] = score
        
        diagnostics = NanoModelDiagnostics(
            selected_nanos=selected_nanos,
            kv_hit_scores=kv_scores,
            fusion_weights=fusion_weights,
            innovation_score=detection_result.score,
            innovation_components={
                "projection_error": detection_result.projection_error,
                "structure_novelty": detection_result.structure_novelty,
                "kv_coverage": detection_result.kv_coverage,
            },
            total_nano_contribution=gamma,
            conflict_detected=conflict_detected,
            conflict_scores=conflict_scores,
            fusion_method=fusion_method,
            inference_time_ms=processing_time,
            query_hash=self._hash_query(query),
        )
        
        # Collect system observation feedback
        self.feedback_collector.collect_system_observation(
            nano_id=selected_nanos[0] if selected_nanos else None,
            query_embedding=query,
            success=True,  # Will be updated by external feedback
            innovation_score=detection_result.score,
            fusion_contribution=gamma,
            details=diagnostics.to_dict(),
        )
        
        return InferenceResult(
            output=final_output,
            innovation_detected=detection_result.is_innovation,
            innovation_score=detection_result.score,
            nanos_selected=selected_nanos,
            fusion_method=fusion_method,
            diagnostics=diagnostics,
            processing_time_ms=processing_time,
        )
    
    def _get_all_kvs(self) -> List[KVEntry]:
        """Get all KV entries from the store."""
        kvs = []
        kvs.extend(self.kv_store.global_store.entries.values())
        kvs.extend(self.kv_store.shared_store.entries.values())
        kvs.extend(self.kv_store.exclusive_store.entries.values())
        return kvs
    
    def _apply_nano(self, hidden_states: np.ndarray, nano: NanoModel) -> np.ndarray:
        """Apply Nano Model to hidden states."""
        output = hidden_states.copy()
        
        for layer_name, lora in nano.lora_weights.items():
            # Apply LoRA transformation
            delta = lora.apply(hidden_states.reshape(-1, hidden_states.shape[-1]))
            output = output + 0.1 * delta.reshape(output.shape)
        
        return output.mean(axis=(0, 1))  # Return [hidden_dim]
    
    def _hash_query(self, query: np.ndarray) -> str:
        """Hash a query for tracking."""
        return hashlib.md5(query.tobytes()).hexdigest()[:16]
    
    # ==================== KV Management ====================
    
    def inject_global_kv(
        self,
        key: np.ndarray,
        value: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Inject a global KV entry."""
        kv_id = self.kv_store.insert_global(key, value, metadata)
        self.db.save_kv_entry(self.kv_store.global_store.get(kv_id))
        return kv_id
    
    def inject_exclusive_kv(
        self,
        key: np.ndarray,
        value: np.ndarray,
        nano_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Inject an exclusive KV entry for a Nano Model."""
        kv_id = self.kv_store.insert_exclusive(key, value, nano_id, metadata)
        self.db.save_kv_entry(self.kv_store.exclusive_store.get(kv_id))
        return kv_id
    
    # ==================== Nano Model Management ====================
    
    def create_nano_emergency(
        self,
        samples: List[Tuple[np.ndarray, np.ndarray, float]],
        innovation_domain: str = "",
    ) -> Optional[NanoModel]:
        """Create a Nano Model in emergency mode."""
        from .core.models import InnovationSample
        
        innovation_samples = [
            InnovationSample(
                query=q,
                target_output=t,
                innovation_score=s,
                innovation_components={},
            )
            for q, t, s in samples
        ]
        
        nano = self.factory.create_emergency(innovation_samples, innovation_domain)
        
        if nano:
            self.db.save_nano_model(nano)
        
        return nano
    
    def get_nano_model(self, nano_id: str) -> Optional[NanoModel]:
        """Get a Nano Model by ID."""
        return self.registry.get(nano_id)
    
    def list_nano_models(
        self,
        state_filter: Optional[NanoLifecycleState] = None,
    ) -> List[NanoModel]:
        """List all Nano Models."""
        return self.registry.get_all_nanos(state_filter)
    
    # ==================== Lifecycle Management ====================
    
    def update_lifecycle(self):
        """Update Nano Model lifecycle states."""
        self.registry.update_lifecycle()
    
    def deprecate_nano(self, nano_id: str) -> bool:
        """Manually deprecate a Nano Model."""
        nano = self.registry.get(nano_id)
        if nano:
            nano.state = NanoLifecycleState.DEPRECATED
            self.db.update_nano_state(nano_id, NanoLifecycleState.DEPRECATED)
            return True
        return False
    
    # ==================== Feedback ====================
    
    def provide_feedback(
        self,
        query_hash: str,
        rating: float,
        nano_id: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> str:
        """Provide user feedback for a query."""
        return self.feedback_collector.collect_user_rating(
            nano_id=nano_id,
            query_hash=query_hash,
            rating=rating,
            comment=comment,
        )
    
    def get_feedback_report(self) -> Dict[str, Any]:
        """Get feedback analysis report."""
        return self.feedback_analyzer.get_summary_report()
    
    # ==================== Statistics ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "system": {
                "total_inferences": self.total_inferences,
                "innovation_detections": self.innovation_detections,
                "nano_activations": self.nano_activations,
                "innovation_rate": self.innovation_detections / max(self.total_inferences, 1),
                "nano_activation_rate": self.nano_activations / max(self.total_inferences, 1),
            },
            "detector": self.detector.get_statistics(),
            "registry": self.registry.get_statistics(),
            "factory": self.factory.get_statistics(),
            "kv_store": self.kv_store.get_statistics(),
            "fusion": self.nano_fusion.get_statistics(),
            "feedback": self.feedback_collector.get_statistics(),
        }
    
    # ==================== Persistence ====================
    
    def save_state(self):
        """Save system state to database."""
        # Save all Nano Models
        for nano in self.registry.get_all_nanos():
            self.db.save_nano_model(nano)
        
        # Save statistics
        self.db.save_statistics("system", self.get_statistics())
        
        logger.info("System state saved")
    
    def load_state(self):
        """Load system state from database."""
        # Load Nano Models
        nanos = self.db.load_all_nano_models()
        for nano in nanos:
            self.registry.nanos[nano.nano_id] = nano
        
        logger.info(f"Loaded {len(nanos)} Nano Models from database")
    
    def close(self):
        """Close system and release resources."""
        self.save_state()
        self.db.close()
        logger.info("NanoModelSystem closed")
