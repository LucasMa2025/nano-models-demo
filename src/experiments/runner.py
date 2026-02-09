"""
Experiment Runner
=================

Automated experimentation system for evaluating Nano Models framework.

Supports:
- Configurable experiment parameters
- Multiple task types (factual, innovation, stability)
- Automated metrics collection
- Result persistence
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging
import json
import hashlib

from ..core.models import NanoModel, KVEntry, InnovationSample, NanoModelDiagnostics
from ..core.innovation_detector import InnovationDetector, InnovationDetectionResult
from ..core.registry import NanoRegistry
from ..core.factory import NanoModelFactory
from ..core.fusion import ConflictAwareFusion, OutputFusion
from ..storage.kv_store import HierarchicalKVStore
from ..storage.database import DatabaseManager
from .metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str = ""
    
    # Task configuration
    task_type: str = "mixed"  # factual, innovation, stability, mixed
    num_queries: int = 1000
    innovation_ratio: float = 0.3  # Ratio of innovation queries
    
    # Model configuration
    hidden_dim: int = 256
    lora_rank: int = 8
    lora_alpha: float = 32.0
    
    # Detection configuration
    innovation_threshold: float = 0.7
    projection_weight: float = 0.4
    structure_weight: float = 0.4
    coverage_weight: float = 0.2
    
    # Fusion configuration
    conflict_threshold: float = 0.3
    direction_weight: float = 0.7
    
    # Lifecycle configuration
    dormant_threshold_days: int = 7
    deprecate_threshold_days: int = 30
    
    # Random seed
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "num_queries": self.num_queries,
            "innovation_ratio": self.innovation_ratio,
            "hidden_dim": self.hidden_dim,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "innovation_threshold": self.innovation_threshold,
            "projection_weight": self.projection_weight,
            "structure_weight": self.structure_weight,
            "coverage_weight": self.coverage_weight,
            "conflict_threshold": self.conflict_threshold,
            "direction_weight": self.direction_weight,
            "dormant_threshold_days": self.dormant_threshold_days,
            "deprecate_threshold_days": self.deprecate_threshold_days,
            "seed": self.seed,
        }


@dataclass
class QueryResult:
    """Result of processing a single query."""
    query_id: str
    query_type: str  # factual or innovation
    is_correct: bool
    innovation_detected: bool
    innovation_score: float
    nano_selected: bool
    selected_nanos: List[str]
    fusion_method: str
    processing_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """
    Automated experiment runner for Nano Models framework.
    
    Runs experiments with:
    - Synthetic query generation
    - Automated evaluation
    - Metrics collection
    - Result persistence
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        db_manager: Optional[DatabaseManager] = None,
    ):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
            db_manager: Optional database manager for persistence
        """
        self.config = config
        self.db = db_manager
        
        # Set random seed
        np.random.seed(config.seed)
        
        # Initialize components
        self._init_components()
        
        # Metrics collector
        self.metrics = MetricsCollector()
        
        # Results
        self.results: List[QueryResult] = []
        self.experiment_id = self._generate_experiment_id()
        
        logger.info(f"ExperimentRunner initialized: {config.name}")
    
    def _init_components(self):
        """Initialize framework components."""
        # Innovation detector
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
        )
        
        # Factory
        self.factory = NanoModelFactory(
            registry=self.registry,
            hidden_dim=self.config.hidden_dim,
            default_lora_rank=self.config.lora_rank,
            default_lora_alpha=self.config.lora_alpha,
        )
        
        # KV Store
        self.kv_store = HierarchicalKVStore()
        
        # Fusion
        self.fusion = ConflictAwareFusion(
            direction_weight=self.config.direction_weight,
            base_conflict_threshold=self.config.conflict_threshold,
        )
        
        self.output_fusion = OutputFusion()
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        content = f"{self.config.name}_{datetime.now().isoformat()}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash_val}"
    
    def run(self) -> Dict[str, Any]:
        """
        Run the experiment.
        
        Returns:
            Dictionary with experiment results and metrics
        """
        logger.info(f"Starting experiment: {self.experiment_id}")
        start_time = datetime.now()
        
        # Save experiment config to database
        if self.db:
            self.db.save_experiment(
                experiment_id=self.experiment_id,
                name=self.config.name,
                description=self.config.description,
                config=self.config.to_dict(),
                status="running",
            )
        
        # Generate and process queries
        for i in range(self.config.num_queries):
            query_type = self._determine_query_type(i)
            query, target = self._generate_query(query_type)
            
            result = self._process_query(i, query_type, query, target)
            self.results.append(result)
            
            # Update metrics
            self.metrics.record_query(result)
            
            # Periodic lifecycle update
            if (i + 1) % 100 == 0:
                self.registry.update_lifecycle()
                logger.info(f"Processed {i + 1}/{self.config.num_queries} queries")
        
        # Compute final metrics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        final_metrics = self._compute_final_metrics(duration)
        
        # Save results to database
        if self.db:
            self.db.update_experiment_results(
                experiment_id=self.experiment_id,
                results={"query_results": [r.__dict__ for r in self.results[-100:]]},  # Last 100
                metrics=final_metrics,
                status="completed",
            )
        
        logger.info(f"Experiment completed: {self.experiment_id}")
        
        return {
            "experiment_id": self.experiment_id,
            "config": self.config.to_dict(),
            "metrics": final_metrics,
            "duration_seconds": duration,
        }
    
    def _determine_query_type(self, index: int) -> str:
        """Determine query type based on configuration."""
        if self.config.task_type == "factual":
            return "factual"
        elif self.config.task_type == "innovation":
            return "innovation"
        else:
            # Mixed: use innovation_ratio
            if np.random.random() < self.config.innovation_ratio:
                return "innovation"
            return "factual"
    
    def _generate_query(self, query_type: str) -> tuple:
        """
        Generate a synthetic query and target.
        
        Args:
            query_type: Type of query (factual or innovation)
            
        Returns:
            Tuple of (query_embedding, target_embedding)
        """
        dim = self.config.hidden_dim
        
        if query_type == "factual":
            # Factual queries: within known semantic space
            # Generate from a mixture of known patterns
            base = np.random.randn(dim) * 0.5
            noise = np.random.randn(dim) * 0.1
            query = base + noise
            
            # Target is a transformation of query
            target = query * 1.1 + np.random.randn(dim) * 0.05
        else:
            # Innovation queries: outside known semantic space
            # Generate with higher variance and different structure
            query = np.random.randn(dim) * 2.0
            
            # Add novel structure
            novel_component = np.zeros(dim)
            novel_component[:dim//4] = np.random.randn(dim//4) * 3.0
            query = query + novel_component
            
            # Target requires novel derivation
            target = np.tanh(query) + np.random.randn(dim) * 0.1
        
        return query, target
    
    def _process_query(
        self,
        index: int,
        query_type: str,
        query: np.ndarray,
        target: np.ndarray,
    ) -> QueryResult:
        """
        Process a single query through the framework.
        
        Args:
            index: Query index
            query_type: Type of query
            query: Query embedding
            target: Target embedding
            
        Returns:
            QueryResult with processing outcome
        """
        import time
        start_time = time.time()
        
        query_id = f"q_{index:06d}"
        
        # Simulate hidden states and attention
        hidden_states = query.reshape(1, 1, -1)  # [batch, seq, hidden]
        attention_weights = np.random.rand(1, 8, 16, 16)  # [batch, heads, seq, seq]
        
        # Get all KV entries for detection
        all_kvs = list(self.kv_store.global_store.entries.values())
        all_kvs.extend(self.kv_store.exclusive_store.entries.values())
        
        # Innovation detection
        detection_result = self.detector.detect(
            hidden_states=hidden_states,
            attention_weights=attention_weights,
            kv_store=all_kvs,
        )
        
        # Update semantic subspaces with low-entropy outputs
        if not detection_result.is_innovation:
            self.detector.update_semantic_subspaces(hidden_states, is_low_entropy=True)
            self.detector.update_reference_patterns(attention_weights)
        
        selected_nanos = []
        fusion_method = "none"
        nano_output = None
        
        if detection_result.is_innovation:
            # Select Nano Models
            nano_selections = self.registry.select(
                query_embedding=query,
                kv_store=all_kvs,
            )
            
            if nano_selections:
                selected_nanos = [n.nano_id for n, _ in nano_selections]
                
                # Generate Nano outputs (simulated)
                nano_outputs = {}
                kv_hit_scores = {}
                for nano, score in nano_selections:
                    nano_outputs[nano.nano_id] = self._simulate_nano_output(nano, query)
                    kv_hit_scores[nano.nano_id] = score
                
                # Fuse outputs
                fusion_result = self.fusion.fuse(
                    nano_outputs=nano_outputs,
                    kv_hit_scores=kv_hit_scores,
                    base_output=query,
                )
                
                nano_output = fusion_result.fused_output
                fusion_method = fusion_result.fusion_method
                
                # Record activation results
                for nano_id in selected_nanos:
                    self.registry.record_activation_result(
                        nano_id=nano_id,
                        success=True,  # Simplified
                        contribution=fusion_result.fusion_weights.get(nano_id, 0),
                    )
            else:
                # Collect sample for future Nano creation
                self.factory.collect_sample(
                    query=query,
                    target_output=target,
                    innovation_score=detection_result.score,
                    innovation_components={
                        "projection_error": detection_result.projection_error,
                        "structure_novelty": detection_result.structure_novelty,
                        "kv_coverage": detection_result.kv_coverage,
                    },
                )
        
        # Compute output (simulated)
        if nano_output is not None:
            output = query + 0.3 * nano_output
        else:
            output = query
        
        # Evaluate correctness (simplified: cosine similarity to target)
        similarity = np.dot(output.flatten(), target.flatten()) / (
            np.linalg.norm(output) * np.linalg.norm(target) + 1e-8
        )
        
        # Correctness threshold depends on query type
        threshold = 0.5 if query_type == "factual" else 0.3
        is_correct = similarity > threshold
        
        processing_time = (time.time() - start_time) * 1000
        
        return QueryResult(
            query_id=query_id,
            query_type=query_type,
            is_correct=is_correct,
            innovation_detected=detection_result.is_innovation,
            innovation_score=detection_result.score,
            nano_selected=len(selected_nanos) > 0,
            selected_nanos=selected_nanos,
            fusion_method=fusion_method,
            processing_time_ms=processing_time,
            details={
                "similarity": float(similarity),
                "projection_error": detection_result.projection_error,
                "structure_novelty": detection_result.structure_novelty,
                "kv_coverage": detection_result.kv_coverage,
            },
        )
    
    def _simulate_nano_output(self, nano: NanoModel, query: np.ndarray) -> np.ndarray:
        """Simulate Nano Model output (for demo purposes)."""
        # Apply LoRA transformation (simplified)
        output = query.copy()
        
        for layer_name, lora in nano.lora_weights.items():
            # Simplified application
            delta = lora.apply(query.reshape(1, -1)).flatten()
            output = output + 0.1 * delta[:len(output)]
        
        return output
    
    def _compute_final_metrics(self, duration: float) -> Dict[str, Any]:
        """Compute final experiment metrics."""
        # Basic metrics
        total = len(self.results)
        correct = sum(1 for r in self.results if r.is_correct)
        
        # By query type
        factual_results = [r for r in self.results if r.query_type == "factual"]
        innovation_results = [r for r in self.results if r.query_type == "innovation"]
        
        factual_correct = sum(1 for r in factual_results if r.is_correct)
        innovation_correct = sum(1 for r in innovation_results if r.is_correct)
        
        # Innovation detection metrics
        true_positives = sum(
            1 for r in innovation_results if r.innovation_detected
        )
        false_positives = sum(
            1 for r in factual_results if r.innovation_detected
        )
        
        # Nano Model metrics
        nano_selections = sum(1 for r in self.results if r.nano_selected)
        
        return {
            "total_queries": total,
            "overall_accuracy": correct / total if total > 0 else 0,
            "factual_accuracy": factual_correct / len(factual_results) if factual_results else 0,
            "innovation_accuracy": innovation_correct / len(innovation_results) if innovation_results else 0,
            "innovation_detection_rate": true_positives / len(innovation_results) if innovation_results else 0,
            "false_positive_rate": false_positives / len(factual_results) if factual_results else 0,
            "nano_selection_rate": nano_selections / total if total > 0 else 0,
            "nano_models_created": self.factory.total_created,
            "active_nano_models": len(self.registry.get_all_nanos()),
            "avg_processing_time_ms": np.mean([r.processing_time_ms for r in self.results]),
            "total_duration_seconds": duration,
            "queries_per_second": total / duration if duration > 0 else 0,
            "detector_stats": self.detector.get_statistics(),
            "registry_stats": self.registry.get_statistics(),
            "fusion_stats": self.fusion.get_statistics(),
        }
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of experiment results."""
        if not self.results:
            return {"status": "no results"}
        
        return {
            "experiment_id": self.experiment_id,
            "total_queries": len(self.results),
            "accuracy": sum(1 for r in self.results if r.is_correct) / len(self.results),
            "nano_models_created": self.factory.total_created,
            "sample_results": [r.__dict__ for r in self.results[:5]],
        }


class AblationStudy:
    """
    Ablation study runner for analyzing component contributions.
    """
    
    def __init__(self, base_config: ExperimentConfig, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize ablation study.
        
        Args:
            base_config: Base experiment configuration
            db_manager: Optional database manager
        """
        self.base_config = base_config
        self.db = db_manager
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def run_ablation(self, ablation_name: str, config_modifier: Callable[[ExperimentConfig], ExperimentConfig]) -> Dict[str, Any]:
        """
        Run a single ablation experiment.
        
        Args:
            ablation_name: Name of the ablation
            config_modifier: Function to modify config
            
        Returns:
            Experiment results
        """
        # Create modified config
        modified_config = config_modifier(ExperimentConfig(**self.base_config.to_dict()))
        modified_config.name = f"{self.base_config.name}_ablation_{ablation_name}"
        
        # Run experiment
        runner = ExperimentRunner(modified_config, self.db)
        results = runner.run()
        
        self.results[ablation_name] = results
        return results
    
    def run_standard_ablations(self) -> Dict[str, Dict[str, Any]]:
        """Run standard ablation studies."""
        
        # Ablation 1: Without consistency (structure novelty)
        def no_structure(config):
            config.structure_weight = 0.0
            config.projection_weight = 0.6
            config.coverage_weight = 0.4
            return config
        
        self.run_ablation("no_structure_novelty", no_structure)
        
        # Ablation 2: Without KV coverage
        def no_coverage(config):
            config.coverage_weight = 0.0
            config.projection_weight = 0.5
            config.structure_weight = 0.5
            return config
        
        self.run_ablation("no_kv_coverage", no_coverage)
        
        # Ablation 3: Projection only
        def projection_only(config):
            config.projection_weight = 1.0
            config.structure_weight = 0.0
            config.coverage_weight = 0.0
            return config
        
        self.run_ablation("projection_only", projection_only)
        
        # Ablation 4: Higher threshold
        def high_threshold(config):
            config.innovation_threshold = 0.9
            return config
        
        self.run_ablation("high_threshold", high_threshold)
        
        # Ablation 5: Lower threshold
        def low_threshold(config):
            config.innovation_threshold = 0.5
            return config
        
        self.run_ablation("low_threshold", low_threshold)
        
        return self.results
    
    def get_comparison_table(self) -> List[Dict[str, Any]]:
        """Get comparison table of ablation results."""
        table = []
        
        for name, result in self.results.items():
            metrics = result.get("metrics", {})
            table.append({
                "ablation": name,
                "overall_accuracy": metrics.get("overall_accuracy", 0),
                "innovation_accuracy": metrics.get("innovation_accuracy", 0),
                "false_positive_rate": metrics.get("false_positive_rate", 0),
                "nano_models_created": metrics.get("nano_models_created", 0),
            })
        
        return table
