"""
Metrics Collector
=================

Collects and computes metrics for Nano Models experiments.

Metrics Categories:
- Accuracy metrics (factual, innovation, overall)
- Detection metrics (precision, recall, F1)
- Efficiency metrics (latency, throughput)
- Nano Model metrics (creation rate, activation rate)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """A snapshot of metrics at a point in time."""
    timestamp: datetime
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
        }


class MetricsCollector:
    """
    Collects and computes metrics for experiments.
    
    Tracks:
    - Per-query results
    - Running averages
    - Time series data
    - Aggregated statistics
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize the metrics collector.
        
        Args:
            window_size: Window size for rolling averages
        """
        self.window_size = window_size
        
        # Raw data
        self.query_results: List[Dict[str, Any]] = []
        
        # Counters
        self.total_queries = 0
        self.correct_queries = 0
        self.factual_queries = 0
        self.factual_correct = 0
        self.innovation_queries = 0
        self.innovation_correct = 0
        
        # Detection counters
        self.true_positives = 0  # Innovation detected as innovation
        self.false_positives = 0  # Factual detected as innovation
        self.true_negatives = 0  # Factual detected as factual
        self.false_negatives = 0  # Innovation detected as factual
        
        # Nano Model counters
        self.nano_selections = 0
        self.nano_creations = 0
        
        # Time series
        self.accuracy_history: List[float] = []
        self.innovation_score_history: List[float] = []
        self.latency_history: List[float] = []
        
        # Snapshots
        self.snapshots: List[MetricSnapshot] = []
        
        logger.debug("MetricsCollector initialized")
    
    def record_query(self, result: Any):
        """
        Record a query result.
        
        Args:
            result: QueryResult object with query outcome
        """
        self.total_queries += 1
        
        # Store raw result
        if hasattr(result, '__dict__'):
            self.query_results.append(result.__dict__)
        else:
            self.query_results.append(result)
        
        # Update counters
        if result.is_correct:
            self.correct_queries += 1
        
        if result.query_type == "factual":
            self.factual_queries += 1
            if result.is_correct:
                self.factual_correct += 1
            
            # Detection metrics
            if result.innovation_detected:
                self.false_positives += 1
            else:
                self.true_negatives += 1
        else:  # innovation
            self.innovation_queries += 1
            if result.is_correct:
                self.innovation_correct += 1
            
            # Detection metrics
            if result.innovation_detected:
                self.true_positives += 1
            else:
                self.false_negatives += 1
        
        # Nano Model metrics
        if result.nano_selected:
            self.nano_selections += 1
        
        # Update time series
        self.accuracy_history.append(1.0 if result.is_correct else 0.0)
        self.innovation_score_history.append(result.innovation_score)
        self.latency_history.append(result.processing_time_ms)
        
        # Periodic snapshot
        if self.total_queries % self.window_size == 0:
            self._take_snapshot()
    
    def record_nano_creation(self):
        """Record a Nano Model creation event."""
        self.nano_creations += 1
    
    def _take_snapshot(self):
        """Take a metrics snapshot."""
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            metrics=self.get_current_metrics(),
        )
        self.snapshots.append(snapshot)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics values."""
        metrics = {}
        
        # Accuracy metrics
        metrics["overall_accuracy"] = self.correct_queries / max(self.total_queries, 1)
        metrics["factual_accuracy"] = self.factual_correct / max(self.factual_queries, 1)
        metrics["innovation_accuracy"] = self.innovation_correct / max(self.innovation_queries, 1)
        
        # Detection metrics
        precision = self.true_positives / max(self.true_positives + self.false_positives, 1)
        recall = self.true_positives / max(self.true_positives + self.false_negatives, 1)
        
        metrics["detection_precision"] = precision
        metrics["detection_recall"] = recall
        metrics["detection_f1"] = 2 * precision * recall / max(precision + recall, 1e-8)
        metrics["false_positive_rate"] = self.false_positives / max(self.factual_queries, 1)
        
        # Nano Model metrics
        metrics["nano_selection_rate"] = self.nano_selections / max(self.total_queries, 1)
        metrics["nano_creation_rate"] = self.nano_creations / max(self.total_queries, 1) * 1000  # Per 1000 queries
        
        # Latency metrics
        if self.latency_history:
            metrics["avg_latency_ms"] = np.mean(self.latency_history)
            metrics["p95_latency_ms"] = np.percentile(self.latency_history, 95)
            metrics["p99_latency_ms"] = np.percentile(self.latency_history, 99)
        
        # Rolling metrics (last window)
        if len(self.accuracy_history) >= self.window_size:
            metrics["rolling_accuracy"] = np.mean(self.accuracy_history[-self.window_size:])
            metrics["rolling_innovation_score"] = np.mean(self.innovation_score_history[-self.window_size:])
        
        return metrics
    
    def get_rolling_average(self, metric_name: str, window: Optional[int] = None) -> float:
        """
        Get rolling average for a metric.
        
        Args:
            metric_name: Name of the metric
            window: Window size (default: self.window_size)
            
        Returns:
            Rolling average value
        """
        window = window or self.window_size
        
        if metric_name == "accuracy":
            data = self.accuracy_history
        elif metric_name == "innovation_score":
            data = self.innovation_score_history
        elif metric_name == "latency":
            data = self.latency_history
        else:
            return 0.0
        
        if not data:
            return 0.0
        
        return float(np.mean(data[-window:]))
    
    def get_confusion_matrix(self) -> Dict[str, int]:
        """Get confusion matrix for innovation detection."""
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "total_queries": self.total_queries,
            "metrics": self.get_current_metrics(),
            "confusion_matrix": self.get_confusion_matrix(),
            "counters": {
                "factual_queries": self.factual_queries,
                "innovation_queries": self.innovation_queries,
                "nano_selections": self.nano_selections,
                "nano_creations": self.nano_creations,
            },
            "num_snapshots": len(self.snapshots),
        }
    
    def get_time_series(self, metric_name: str) -> List[float]:
        """
        Get time series data for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of metric values over time
        """
        if metric_name == "accuracy":
            return self.accuracy_history.copy()
        elif metric_name == "innovation_score":
            return self.innovation_score_history.copy()
        elif metric_name == "latency":
            return self.latency_history.copy()
        else:
            return []
    
    def get_snapshot_history(self) -> List[Dict[str, Any]]:
        """Get history of metric snapshots."""
        return [s.to_dict() for s in self.snapshots]
    
    def reset(self):
        """Reset all metrics."""
        self.query_results.clear()
        self.total_queries = 0
        self.correct_queries = 0
        self.factual_queries = 0
        self.factual_correct = 0
        self.innovation_queries = 0
        self.innovation_correct = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.nano_selections = 0
        self.nano_creations = 0
        self.accuracy_history.clear()
        self.innovation_score_history.clear()
        self.latency_history.clear()
        self.snapshots.clear()
        
        logger.debug("MetricsCollector reset")


class ExperimentComparator:
    """
    Compares results across multiple experiments.
    """
    
    def __init__(self):
        """Initialize the comparator."""
        self.experiments: Dict[str, Dict[str, Any]] = {}
    
    def add_experiment(self, name: str, results: Dict[str, Any]):
        """Add experiment results for comparison."""
        self.experiments[name] = results
    
    def compare_metric(self, metric_name: str) -> Dict[str, float]:
        """
        Compare a specific metric across experiments.
        
        Args:
            metric_name: Name of the metric to compare
            
        Returns:
            Dictionary of experiment name -> metric value
        """
        comparison = {}
        
        for name, results in self.experiments.items():
            metrics = results.get("metrics", {})
            comparison[name] = metrics.get(metric_name, 0.0)
        
        return comparison
    
    def get_comparison_table(self, metric_names: List[str]) -> List[Dict[str, Any]]:
        """
        Get comparison table for multiple metrics.
        
        Args:
            metric_names: List of metric names to compare
            
        Returns:
            List of dictionaries with experiment comparisons
        """
        table = []
        
        for name, results in self.experiments.items():
            row = {"experiment": name}
            metrics = results.get("metrics", {})
            
            for metric in metric_names:
                row[metric] = metrics.get(metric, 0.0)
            
            table.append(row)
        
        return table
    
    def find_best(self, metric_name: str, higher_is_better: bool = True) -> str:
        """
        Find the best experiment for a metric.
        
        Args:
            metric_name: Metric to optimize
            higher_is_better: Whether higher values are better
            
        Returns:
            Name of the best experiment
        """
        comparison = self.compare_metric(metric_name)
        
        if not comparison:
            return ""
        
        if higher_is_better:
            return max(comparison.keys(), key=lambda k: comparison[k])
        else:
            return min(comparison.keys(), key=lambda k: comparison[k])
