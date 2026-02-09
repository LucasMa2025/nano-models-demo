"""
Feedback Analyzer
=================

Analyzes collected feedback to:
- Assess Nano Model quality
- Tune innovation detection parameters
- Recommend lifecycle transitions
- Identify patterns and anomalies
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import logging

from .collector import FeedbackCollector, FeedbackEntry, FeedbackType

logger = logging.getLogger(__name__)


@dataclass
class NanoQualityAssessment:
    """Quality assessment for a Nano Model."""
    nano_id: str
    overall_score: float
    success_rate: float
    avg_rating: float
    feedback_count: int
    trend: str  # "improving", "stable", "declining"
    recommendation: str  # "promote", "maintain", "demote", "deprecate"
    details: Dict[str, Any]


@dataclass
class DetectionTuningRecommendation:
    """Recommendation for tuning innovation detection."""
    current_threshold: float
    recommended_threshold: float
    current_weights: Dict[str, float]
    recommended_weights: Dict[str, float]
    rationale: str


class FeedbackAnalyzer:
    """
    Analyzes feedback to provide insights and recommendations.
    
    Analysis Types:
    1. Nano Model Quality Assessment
    2. Innovation Detection Tuning
    3. Lifecycle Recommendations
    4. Pattern Detection
    5. Anomaly Detection
    """
    
    def __init__(self, collector: FeedbackCollector):
        """
        Initialize the feedback analyzer.
        
        Args:
            collector: FeedbackCollector instance
        """
        self.collector = collector
        
        # Analysis cache
        self._quality_cache: Dict[str, NanoQualityAssessment] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)
        
        logger.info("FeedbackAnalyzer initialized")
    
    def assess_nano_quality(self, nano_id: str) -> NanoQualityAssessment:
        """
        Assess the quality of a Nano Model based on feedback.
        
        Args:
            nano_id: ID of the Nano Model
            
        Returns:
            NanoQualityAssessment with quality metrics and recommendations
        """
        # Check cache
        if self._is_cache_valid() and nano_id in self._quality_cache:
            return self._quality_cache[nano_id]
        
        feedback = self.collector.get_feedback_for_nano(nano_id)
        
        if not feedback:
            return NanoQualityAssessment(
                nano_id=nano_id,
                overall_score=0.5,
                success_rate=0.0,
                avg_rating=0.0,
                feedback_count=0,
                trend="unknown",
                recommendation="maintain",
                details={"reason": "no feedback available"},
            )
        
        # Compute metrics
        success_rate = sum(1 for f in feedback if f.success) / len(feedback)
        
        ratings = [f.rating for f in feedback if f.rating is not None]
        avg_rating = np.mean(ratings) if ratings else 0.5
        
        # Compute trend (compare recent vs older feedback)
        if len(feedback) >= 10:
            recent = feedback[-5:]
            older = feedback[-10:-5]
            
            recent_success = sum(1 for f in recent if f.success) / len(recent)
            older_success = sum(1 for f in older if f.success) / len(older)
            
            if recent_success > older_success + 0.1:
                trend = "improving"
            elif recent_success < older_success - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        # Overall score (weighted combination)
        overall_score = 0.6 * success_rate + 0.4 * avg_rating
        
        # Recommendation
        recommendation = self._compute_recommendation(overall_score, trend, len(feedback))
        
        assessment = NanoQualityAssessment(
            nano_id=nano_id,
            overall_score=overall_score,
            success_rate=success_rate,
            avg_rating=avg_rating,
            feedback_count=len(feedback),
            trend=trend,
            recommendation=recommendation,
            details={
                "feedback_types": self._count_feedback_types(feedback),
                "recent_success_rate": sum(1 for f in feedback[-10:] if f.success) / min(len(feedback), 10),
            },
        )
        
        # Cache result
        self._quality_cache[nano_id] = assessment
        self._cache_timestamp = datetime.now()
        
        return assessment
    
    def _compute_recommendation(
        self,
        overall_score: float,
        trend: str,
        feedback_count: int,
    ) -> str:
        """Compute lifecycle recommendation."""
        if feedback_count < 5:
            return "maintain"  # Not enough data
        
        if overall_score >= 0.8 and trend in ["improving", "stable"]:
            return "promote"
        elif overall_score >= 0.6:
            return "maintain"
        elif overall_score >= 0.4 or trend == "improving":
            return "demote"
        else:
            return "deprecate"
    
    def _count_feedback_types(self, feedback: List[FeedbackEntry]) -> Dict[str, int]:
        """Count feedback by type."""
        counts = defaultdict(int)
        for f in feedback:
            counts[f.feedback_type.value] += 1
        return dict(counts)
    
    def recommend_detection_tuning(
        self,
        current_threshold: float,
        current_weights: Dict[str, float],
    ) -> DetectionTuningRecommendation:
        """
        Recommend tuning for innovation detection parameters.
        
        Args:
            current_threshold: Current innovation threshold
            current_weights: Current detection weights
            
        Returns:
            DetectionTuningRecommendation with suggested changes
        """
        # Analyze system observations
        observations = self.collector.get_feedback_by_type(FeedbackType.SYSTEM_OBSERVATION)
        
        if len(observations) < 50:
            return DetectionTuningRecommendation(
                current_threshold=current_threshold,
                recommended_threshold=current_threshold,
                current_weights=current_weights,
                recommended_weights=current_weights,
                rationale="Insufficient data for tuning recommendations",
            )
        
        # Analyze innovation scores vs success
        innovation_scores = []
        successes = []
        
        for obs in observations:
            if "innovation_score" in obs.details:
                innovation_scores.append(obs.details["innovation_score"])
                successes.append(obs.success)
        
        if not innovation_scores:
            return DetectionTuningRecommendation(
                current_threshold=current_threshold,
                recommended_threshold=current_threshold,
                current_weights=current_weights,
                recommended_weights=current_weights,
                rationale="No innovation score data available",
            )
        
        # Find optimal threshold
        best_threshold = current_threshold
        best_f1 = 0.0
        
        for threshold in np.arange(0.3, 0.9, 0.05):
            predictions = [s >= threshold for s in innovation_scores]
            
            # Compute F1 (simplified: treating high innovation score as positive)
            tp = sum(1 for p, s in zip(predictions, successes) if p and s)
            fp = sum(1 for p, s in zip(predictions, successes) if p and not s)
            fn = sum(1 for p, s in zip(predictions, successes) if not p and s)
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Determine rationale
        if abs(best_threshold - current_threshold) < 0.05:
            rationale = "Current threshold is near optimal"
        elif best_threshold > current_threshold:
            rationale = f"Recommend increasing threshold to reduce false positives (F1: {best_f1:.3f})"
        else:
            rationale = f"Recommend decreasing threshold to improve recall (F1: {best_f1:.3f})"
        
        return DetectionTuningRecommendation(
            current_threshold=current_threshold,
            recommended_threshold=float(best_threshold),
            current_weights=current_weights,
            recommended_weights=current_weights,  # Weight tuning would require more analysis
            rationale=rationale,
        )
    
    def get_lifecycle_recommendations(self) -> List[Tuple[str, str, str]]:
        """
        Get lifecycle recommendations for all Nano Models.
        
        Returns:
            List of (nano_id, current_recommendation, reason) tuples
        """
        recommendations = []
        
        # Get unique Nano IDs from feedback
        nano_ids = set(f.nano_id for f in self.collector.feedback_buffer if f.nano_id)
        
        for nano_id in nano_ids:
            assessment = self.assess_nano_quality(nano_id)
            
            if assessment.recommendation != "maintain":
                reason = f"Score: {assessment.overall_score:.2f}, Trend: {assessment.trend}"
                recommendations.append((nano_id, assessment.recommendation, reason))
        
        return recommendations
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect anomalies in feedback patterns.
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check for sudden success rate drops
        recent = self.collector.get_recent_feedback(100)
        if len(recent) >= 50:
            first_half = recent[:50]
            second_half = recent[50:]
            
            first_success = sum(1 for f in first_half if f.success) / len(first_half)
            second_success = sum(1 for f in second_half if f.success) / len(second_half)
            
            if first_success - second_success > 0.2:
                anomalies.append({
                    "type": "success_rate_drop",
                    "severity": "high",
                    "details": {
                        "previous_rate": first_success,
                        "current_rate": second_success,
                        "drop": first_success - second_success,
                    },
                })
        
        # Check for high error rate
        errors = self.collector.get_feedback_by_type(FeedbackType.ERROR_REPORT)
        if len(errors) > 10:
            recent_errors = [e for e in errors if (datetime.now() - e.timestamp).seconds < 3600]
            if len(recent_errors) > 5:
                anomalies.append({
                    "type": "high_error_rate",
                    "severity": "medium",
                    "details": {
                        "recent_errors": len(recent_errors),
                        "error_types": list(set(e.details.get("error_type", "unknown") for e in recent_errors)),
                    },
                })
        
        # Check for Nano Models with consistently low ratings
        for nano_id in set(f.nano_id for f in self.collector.feedback_buffer if f.nano_id):
            feedback = self.collector.get_feedback_for_nano(nano_id)
            ratings = [f.rating for f in feedback if f.rating is not None]
            
            if len(ratings) >= 10 and np.mean(ratings) < 0.3:
                anomalies.append({
                    "type": "low_quality_nano",
                    "severity": "medium",
                    "details": {
                        "nano_id": nano_id,
                        "avg_rating": np.mean(ratings),
                        "feedback_count": len(ratings),
                    },
                })
        
        return anomalies
    
    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report.
        
        Returns:
            Dictionary with summary statistics and insights
        """
        stats = self.collector.get_statistics()
        
        # Quality assessments for all Nanos
        nano_ids = set(f.nano_id for f in self.collector.feedback_buffer if f.nano_id)
        assessments = {
            nano_id: self.assess_nano_quality(nano_id).overall_score
            for nano_id in nano_ids
        }
        
        # Anomalies
        anomalies = self.detect_anomalies()
        
        # Recommendations
        recommendations = self.get_lifecycle_recommendations()
        
        return {
            "statistics": stats,
            "nano_quality_scores": assessments,
            "anomalies": anomalies,
            "lifecycle_recommendations": recommendations,
            "overall_health": self._compute_overall_health(stats, anomalies),
        }
    
    def _compute_overall_health(
        self,
        stats: Dict[str, Any],
        anomalies: List[Dict[str, Any]],
    ) -> str:
        """Compute overall system health."""
        success_rate = stats.get("success_rate", 0.5)
        high_severity_anomalies = sum(1 for a in anomalies if a.get("severity") == "high")
        
        if success_rate >= 0.8 and high_severity_anomalies == 0:
            return "healthy"
        elif success_rate >= 0.6 and high_severity_anomalies <= 1:
            return "moderate"
        else:
            return "needs_attention"
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache_timestamp is None:
            return False
        return datetime.now() - self._cache_timestamp < self._cache_ttl
    
    def clear_cache(self):
        """Clear analysis cache."""
        self._quality_cache.clear()
        self._cache_timestamp = None
