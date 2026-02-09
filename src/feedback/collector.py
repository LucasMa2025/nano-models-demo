"""
Feedback Collector
==================

Collects feedback from various sources:
- System observations (automatic)
- User ratings (manual)
- Performance metrics (automatic)

Used for:
- Nano Model quality assessment
- Innovation detection tuning
- Lifecycle management decisions
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import logging

from ..storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback."""
    SYSTEM_OBSERVATION = "system_observation"  # Automatic system feedback
    USER_RATING = "user_rating"                # Manual user rating
    PERFORMANCE_METRIC = "performance_metric"  # Performance-based feedback
    VALIDATION_RESULT = "validation_result"    # Validation outcome
    ERROR_REPORT = "error_report"              # Error or failure report


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    feedback_id: str
    feedback_type: FeedbackType
    timestamp: datetime
    nano_id: Optional[str]
    query_hash: str
    rating: Optional[float]  # 0-1 scale
    success: bool
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type.value,
            "timestamp": self.timestamp.isoformat(),
            "nano_id": self.nano_id,
            "query_hash": self.query_hash,
            "rating": self.rating,
            "success": self.success,
            "details": self.details,
        }


class FeedbackCollector:
    """
    Collects and manages feedback for the Nano Models framework.
    
    Feedback Sources:
    1. System Observations: Automatic feedback from inference
    2. User Ratings: Manual quality ratings
    3. Performance Metrics: Latency, accuracy, etc.
    4. Validation Results: TRIAL → ACTIVE transitions
    5. Error Reports: Failures and anomalies
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the feedback collector.
        
        Args:
            db_manager: Optional database manager for persistence
        """
        self.db = db_manager
        
        # In-memory feedback buffer
        self.feedback_buffer: List[FeedbackEntry] = []
        self.buffer_limit = 1000
        
        # Statistics
        self.total_feedback = 0
        self.feedback_by_type: Dict[FeedbackType, int] = {t: 0 for t in FeedbackType}
        self.feedback_by_nano: Dict[str, int] = {}
        
        logger.info("FeedbackCollector initialized")
    
    def collect_system_observation(
        self,
        nano_id: Optional[str],
        query_embedding: Any,
        success: bool,
        innovation_score: float,
        fusion_contribution: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Collect automatic system observation feedback.
        
        Args:
            nano_id: ID of the Nano Model (if any)
            query_embedding: Query embedding for hashing
            success: Whether the inference was successful
            innovation_score: Innovation detection score
            fusion_contribution: Nano Model's contribution weight
            details: Additional details
            
        Returns:
            Feedback ID
        """
        query_hash = self._hash_query(query_embedding)
        
        entry = FeedbackEntry(
            feedback_id=self._generate_feedback_id(),
            feedback_type=FeedbackType.SYSTEM_OBSERVATION,
            timestamp=datetime.now(),
            nano_id=nano_id,
            query_hash=query_hash,
            rating=fusion_contribution if success else 0.0,
            success=success,
            details={
                "innovation_score": innovation_score,
                "fusion_contribution": fusion_contribution,
                **(details or {}),
            },
        )
        
        return self._store_feedback(entry)
    
    def collect_user_rating(
        self,
        nano_id: Optional[str],
        query_hash: str,
        rating: float,
        comment: Optional[str] = None,
    ) -> str:
        """
        Collect manual user rating feedback.
        
        Args:
            nano_id: ID of the Nano Model
            query_hash: Hash of the query
            rating: User rating (0-1 scale)
            comment: Optional user comment
            
        Returns:
            Feedback ID
        """
        entry = FeedbackEntry(
            feedback_id=self._generate_feedback_id(),
            feedback_type=FeedbackType.USER_RATING,
            timestamp=datetime.now(),
            nano_id=nano_id,
            query_hash=query_hash,
            rating=rating,
            success=rating >= 0.5,
            details={"comment": comment} if comment else {},
        )
        
        return self._store_feedback(entry)
    
    def collect_performance_metric(
        self,
        nano_id: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
    ) -> str:
        """
        Collect performance metric feedback.
        
        Args:
            nano_id: ID of the Nano Model
            metric_name: Name of the metric
            metric_value: Metric value
            threshold: Threshold for success
            
        Returns:
            Feedback ID
        """
        success = metric_value >= threshold
        
        entry = FeedbackEntry(
            feedback_id=self._generate_feedback_id(),
            feedback_type=FeedbackType.PERFORMANCE_METRIC,
            timestamp=datetime.now(),
            nano_id=nano_id,
            query_hash="",
            rating=metric_value,
            success=success,
            details={
                "metric_name": metric_name,
                "metric_value": metric_value,
                "threshold": threshold,
            },
        )
        
        return self._store_feedback(entry)
    
    def collect_validation_result(
        self,
        nano_id: str,
        validation_type: str,
        passed: bool,
        score: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Collect validation result feedback.
        
        Args:
            nano_id: ID of the Nano Model
            validation_type: Type of validation
            passed: Whether validation passed
            score: Validation score
            details: Additional details
            
        Returns:
            Feedback ID
        """
        entry = FeedbackEntry(
            feedback_id=self._generate_feedback_id(),
            feedback_type=FeedbackType.VALIDATION_RESULT,
            timestamp=datetime.now(),
            nano_id=nano_id,
            query_hash="",
            rating=score,
            success=passed,
            details={
                "validation_type": validation_type,
                **(details or {}),
            },
        )
        
        return self._store_feedback(entry)
    
    def collect_error_report(
        self,
        nano_id: Optional[str],
        error_type: str,
        error_message: str,
        query_hash: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Collect error report feedback.
        
        Args:
            nano_id: ID of the Nano Model (if applicable)
            error_type: Type of error
            error_message: Error message
            query_hash: Hash of the query (if applicable)
            details: Additional details
            
        Returns:
            Feedback ID
        """
        entry = FeedbackEntry(
            feedback_id=self._generate_feedback_id(),
            feedback_type=FeedbackType.ERROR_REPORT,
            timestamp=datetime.now(),
            nano_id=nano_id,
            query_hash=query_hash,
            rating=0.0,
            success=False,
            details={
                "error_type": error_type,
                "error_message": error_message,
                **(details or {}),
            },
        )
        
        return self._store_feedback(entry)
    
    def _store_feedback(self, entry: FeedbackEntry) -> str:
        """Store feedback entry."""
        # Add to buffer
        self.feedback_buffer.append(entry)
        
        # Update statistics
        self.total_feedback += 1
        self.feedback_by_type[entry.feedback_type] += 1
        
        if entry.nano_id:
            self.feedback_by_nano[entry.nano_id] = self.feedback_by_nano.get(entry.nano_id, 0) + 1
        
        # Persist to database
        if self.db:
            self.db.save_feedback(
                feedback_type=entry.feedback_type.value,
                nano_id=entry.nano_id,
                query_hash=entry.query_hash,
                rating=entry.rating,
                success=entry.success,
                details=entry.details,
            )
        
        # Trim buffer if needed
        if len(self.feedback_buffer) > self.buffer_limit:
            self.feedback_buffer = self.feedback_buffer[-self.buffer_limit:]
        
        logger.debug(f"Stored feedback {entry.feedback_id} (type={entry.feedback_type.value})")
        
        return entry.feedback_id
    
    def _generate_feedback_id(self) -> str:
        """Generate unique feedback ID."""
        content = f"{datetime.now().isoformat()}_{self.total_feedback}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"fb_{hash_val}"
    
    def _hash_query(self, query_embedding: Any) -> str:
        """Hash a query embedding."""
        if hasattr(query_embedding, 'tobytes'):
            content = query_embedding.tobytes()
        else:
            content = str(query_embedding).encode()
        return hashlib.md5(content).hexdigest()[:16]
    
    def get_feedback_for_nano(self, nano_id: str) -> List[FeedbackEntry]:
        """Get all feedback for a Nano Model."""
        return [f for f in self.feedback_buffer if f.nano_id == nano_id]
    
    def get_recent_feedback(self, limit: int = 100) -> List[FeedbackEntry]:
        """Get recent feedback entries."""
        return self.feedback_buffer[-limit:]
    
    def get_feedback_by_type(self, feedback_type: FeedbackType) -> List[FeedbackEntry]:
        """Get feedback by type."""
        return [f for f in self.feedback_buffer if f.feedback_type == feedback_type]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        return {
            "total_feedback": self.total_feedback,
            "by_type": {t.value: c for t, c in self.feedback_by_type.items()},
            "by_nano": self.feedback_by_nano,
            "buffer_size": len(self.feedback_buffer),
            "success_rate": sum(1 for f in self.feedback_buffer if f.success) / max(len(self.feedback_buffer), 1),
            "avg_rating": sum(f.rating for f in self.feedback_buffer if f.rating is not None) / max(sum(1 for f in self.feedback_buffer if f.rating is not None), 1),
        }
    
    def clear_buffer(self):
        """Clear the feedback buffer."""
        self.feedback_buffer.clear()
        logger.info("Feedback buffer cleared")
