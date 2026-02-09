"""
SQLite Database Manager
=======================

Provides persistence for:
- Nano Models
- KV Entries
- Experiment results
- Feedback data
- System statistics
"""

import sqlite3
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import logging

try:
    from ..core.models import NanoModel, KVEntry, NanoLifecycleState
except ImportError:
    from core.models import NanoModel, KVEntry, NanoLifecycleState

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    SQLite database manager for Nano Models framework.
    
    Tables:
    - nano_models: Nano Model metadata and state
    - kv_entries: KV entry data
    - experiments: Experiment configurations and results
    - feedback: User feedback and system observations
    - statistics: System statistics snapshots
    - events: Event log
    """
    
    def __init__(self, db_path: str = "data/nano_models.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()
        
        logger.info(f"DatabaseManager initialized: {db_path}")
    
    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Nano Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nano_models (
                nano_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                creation_mode TEXT,
                confidence_score REAL,
                innovation_domain TEXT,
                lora_rank INTEGER,
                lora_alpha REAL,
                target_modules TEXT,
                activation_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                total_contribution REAL DEFAULT 0,
                created_at TEXT,
                last_activated TEXT,
                lora_weights_json TEXT,
                bound_kv_ids_json TEXT,
                metadata_json TEXT
            )
        """)
        
        # KV Entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kv_entries (
                kv_id TEXT PRIMARY KEY,
                nano_id TEXT,
                access_mode TEXT,
                version INTEGER DEFAULT 1,
                access_count INTEGER DEFAULT 0,
                created_at TEXT,
                key_data BLOB,
                value_data BLOB,
                metadata_json TEXT,
                FOREIGN KEY (nano_id) REFERENCES nano_models(nano_id)
            )
        """)
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                config_json TEXT,
                status TEXT DEFAULT 'pending',
                started_at TEXT,
                completed_at TEXT,
                results_json TEXT,
                metrics_json TEXT
            )
        """)
        
        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                feedback_type TEXT,
                nano_id TEXT,
                query_hash TEXT,
                rating REAL,
                success INTEGER,
                details_json TEXT,
                FOREIGN KEY (nano_id) REFERENCES nano_models(nano_id)
            )
        """)
        
        # Statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statistics (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                component TEXT,
                stats_json TEXT
            )
        """)
        
        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                component TEXT,
                entity_id TEXT,
                details_json TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nano_state ON nano_models(state)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kv_nano ON kv_entries(nano_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_nano ON feedback(nano_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
        
        self.conn.commit()
        logger.debug("Database tables created/verified")
    
    # ==================== Nano Model Operations ====================
    
    def save_nano_model(self, nano: NanoModel):
        """Save a Nano Model to the database."""
        cursor = self.conn.cursor()
        
        # Serialize LoRA weights
        lora_weights_json = json.dumps({
            name: lora.to_dict()
            for name, lora in nano.lora_weights.items()
        })
        
        cursor.execute("""
            INSERT OR REPLACE INTO nano_models (
                nano_id, state, creation_mode, confidence_score, innovation_domain,
                lora_rank, lora_alpha, target_modules, activation_count,
                success_count, failure_count, total_contribution,
                created_at, last_activated, lora_weights_json,
                bound_kv_ids_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            nano.nano_id,
            nano.state.value,
            nano.creation_mode,
            nano.confidence_score,
            nano.innovation_domain,
            nano.lora_rank,
            nano.lora_alpha,
            json.dumps(nano.target_modules),
            nano.activation_count,
            nano.success_count,
            nano.failure_count,
            nano.total_contribution,
            nano.created_at.isoformat(),
            nano.last_activated.isoformat(),
            lora_weights_json,
            json.dumps(list(nano.bound_kv_ids)),
            json.dumps(nano.metadata),
        ))
        
        self.conn.commit()
        logger.debug(f"Saved Nano Model {nano.nano_id}")
    
    def load_nano_model(self, nano_id: str) -> Optional[NanoModel]:
        """Load a Nano Model from the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM nano_models WHERE nano_id = ?", (nano_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_nano_model(row)
    
    def load_all_nano_models(
        self,
        state_filter: Optional[str] = None,
    ) -> List[NanoModel]:
        """Load all Nano Models, optionally filtered by state."""
        cursor = self.conn.cursor()
        
        if state_filter:
            cursor.execute(
                "SELECT * FROM nano_models WHERE state = ?",
                (state_filter,)
            )
        else:
            cursor.execute("SELECT * FROM nano_models")
        
        return [self._row_to_nano_model(row) for row in cursor.fetchall()]
    
    def _row_to_nano_model(self, row: sqlite3.Row) -> NanoModel:
        """Convert database row to NanoModel."""
        from ..core.models import LoRAWeights
        
        lora_weights_data = json.loads(row["lora_weights_json"])
        lora_weights = {
            name: LoRAWeights.from_dict(data)
            for name, data in lora_weights_data.items()
        }
        
        return NanoModel(
            nano_id=row["nano_id"],
            lora_weights=lora_weights,
            bound_kv_ids=set(json.loads(row["bound_kv_ids_json"])),
            state=NanoLifecycleState(row["state"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_activated=datetime.fromisoformat(row["last_activated"]),
            activation_count=row["activation_count"],
            creation_mode=row["creation_mode"],
            confidence_score=row["confidence_score"],
            innovation_domain=row["innovation_domain"] or "",
            lora_rank=row["lora_rank"],
            lora_alpha=row["lora_alpha"],
            target_modules=json.loads(row["target_modules"]),
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            total_contribution=row["total_contribution"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )
    
    def delete_nano_model(self, nano_id: str) -> bool:
        """Delete a Nano Model from the database."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM nano_models WHERE nano_id = ?", (nano_id,))
        self.conn.commit()
        return cursor.rowcount > 0
    
    def update_nano_state(self, nano_id: str, state: NanoLifecycleState):
        """Update Nano Model state."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE nano_models SET state = ? WHERE nano_id = ?",
            (state.value, nano_id)
        )
        self.conn.commit()
    
    # ==================== KV Entry Operations ====================
    
    def save_kv_entry(self, entry: KVEntry):
        """Save a KV entry to the database."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO kv_entries (
                kv_id, nano_id, access_mode, version, access_count,
                created_at, key_data, value_data, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.kv_id,
            entry.nano_id,
            entry.access_mode,
            entry.version,
            entry.access_count,
            entry.created_at.isoformat(),
            entry.key.tobytes(),
            entry.value.tobytes(),
            json.dumps(entry.metadata),
        ))
        
        self.conn.commit()
    
    def load_kv_entry(self, kv_id: str) -> Optional[KVEntry]:
        """Load a KV entry from the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM kv_entries WHERE kv_id = ?", (kv_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_kv_entry(row)
    
    def load_kv_entries_for_nano(self, nano_id: str) -> List[KVEntry]:
        """Load all KV entries for a Nano Model."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM kv_entries WHERE nano_id = ?",
            (nano_id,)
        )
        return [self._row_to_kv_entry(row) for row in cursor.fetchall()]
    
    def _row_to_kv_entry(self, row: sqlite3.Row) -> KVEntry:
        """Convert database row to KVEntry."""
        return KVEntry(
            kv_id=row["kv_id"],
            key=np.frombuffer(row["key_data"], dtype=np.float64),
            value=np.frombuffer(row["value_data"], dtype=np.float64),
            nano_id=row["nano_id"],
            access_mode=row["access_mode"],
            version=row["version"],
            access_count=row["access_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )
    
    # ==================== Experiment Operations ====================
    
    def save_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str,
        config: Dict[str, Any],
        status: str = "pending",
    ):
        """Save an experiment configuration."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO experiments (
                experiment_id, name, description, config_json, status, started_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            name,
            description,
            json.dumps(config),
            status,
            datetime.now().isoformat(),
        ))
        self.conn.commit()
    
    def update_experiment_results(
        self,
        experiment_id: str,
        results: Dict[str, Any],
        metrics: Dict[str, Any],
        status: str = "completed",
    ):
        """Update experiment results."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE experiments SET
                results_json = ?,
                metrics_json = ?,
                status = ?,
                completed_at = ?
            WHERE experiment_id = ?
        """, (
            json.dumps(results),
            json.dumps(metrics),
            status,
            datetime.now().isoformat(),
            experiment_id,
        ))
        self.conn.commit()
    
    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load an experiment."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return {
            "experiment_id": row["experiment_id"],
            "name": row["name"],
            "description": row["description"],
            "config": json.loads(row["config_json"]) if row["config_json"] else {},
            "status": row["status"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "results": json.loads(row["results_json"]) if row["results_json"] else {},
            "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else {},
        }
    
    def list_experiments(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments."""
        cursor = self.conn.cursor()
        
        if status_filter:
            cursor.execute(
                "SELECT * FROM experiments WHERE status = ? ORDER BY started_at DESC",
                (status_filter,)
            )
        else:
            cursor.execute("SELECT * FROM experiments ORDER BY started_at DESC")
        
        return [
            {
                "experiment_id": row["experiment_id"],
                "name": row["name"],
                "status": row["status"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
            }
            for row in cursor.fetchall()
        ]
    
    # ==================== Feedback Operations ====================
    
    def save_feedback(
        self,
        feedback_type: str,
        nano_id: Optional[str],
        query_hash: str,
        rating: Optional[float],
        success: bool,
        details: Dict[str, Any],
    ) -> int:
        """Save feedback data."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (
                timestamp, feedback_type, nano_id, query_hash,
                rating, success, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            feedback_type,
            nano_id,
            query_hash,
            rating,
            1 if success else 0,
            json.dumps(details),
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_feedback_for_nano(self, nano_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a Nano Model."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM feedback WHERE nano_id = ? ORDER BY timestamp DESC",
            (nano_id,)
        )
        return [
            {
                "feedback_id": row["feedback_id"],
                "timestamp": row["timestamp"],
                "feedback_type": row["feedback_type"],
                "query_hash": row["query_hash"],
                "rating": row["rating"],
                "success": bool(row["success"]),
                "details": json.loads(row["details_json"]) if row["details_json"] else {},
            }
            for row in cursor.fetchall()
        ]
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get feedback summary statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM feedback")
        total = cursor.fetchone()["total"]
        
        cursor.execute("SELECT AVG(rating) as avg_rating FROM feedback WHERE rating IS NOT NULL")
        avg_rating = cursor.fetchone()["avg_rating"]
        
        cursor.execute("SELECT SUM(success) as success_count FROM feedback")
        success_count = cursor.fetchone()["success_count"] or 0
        
        return {
            "total_feedback": total,
            "average_rating": avg_rating,
            "success_count": success_count,
            "success_rate": success_count / total if total > 0 else 0,
        }
    
    # ==================== Statistics Operations ====================
    
    def save_statistics(self, component: str, stats: Dict[str, Any]):
        """Save a statistics snapshot."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO statistics (timestamp, component, stats_json)
            VALUES (?, ?, ?)
        """, (
            datetime.now().isoformat(),
            component,
            json.dumps(stats),
        ))
        self.conn.commit()
    
    def get_statistics_history(
        self,
        component: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get statistics history for a component."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM statistics
            WHERE component = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (component, limit))
        
        return [
            {
                "snapshot_id": row["snapshot_id"],
                "timestamp": row["timestamp"],
                "stats": json.loads(row["stats_json"]),
            }
            for row in cursor.fetchall()
        ]
    
    # ==================== Event Operations ====================
    
    def log_event(
        self,
        event_type: str,
        component: str,
        entity_id: Optional[str],
        details: Dict[str, Any],
    ):
        """Log an event."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO events (timestamp, event_type, component, entity_id, details_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            event_type,
            component,
            entity_id,
            json.dumps(details),
        ))
        self.conn.commit()
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        component: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get events with optional filtering."""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if component:
            query += " AND component = ?"
            params.append(component)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        return [
            {
                "event_id": row["event_id"],
                "timestamp": row["timestamp"],
                "event_type": row["event_type"],
                "component": row["component"],
                "entity_id": row["entity_id"],
                "details": json.loads(row["details_json"]) if row["details_json"] else {},
            }
            for row in cursor.fetchall()
        ]
    
    # ==================== Utility Operations ====================
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    def vacuum(self):
        """Optimize database by running VACUUM."""
        self.conn.execute("VACUUM")
        logger.info("Database vacuumed")
    
    def get_table_counts(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        cursor = self.conn.cursor()
        tables = ["nano_models", "kv_entries", "experiments", "feedback", "statistics", "events"]
        
        counts = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            counts[table] = cursor.fetchone()["count"]
        
        return counts
