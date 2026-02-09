"""
Nano Models Demo System
=======================

A complete implementation of the Nano Models framework for innovation-triggered
derivation patches in frozen Large Language Models.

Core Components:
- Innovation Detector: Three-stage detection mechanism
- Nano Model Registry: Lifecycle management
- KV Store: Versioned key-value storage with conflict detection
- Fusion Engine: Conflict-aware multi-Nano fusion
- Experiment System: Automated experimentation and metrics
- Feedback System: Performance tracking and adaptation

Author: Ma MingJian
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Ma MingJian"

from .core.models import NanoModel, KVEntry, NanoLifecycleState
from .core.innovation_detector import InnovationDetector
from .core.registry import NanoRegistry
from .core.factory import NanoModelFactory
from .core.fusion import ConflictAwareFusion
from .storage.kv_store import VersionedKVStore
from .storage.database import DatabaseManager

__all__ = [
    "NanoModel",
    "KVEntry", 
    "NanoLifecycleState",
    "InnovationDetector",
    "NanoRegistry",
    "NanoModelFactory",
    "ConflictAwareFusion",
    "VersionedKVStore",
    "DatabaseManager",
]
