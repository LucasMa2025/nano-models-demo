"""
Nano Models Demo - Main Entry Point
====================================

This script demonstrates the complete Nano Models framework.

Usage:
    python main.py                    # Run demo
    python main.py --experiment       # Run experiment
    python main.py --ablation         # Run ablation study
"""

import argparse
import logging
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_demo():
    """Run a demonstration of the Nano Models system."""
    from src.system import NanoModelSystem, SystemConfig
    
    print("=" * 70)
    print("NANO MODELS DEMO SYSTEM")
    print("=" * 70)
    print()
    
    # Initialize system
    print("[1/6] Initializing system...")
    config = SystemConfig(
        hidden_dim=128,
        innovation_threshold=0.7,
        sample_buffer_threshold=10,
        auto_create_nanos=True,
    )
    system = NanoModelSystem(config)
    print(f"      System initialized with hidden_dim={config.hidden_dim}")
    print()
    
    # Inject some global knowledge
    print("[2/6] Injecting global knowledge...")
    for i in range(5):
        key = np.random.randn(128)
        value = np.random.randn(128)
        kv_id = system.inject_global_kv(key, value, {"domain": f"domain_{i}"})
        print(f"      Injected KV entry: {kv_id}")
    print()
    
    # Run some inferences to build semantic subspaces
    print("[3/6] Running initial inferences to build semantic subspaces...")
    for i in range(100):
        query = np.random.randn(128) * 0.5  # Low variance (factual)
        result = system.infer(query)
        if (i + 1) % 25 == 0:
            print(f"      Processed {i + 1}/100 queries")
    print(f"      Innovation rate: {system.innovation_detections / system.total_inferences:.2%}")
    print()
    
    # Create an emergency Nano Model
    print("[4/6] Creating emergency Nano Model...")
    samples = [
        (np.random.randn(128), np.random.randn(128), 0.95)
        for _ in range(3)
    ]
    nano = system.create_nano_emergency(samples, "emergency_test_domain")
    if nano:
        print(f"      Created Nano Model: {nano.nano_id}")
        print(f"      State: {nano.state.value}")
        print(f"      Confidence: {nano.confidence_score:.3f}")
    print()
    
    # Run innovation queries
    print("[5/6] Running innovation queries...")
    innovation_results = []
    for i in range(20):
        query = np.random.randn(128) * 3.0  # High variance (innovation)
        result = system.infer(query)
        innovation_results.append(result)
        
    innovation_detected = sum(1 for r in innovation_results if r.innovation_detected)
    nano_selected = sum(1 for r in innovation_results if r.nanos_selected)
    print(f"      Innovation detected: {innovation_detected}/20")
    print(f"      Nano Models selected: {nano_selected}/20")
    print()
    
    # Get statistics
    print("[6/6] System Statistics:")
    stats = system.get_statistics()
    print(f"      Total inferences: {stats['system']['total_inferences']}")
    print(f"      Innovation detections: {stats['system']['innovation_detections']}")
    print(f"      Innovation rate: {stats['system']['innovation_rate']:.2%}")
    print(f"      Nano activations: {stats['system']['nano_activations']}")
    print(f"      Active Nano Models: {stats['registry']['current_count']}")
    print(f"      KV entries (global): {stats['kv_store']['global']['total_entries']}")
    print()
    
    # Provide feedback
    print("Providing feedback...")
    for result in innovation_results[:5]:
        system.provide_feedback(
            query_hash=result.diagnostics.query_hash,
            rating=0.8,
            comment="Good response",
        )
    
    # Get feedback report
    report = system.get_feedback_report()
    print(f"      Total feedback: {report['statistics']['total_feedback']}")
    print(f"      Overall health: {report['overall_health']}")
    print()
    
    # Clean up
    system.close()
    
    print("=" * 70)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70)


def run_experiment():
    """Run a full experiment."""
    from src.experiments.runner import ExperimentRunner, ExperimentConfig
    from src.storage.database import DatabaseManager
    
    print("=" * 70)
    print("NANO MODELS EXPERIMENT")
    print("=" * 70)
    print()
    
    # Configure experiment
    config = ExperimentConfig(
        name="demo_experiment",
        description="Demonstration experiment for Nano Models",
        num_queries=500,
        innovation_ratio=0.3,
        hidden_dim=128,
        innovation_threshold=0.7,
    )
    
    print(f"Experiment: {config.name}")
    print(f"Queries: {config.num_queries}")
    print(f"Innovation ratio: {config.innovation_ratio}")
    print()
    
    # Initialize database
    db = DatabaseManager("data/experiment.db")
    
    # Run experiment
    print("Running experiment...")
    runner = ExperimentRunner(config, db)
    results = runner.run()
    
    # Print results
    print()
    print("=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)
    print()
    
    metrics = results["metrics"]
    print(f"Overall Accuracy:      {metrics['overall_accuracy']:.4f}")
    print(f"Factual Accuracy:      {metrics['factual_accuracy']:.4f}")
    print(f"Innovation Accuracy:   {metrics['innovation_accuracy']:.4f}")
    print(f"Detection Rate:        {metrics['innovation_detection_rate']:.4f}")
    print(f"False Positive Rate:   {metrics['false_positive_rate']:.4f}")
    print(f"Nano Selection Rate:   {metrics['nano_selection_rate']:.4f}")
    print(f"Nano Models Created:   {metrics['nano_models_created']}")
    print(f"Active Nano Models:    {metrics['active_nano_models']}")
    print(f"Avg Processing Time:   {metrics['avg_processing_time_ms']:.2f} ms")
    print(f"Total Duration:        {results['duration_seconds']:.2f} s")
    print(f"Queries/Second:        {metrics['queries_per_second']:.2f}")
    print()
    
    db.close()
    
    print("=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)


def run_ablation():
    """Run ablation study."""
    from src.experiments.runner import ExperimentRunner, ExperimentConfig, AblationStudy
    
    print("=" * 70)
    print("NANO MODELS ABLATION STUDY")
    print("=" * 70)
    print()
    
    # Base configuration
    base_config = ExperimentConfig(
        name="ablation_base",
        num_queries=200,
        innovation_ratio=0.3,
        hidden_dim=128,
    )
    
    # Run ablation study
    print("Running ablation study...")
    study = AblationStudy(base_config)
    results = study.run_standard_ablations()
    
    # Print comparison table
    print()
    print("=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print()
    
    table = study.get_comparison_table()
    
    # Header
    print(f"{'Ablation':<25} {'Overall Acc':<12} {'Innovation Acc':<15} {'FP Rate':<10} {'Nanos':<8}")
    print("-" * 70)
    
    for row in table:
        print(f"{row['ablation']:<25} {row['overall_accuracy']:<12.4f} {row['innovation_accuracy']:<15.4f} {row['false_positive_rate']:<10.4f} {row['nano_models_created']:<8}")
    
    print()
    print("=" * 70)
    print("ABLATION STUDY COMPLETED")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Nano Models Demo System")
    parser.add_argument("--experiment", action="store_true", help="Run experiment")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    
    args = parser.parse_args()
    
    if args.experiment:
        run_experiment()
    elif args.ablation:
        run_ablation()
    else:
        run_demo()


if __name__ == "__main__":
    main()
