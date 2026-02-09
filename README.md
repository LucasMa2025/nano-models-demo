# Nano Models Demo System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete implementation of the **Nano Models** framework for innovation-triggered derivation patches in frozen Large Language Models.

[English](#english) | [中文](#中文)

---

<a name="english"></a>

## Overview

Nano Models are **innovation-triggered lightweight derivation patches** that dynamically create frozen LoRA sub-modules only when genuine innovation gaps are detected. Unlike traditional fine-tuning or RAG approaches, Nano Models modify _how_ the model reasons, not just _what_ it knows.

### Key Features

-   **Three-Stage Innovation Detection**: Projection error + Attention anomaly + KV coverage
-   **Hierarchical KV Binding**: Global, Shared, and Exclusive access modes
-   **Conflict-Aware Fusion**: Handles multiple Nano Model outputs with conflict detection
-   **Lifecycle Management**: TRIAL → ACTIVE → DORMANT → DEPRECATED state machine
-   **Experiment System**: Automated experimentation with metrics collection
-   **Feedback System**: Quality assessment and parameter tuning recommendations

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         NANO MODEL SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐                                                                │
│  │   Query     │                                                                │
│  │   Input     │                                                                │
│  └──────┬──────┘                                                                │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    INNOVATION DETECTOR                                  │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │    │
│  │  │ Stage 1:        │  │ Stage 2:        │  │ Stage 3:        │         │    │
│  │  │ Projection      │  │ Attention       │  │ KV Coverage     │         │    │
│  │  │ Error           │  │ Anomaly (MMD)   │  │ Check           │         │    │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │    │
│  │           │                    │                    │                  │    │
│  │           └────────────────────┼────────────────────┘                  │    │
│  │                                ▼                                       │    │
│  │                    InnovationScore = w1*PE + w2*SN + w3*(1-KVC)        │    │
│  └─────────────────────────────────┬───────────────────────────────────────┘    │
│                                    │                                            │
│              ┌─────────────────────┴─────────────────────┐                      │
│              │                                           │                      │
│              ▼                                           ▼                      │
│  ┌───────────────────────┐                   ┌───────────────────────┐          │
│  │  Score < Threshold    │                   │  Score >= Threshold   │          │
│  │  (No Innovation)      │                   │  (Innovation Gap)     │          │
│  └───────────┬───────────┘                   └───────────┬───────────┘          │
│              │                                           │                      │
│              ▼                                           ▼                      │
│  ┌───────────────────────┐           ┌───────────────────────────────────┐      │
│  │  Update Semantic      │           │         NANO REGISTRY             │      │
│  │  Subspaces (PCA)      │           │  ┌─────────────────────────────┐  │      │
│  │  Update Reference     │           │  │  Select Matching Nanos      │  │      │
│  │  Patterns             │           │  │  (KV Hit Detection)         │  │      │
│  └───────────────────────┘           │  └──────────────┬──────────────┘  │      │
│                                      │                 │                 │      │
│                                      │    ┌────────────┴────────────┐    │      │
│                                      │    │                         │    │      │
│                                      │    ▼                         ▼    │      │
│                                      │  ┌─────────┐           ┌─────────┐│      │
│                                      │  │ Nano_1  │    ...    │ Nano_k  ││      │
│                                      │  │ + KV_1  │           │ + KV_k  ││      │
│                                      │  └────┬────┘           └────┬────┘│      │
│                                      └───────┼─────────────────────┼─────┘      │
│                                              │                     │            │
│                                              └──────────┬──────────┘            │
│                                                         │                       │
│                                                         ▼                       │
│                                      ┌───────────────────────────────────┐      │
│                                      │     CONFLICT-AWARE FUSION         │      │
│                                      │  ┌─────────────────────────────┐  │      │
│                                      │  │ 1. Conflict Detection       │  │      │
│                                      │  │    (Direction + Magnitude)  │  │      │
│                                      │  │ 2. Adaptive Resolution      │  │      │
│                                      │  │ 3. Weighted Avg / Winner    │  │      │
│                                      │  └─────────────────────────────┘  │      │
│                                      └───────────────┬───────────────────┘      │
│                                                      │                          │
│                                                      ▼                          │
│                                      ┌───────────────────────────────────┐      │
│                                      │        OUTPUT FUSION              │      │
│                                      │  O_final = O_base + α*O_AGA       │      │
│                                      │           + γ*O_nano              │      │
│                                      └───────────────┬───────────────────┘      │
│                                                      │                          │
│                                                      ▼                          │
│                                               ┌─────────────┐                   │
│                                               │   Output    │                   │
│                                               └─────────────┘                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Lifecycle State Machine

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         NANO MODEL LIFECYCLE                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                              ┌──────────────┐                                   │
│                              │    CREATE    │                                   │
│                              └──────┬───────┘                                   │
│                                     │                                           │
│                    ┌────────────────┴────────────────┐                          │
│                    │                                 │                          │
│                    ▼                                 ▼                          │
│           ┌────────────────┐                ┌────────────────┐                  │
│           │     TRIAL      │                │     ACTIVE     │                  │
│           │  (Emergency/   │                │   (Standard/   │                  │
│           │   Few-shot)    │                │   Confident)   │                  │
│           └───────┬────────┘                └───────┬────────┘                  │
│                   │                                 │                           │
│       ┌───────────┴───────────┐                     │                           │
│       │                       │                     │                           │
│       ▼                       ▼                     │                           │
│  ┌─────────┐           ┌─────────────┐              │                           │
│  │ ACTIVE  │           │ DEPRECATED  │              │                           │
│  │(Passed) │           │ (Failed)    │              │                           │
│  └────┬────┘           └──────┬──────┘              │                           │
│       │                       │                     │                           │
│       └───────────────────────┼─────────────────────┘                           │
│                               │                                                 │
│                               │  Low Usage                                      │
│                               ▼                                                 │
│                        ┌─────────────┐                                          │
│                        │   DORMANT   │◄─────────────────────────────┐           │
│                        └──────┬──────┘                              │           │
│                               │                                     │           │
│              ┌────────────────┴────────────────┐                    │           │
│              │                                 │                    │           │
│              ▼                                 ▼                    │           │
│       ┌─────────────┐                  ┌─────────────┐              │           │
│       │ DEPRECATED  │                  │   ACTIVE    │──────────────┘           │
│       │ (Prolonged  │                  │(Reactivated)│  (KV Hit)                │
│       │  Inactivity)│                  └─────────────┘                          │
│       └──────┬──────┘                                                           │
│              │                                                                  │
│              │  Grace Period                                                    │
│              ▼                                                                  │
│       ┌─────────────┐                                                           │
│       │   DESTROY   │                                                           │
│       └─────────────┘                                                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/LucasMa2025/nano-models-demo.git
cd nano-models-demo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.system import NanoModelSystem, SystemConfig
import numpy as np

# Initialize system
config = SystemConfig(
    hidden_dim=256,
    innovation_threshold=0.7,
)
system = NanoModelSystem(config)

# Run inference
query = np.random.randn(256)
result = system.infer(query)

print(f"Innovation detected: {result.innovation_detected}")
print(f"Innovation score: {result.innovation_score:.4f}")
print(f"Nanos selected: {result.nanos_selected}")

# Inject knowledge
system.inject_global_kv(
    key=np.random.randn(256),
    value=np.random.randn(256),
)

# Create emergency Nano Model
samples = [(np.random.randn(256), np.random.randn(256), 0.9) for _ in range(3)]
nano = system.create_nano_emergency(samples, "novel_domain")

# Get statistics
stats = system.get_statistics()
print(f"Total inferences: {stats['system']['total_inferences']}")

# Clean up
system.close()
```

## Running Experiments

```python
from src.experiments.runner import ExperimentRunner, ExperimentConfig

# Configure experiment
config = ExperimentConfig(
    name="innovation_detection_test",
    num_queries=1000,
    innovation_ratio=0.3,
    innovation_threshold=0.7,
)

# Run experiment
runner = ExperimentRunner(config)
results = runner.run()

print(f"Overall accuracy: {results['metrics']['overall_accuracy']:.4f}")
print(f"Innovation accuracy: {results['metrics']['innovation_accuracy']:.4f}")
print(f"Nano Models created: {results['metrics']['nano_models_created']}")
```

## Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run with verbose output
python tests/run_tests.py -v

# Run specific test module
python tests/run_tests.py models
python tests/run_tests.py innovation_detector
python tests/run_tests.py registry
```

## Project Structure

```
NanoModelDemo/
├── src/
│   ├── core/
│   │   ├── models.py           # Core data models
│   │   ├── innovation_detector.py  # Three-stage detection
│   │   ├── registry.py         # Lifecycle management
│   │   ├── factory.py          # Nano Model creation
│   │   └── fusion.py           # Conflict-aware fusion
│   ├── storage/
│   │   ├── kv_store.py         # Hierarchical KV store
│   │   └── database.py         # SQLite persistence
│   ├── experiments/
│   │   ├── runner.py           # Experiment runner
│   │   └── metrics.py          # Metrics collection
│   ├── feedback/
│   │   ├── collector.py        # Feedback collection
│   │   └── analyzer.py         # Feedback analysis
│   └── system.py               # Integrated system
├── tests/
│   ├── test_models.py
│   ├── test_innovation_detector.py
│   ├── test_registry.py
│   ├── test_factory.py
│   ├── test_fusion.py
│   ├── test_kv_store.py
│   ├── test_system.py
│   └── run_tests.py
├── data/                       # Database storage
├── requirements.txt
└── README.md
```

## Future Work

### Short-term (1-3 months)

1. **Self-supervised Innovation Detection**: Remove need for ground truth labels
2. **Real LLM Integration**: Connect to actual transformer models (LLaMA, Mistral)
3. **Distributed KV Store**: Scale to multi-node deployments
4. **Web Dashboard**: Real-time monitoring and visualization

### Medium-term (3-6 months)

1. **Multi-Nano Composition**: Handle scenarios requiring multiple novel derivations
2. **Cross-modal Nano Models**: Extend to vision-language models
3. **Predictive Innovation**: Forecast innovation needs and pre-create Nanos
4. **Formal Verification**: Prove safety properties of Nano Model outputs

### Long-term (6-12 months)

1. **Federated Nano Learning**: Collaborative Nano creation across organizations
2. **Nano Model Marketplace**: Share and discover domain-specific Nanos
3. **Automated Domain Adaptation**: Self-evolving Nano ecosystems
4. **Integration with AGA**: Full AGA + Nano Model production system

---

<a name="中文"></a>

## 概述

Nano Models 是**创新触发的轻量级推导补丁**，仅在检测到真正的创新缺口时才动态创建冻结的 LoRA 子模块。与传统的微调或 RAG 方法不同，Nano Models 修改的是模型*如何推理*，而不仅仅是它*知道什么*。

### 核心特性

-   **三阶段创新检测**：投影误差 + 注意力异常 + KV 覆盖率
-   **分层 KV 绑定**：全局、共享和独占访问模式
-   **冲突感知融合**：处理多个 Nano Model 输出的冲突检测
-   **生命周期管理**：TRIAL → ACTIVE → DORMANT → DEPRECATED 状态机
-   **实验系统**：自动化实验与指标收集
-   **反馈系统**：质量评估和参数调优建议

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         NANO MODEL 系统架构                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐                                                                │
│  │   Query     │                                                                │
│  │   Input     │                                                                │
│  └──────┬──────┘                                                                │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    INNOVATION DETECTOR (创新检测器)                      │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │    │
│  │  │ Stage 1:        │  │ Stage 2:        │  │ Stage 3:        │         │    │
│  │  │ Projection      │  │ Attention       │  │ KV Coverage     │         │    │
│  │  │ Error (投影误差) │  │ Anomaly (MMD)   │  │ Check (覆盖检查) │         │    │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │    │
│  │           │                    │                    │                  │    │
│  │           └────────────────────┼────────────────────┘                  │    │
│  │                                ▼                                       │    │
│  │                    InnovationScore = w1*PE + w2*SN + w3*(1-KVC)        │    │
│  └─────────────────────────────────┬───────────────────────────────────────┘    │
│                                    │                                            │
│              ┌─────────────────────┴─────────────────────┐                      │
│              │                                           │                      │
│              ▼                                           ▼                      │
│  ┌───────────────────────┐                   ┌───────────────────────┐          │
│  │  Score < Threshold    │                   │  Score >= Threshold   │          │
│  │  (无创新)              │                   │  (创新缺口)            │          │
│  └───────────┬───────────┘                   └───────────┬───────────┘          │
│              │                                           │                      │
│              ▼                                           ▼                      │
│  ┌───────────────────────┐           ┌───────────────────────────────────┐      │
│  │  更新语义子空间 (PCA)  │           │         NANO REGISTRY             │      │
│  │  更新参考模式          │           │  ┌─────────────────────────────┐  │      │
│  └───────────────────────┘           │  │  选择匹配的 Nano Models     │  │      │
│                                      │  │  (KV 命中检测)              │  │      │
│                                      │  └──────────────┬──────────────┘  │      │
│                                      └─────────────────┼─────────────────┘      │
│                                                        │                        │
│                                                        ▼                        │
│                                      ┌───────────────────────────────────┐      │
│                                      │     CONFLICT-AWARE FUSION         │      │
│                                      │     (冲突感知融合)                 │      │
│                                      └───────────────┬───────────────────┘      │
│                                                      │                          │
│                                                      ▼                          │
│                                      ┌───────────────────────────────────┐      │
│                                      │        OUTPUT FUSION              │      │
│                                      │  O_final = O_base + α*O_AGA       │      │
│                                      │           + γ*O_nano              │      │
│                                      └───────────────┬───────────────────┘      │
│                                                      │                          │
│                                                      ▼                          │
│                                               ┌─────────────┐                   │
│                                               │   Output    │                   │
│                                               └─────────────┘                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/LucasMa2025/nano-models-demo.git
cd nano-models-demo

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

```python
from src.system import NanoModelSystem, SystemConfig
import numpy as np

# 初始化系统
config = SystemConfig(
    hidden_dim=256,
    innovation_threshold=0.7,
)
system = NanoModelSystem(config)

# 运行推理
query = np.random.randn(256)
result = system.infer(query)

print(f"检测到创新: {result.innovation_detected}")
print(f"创新分数: {result.innovation_score:.4f}")
print(f"选中的 Nanos: {result.nanos_selected}")

# 注入知识
system.inject_global_kv(
    key=np.random.randn(256),
    value=np.random.randn(256),
)

# 创建紧急 Nano Model
samples = [(np.random.randn(256), np.random.randn(256), 0.9) for _ in range(3)]
nano = system.create_nano_emergency(samples, "novel_domain")

# 获取统计信息
stats = system.get_statistics()
print(f"总推理次数: {stats['system']['total_inferences']}")

# 清理
system.close()
```

## 运行测试

```bash
# 运行所有测试
python tests/run_tests.py

# 详细输出
python tests/run_tests.py -v

# 运行特定测试模块
python tests/run_tests.py models
python tests/run_tests.py innovation_detector
```

## 项目结构

```
NanoModelDemo/
├── src/
│   ├── core/
│   │   ├── models.py           # 核心数据模型
│   │   ├── innovation_detector.py  # 三阶段检测
│   │   ├── registry.py         # 生命周期管理
│   │   ├── factory.py          # Nano Model 创建
│   │   └── fusion.py           # 冲突感知融合
│   ├── storage/
│   │   ├── kv_store.py         # 分层 KV 存储
│   │   └── database.py         # SQLite 持久化
│   ├── experiments/
│   │   ├── runner.py           # 实验运行器
│   │   └── metrics.py          # 指标收集
│   ├── feedback/
│   │   ├── collector.py        # 反馈收集
│   │   └── analyzer.py         # 反馈分析
│   └── system.py               # 集成系统
├── tests/                      # 单元测试
├── data/                       # 数据库存储
├── requirements.txt
└── README.md
```

## 未来工作

### 短期 (1-3 个月)

1. **自监督创新检测**：消除对标注数据的依赖
2. **真实 LLM 集成**：连接实际的 Transformer 模型 (LLaMA, Mistral)
3. **分布式 KV 存储**：扩展到多节点部署
4. **Web 仪表板**：实时监控和可视化

### 中期 (3-6 个月)

1. **多 Nano 组合**：处理需要多个新颖推导的场景
2. **跨模态 Nano Models**：扩展到视觉-语言模型
3. **预测性创新**：预测创新需求并预创建 Nanos
4. **形式化验证**：证明 Nano Model 输出的安全属性

### 长期 (6-12 个月)

1. **联邦 Nano 学习**：跨组织协作创建 Nano
2. **Nano Model 市场**：共享和发现领域特定的 Nanos
3. **自动领域适应**：自演化的 Nano 生态系统
4. **与 AGA 集成**：完整的 AGA + Nano Model 生产系统

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Papers

This demo project is based on the paper "Nano Models: Innovation-Triggered Derivation Patches for Auxiliary-Governed Attention in Frozen Large Language Models".

## Acknowledgments

-   Claude Opus 4.5 (Anthropic) for theoretical framework refinement
-   Grok 4.0 (xAI) for simulation experiment support
