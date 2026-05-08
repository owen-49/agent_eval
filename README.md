## 项目概述

这是一个专门针对复杂多跳问答任务设计的自主智能体（Autonomous Agent）。该智能体基于 **ReAct** (Reasoning and Acting) 框架，并集成了一套针对长序列推理和极端检索环境的**鲁棒性自愈协议**。
通过多轮迭代优化，本项目成功将 Agent 的推理成功率从 **34% 提升至 75%**，并在处理知识库缺失和指令漂移方面展现了出色的性能。

---

## 核心特性 (Key Features)

- **多级检索降级协议 (Multi-level Fallback Protocol)**: 当 Wikipedia 无法精准匹配条目时，系统自动切换至“碎片化语义聚合”模式，抓取搜索列表前 3 项摘要进行知识重组。
- **周期性认知重锚定 (Periodic Re-anchoring)**: 每 3 轮推理自动刷新系统指令约束，有效对抗长上下文带来的“注意力衰减”与“指令偏移”。
- **语法自愈监控器 (Syntax Self-healing)**: 实时拦截并纠正非标的工具调用格式（如 JSON 闭合错误或非法标签），确保推理链条不因格式问题中断。
- **语义感知评估 (Semantics-aware Eval)**: 升级了 F1 与 EM 评测标准，引入长实体清洗逻辑，精准衡量模型在复杂命名实体上的真实表现。

---

## 评测数据 (Evaluation)

在 **HotpotQA (200 samples)** 数据集上的性能提升曲线：

| 版本阶段 | 核心变更点 | Exact Match (EM) | F1-Score | F1-EM Gap |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | 基础 ReAct 逻辑 | 34.67% | 29.89% | -10.11% |
| v2.0 | 冗余性清洗 + 启发式引导 | 65.00% | 57.46% | -7.54% |
| **v3.0 (SOTA)** | **暴力检索增强 + 认知重锚定** | **75.00%** | **73.11%** | **-1.89%** |

---
## 模块架构（Architecture）
该项目采用模块化设计，确保了推理逻辑与工具执行的高度解耦，便于针对不同组件进行独立优化：

- core_function/agent.py: 核心决策内核。集成了指令自愈机制（Self-healing）与认知重锚定逻辑，负责管理复杂的多步推理链路，并确保长上下文环境下的指令遵循度。

- core_function/tools.py: 智能工具管理器。支持启发式搜索与暴力回退逻辑（Fallback）。当精准信息缺失时，该模块能够动态推送线索，有效破解检索死锁。

- core_function/parser.py: 高容错解析器。基于精心设计的正则表达式，负责将 LLM 生成的自然语言响应精准转换为结构化的工具调用指令，具备极强的格式鲁棒性。

- tests/evaluate.py: 多维度评测脚本。负责驱动全自动化评测流水线，并生成包含 Exact Match (EM)、F1-Score 以及详细错误诊断信息的 JSON 评测报告。
## 🛠️ 安装与运行 (Setup)

### 1. 环境准备
推荐使用 Python 3.10 环境：
```bash
conda create -n agent_eval python=3.10
conda activate agent_eval
pip install -r requirements.txt
安装后运行evaluate.py
python -m tests.evaluate --limit 200