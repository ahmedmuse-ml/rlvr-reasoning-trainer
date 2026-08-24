<div align="center">

# 🧠 RLVR Reasoning Trainer
### Production-Grade Reinforcement Learning with Verifiable Rewards (GRPO) for Large Language Models

[![CI / Test Suite](https://github.com/ahmedmuse-ml/rlvr-reasoning-trainer/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ahmedmuse-ml/rlvr-reasoning-trainer)
[![Pytest](https://img.shields.io/badge/pytest-14%20passed%20(100%25)-success?style=flat-square&logo=pytest)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Architecture](https://img.shields.io/badge/RLVR-GRPO%20%2B%20LoRA-FF6F00?style=flat-square)](src/trainer/grpo_engine.py)
[![Verifiers](https://img.shields.io/badge/Verifiers-SymPy%20%2B%20CodeSandbox-4B8BBE?style=flat-square)](src/verifiers/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<p align="center">
  <b>An end-to-end post-training infrastructure designed to train Large Language Models on complex, multi-step reasoning tasks (Symbolic Math & Program Synthesis) using Group Relative Policy Optimization (GRPO) with deterministic ground-truth verification.</b>
</p>

</div>

---

## 📌 Executive Summary

Traditional Reinforcement Learning from Human Feedback (RLHF) and Proximal Policy Optimization (PPO) rely heavily on neural Reward Models (RM), which frequently suffer from **reward overoptimization, hallucination vulnerability, and substantial GPU memory overhead** due to dedicated Value/Critic networks.

**RLVR Reasoning Trainer** implements an enterprise-grade post-training pipeline that:
1. Replaces neural reward approximations with **Deterministic Multi-Domain Verifiers** (Symbolic Math and Isolated Sandbox Execution).
2. Adopts **Group Relative Policy Optimization (GRPO)**, normalizing sample advantages directly across group rollouts ($N=8$) to eliminate the Value Network entirely.
3. Implements **Strict Format Regularization & Length-Penalties** to mitigate verbosity-based reward hacking.

---

## 🏗️ System Architecture

```text
                                  ┌──────────────────────────────────────────────┐
                                  │           Multi-Domain Data Pool             │
                                  │   [GSM8K Math (80%) + MBPP Code (20%)]       │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │          Batched Policy Sampling             │
                                  │   Generates N completions per prompt         │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                         ┌───────────────────────────────┴───────────────────────────────┐
                         ▼                                                               ▼
        ┌──────────────────────────────────┐                            ┌──────────────────────────────────┐
        │       Mathematical Domain        │                            │      Code Synthesis Domain       │
        └────────────────┬─────────────────┘                            └────────────────┬─────────────────┘
                         │                                                               │
                         ▼                                                               ▼
        ┌──────────────────────────────────┐                            ┌──────────────────────────────────┐
        │      LaTeX Stack Parser          │                            │      Subprocess Sandbox          │
        │  • Recursive Bracket Matcher     │                            │  • Zero-Network Subprocess / DinD│
        │  • SymPy Equivalence Engine      │                            │  • Memory & PID Capped Execution │
        └────────────────┬─────────────────┘                            └────────────────┬─────────────────┘
                         │                                                               │
                         └───────────────────────────────┬───────────────────────────────┘
                                                         │
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │           Multi-Objective Reward             │
                                  │  R = w₁·R_accuracy + w₂·R_format - w₃·Length │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │        Group Advantage Normalization         │
                                  │       A_i = (R_i - mean(R)) / std(R)         │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │       Policy Update via LoRA (PEFT)          │
                                  │     Backpropagates Loss with KL Penalty      │
                                  └──────────────────────────────────────────────┘