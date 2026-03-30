Project Context: DRL-Based XAUUSD Trading System

1. Project Overview

The objective of this project is to develop an automated Algorithmic Trading system for the Gold/US Dollar pair (XAUUSD) utilizing Deep Reinforcement Learning (DRL). The system allows an AI agent to learn price dynamics and make autonomous trading decisions. The project emphasizes a Hybrid Architecture, strictly separating the offline training environment (Python) from the online execution and backtesting environment (MQL5/MetaTrader 5).

2. System Architecture

The system is divided into two primary phases:

Phase 1: Offline Training (Python Lab)

Data Preparation: Fetch historical OHLCV data (synchronized with MT5 historical data). Perform Feature Engineering (e.g., RSI, MACD, Moving Averages) and apply strict Data Normalization/Scaling methodologies.

Simulation: Build a custom trading environment using the Gymnasium library to simulate market mechanics.

Model Training: Train a DRL agent (utilizing algorithms such as PPO, SAC, or DQN) to optimize the trading policy.

Model Export: Once training is satisfactory, export the trained Actor Network into the ONNX (.onnx) format for cross-platform deployment.

Phase 2: Online Execution & Backtesting (MQL5/MT5)

Integration: Develop an Expert Advisor (EA) in MetaTrader 5 that embeds the trained .onnx model using the #resource directive.

Core Inference Logic: * Extract real-time tick/bar data and calculate technical indicators natively in MQL5.

CRITICAL: Replicate the exact mathematical Data Normalization equations used in the Python environment to ensure data consistency.

Pass the normalized state vector to the OnnxRun() function to infer the optimal Action (Buy, Sell, Hold).

Execution: Process the model's output through a Risk Management module (handling Lot sizing, Trailing Stops, etc.) before executing OrderSend() for live trading or Strategy Tester backtesting.

3. Reinforcement Learning Formulation

State Space: A normalized continuous vector representing Price Action, Technical Indicators, and the current Portfolio State (balance, open positions, unrealized PnL).

Action Space: Discrete action space (e.g., 0 = Hold, 1 = Buy, 2 = Sell).

Reward Function: A composite function primarily driven by Realized PnL, penalized by risk metrics such as Maximum Drawdown or excessive holding periods to encourage stable growth.

4. Infrastructure & Development Pipeline

Development & Debugging: Initial development utilizes a Native Environment (via venv or conda) on a GPU-enabled machine to ensure low overhead, rapid iteration, and easy debugging.

Deployment & Scaling: For long-term training and cloud deployment, the environment is containerized using Docker (configured with the NVIDIA Container Toolkit for GPU Passthrough). This guarantees dependency stability and seamless portability across different servers or cloud providers.
