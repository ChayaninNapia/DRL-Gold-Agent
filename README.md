# Project Context: DRL-Based XAUUSD Trading System

## 1. Project Overview

The objective of this project is to develop an automated Algorithmic Trading system for the Gold/US Dollar pair (XAUUSD) utilizing Deep Reinforcement Learning (DRL). The system allows an AI agent to learn price dynamics and make autonomous trading decisions. The project emphasizes a Hybrid Architecture, strictly separating the offline training environment (Python) from the online execution and backtesting environment (MQL5/MetaTrader 5).

## 2. System Architecture

The architecture is divided into Model Development (Phase 1) and Deployment (Phase 2).

### Phase 1: Model Development (Offline)
- **Phase 1A: Data Preparation**: Fetch raw historical data (CSV/DB) from MT5. Perform preprocessing, feature engineering (cleaning, normalizing, indicators), and create windowed state representations (e.g., PyTorch Tensors).
- **Phase 1B: Reinforcement Learning**: Train the DRL agent (e.g., REINFORCE, TD, Actor-Critic) using a Custom Gym Environment featuring a step-by-step market simulator and a predefined reward function (Profit - Drawdown Penalty).
- **Phase 1C: Evaluation & Export**: Conduct out-of-sample backtesting on unseen data to validate the agent and output the best-performing model.

### Phase 2: Deployment Architecture (Online)
The system supports two parallel deployment options:
- **Way 1: Standalone EA (ONNX)**: Export the final trained model into an ONNX format (`.onnx`), compile it natively into an MQL5 Expert Advisor (EA), and execute directly on an MT5 terminal (e.g., via VPS).
- **Way 2: Python Integration**: Load model weights into a continuous Python main script. Utilize the `MetaTrader5` library to communicate directly with an MT5 terminal running in the background for real-time OHLCV data retrieval and order execution.

## 3. Reinforcement Learning Formulation

- **State Space**: A normalized continuous vector representing Price Action, Technical Indicators, and the current Portfolio State (balance, open positions, unrealized PnL).

- **Action Space**: Discrete action space (e.g., `0 = Hold`, `1 = Buy`, `2 = Sell`).

- **Reward Function**: A composite function primarily driven by Realized PnL, penalized by risk metrics such as Maximum Drawdown or excessive holding periods to encourage stable growth.

## 4. Infrastructure & Development Pipeline

- **Development & Debugging**: Initial development utilizes a Native Environment (via venv or conda) on a GPU-enabled machine to ensure low overhead, rapid iteration, and easy debugging.

- **Deployment & Scaling**: For long-term training and cloud deployment, the environment is containerized using Docker (configured with the NVIDIA Container Toolkit for GPU Passthrough). This guarantees dependency stability and seamless portability across different servers or cloud providers.
