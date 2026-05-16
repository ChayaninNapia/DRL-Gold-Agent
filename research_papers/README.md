# Research Papers Library Overview

This library is a curated local collection for reinforcement learning research with a focus on deep reinforcement learning for trading. It is intended to help future agents quickly understand what is available before reading individual PDFs.

The detailed machine-readable catalog is:

`06_frameworks_and_surveys/PAPER_METADATA.csv`

That CSV is the source of truth for title, authors, year, venue, DOI/arXiv/URL, local path, broad category, trading task, asset class, RL algorithm, data frequency, input features, action space, reward type, contribution, and notes.

## Folder Map

| folder | count | purpose |
|---|---:|---|
| `01_rl_methods/` | 5 | Core RL method papers: PPO/TRPO, recurrent PPO, R2D2, policy optimization, and long-horizon PPO behavior |
| `02_rl_stability_and_training/` | 8 | Training stability references: PPO implementation details, plasticity, normalization, dormant neurons, primacy bias, and optimizer schedules |
| `03_trading_strategies/` | 19 | DRL trading strategies for stocks, futures, FX, crypto, intraday trading, signal representation, and sequence models |
| `04_portfolio_management/` | 6 | Portfolio allocation, portfolio construction, market-regime conditioning, and risk-return portfolio management |
| `05_execution_costs_and_hedging/` | 6 | Trade execution, order execution, slippage, transaction costs, compliance constraints, and derivative hedging |
| `06_frameworks_and_surveys/` | 3 | FinRL, FinRL-Meta, and metadata |

Total unique paper records in metadata: **46**.

## 01_rl_methods

| file | brief explanation |
|---|---|
| `2005.12729_engstrom_implementation_matters_deep_pg.pdf` | Shows that PPO/TRPO performance depends heavily on implementation details, not only the headline algorithm. Good for checking PPO code choices. |
| `2205.11104_recurrent_ppo_mayhems.pdf` | Explains practical recurrent PPO pitfalls: hidden-state handling, padding, masking, and sequence construction. Useful before adding LSTM/GRU policies. |
| `2401.16025_simple_policy_optimization.pdf` | Presents Simple Policy Optimization as a simplified trust-region/policy-optimization method. Useful for thinking beyond vanilla PPO. |
| `2503.01491_vc_ppo_long_horizon_decoupled_gae.pdf` | Studies PPO collapse in long-horizon reasoning tasks and proposes value-calibrated PPO. Not trading-specific, but relevant to value initialization and GAE design. |
| `r1lytjaqyx_r2d2_recurrent_experience_replay.pdf` | R2D2 paper on recurrent replay, burn-in, distributed recurrent DQN, and stale recurrent states. Useful for recurrent off-policy agents. |

## 02_rl_stability_and_training

| file | brief explanation |
|---|---|
| `1608.03983_sgdr.pdf` | Introduces stochastic gradient descent with warm restarts and cosine schedules. Useful as an optimizer/scheduler reference. |
| `2006.05990_what_matters_on_policy_rl.pdf` | Large empirical study of what actually matters in on-policy RL. Good checklist for PPO/A2C ablations. |
| `2106.01151_bjorck_towards_deeper_drl_spectral_norm.pdf` | Shows spectral normalization can stabilize deeper actor-critic networks. Relevant for robust policy/value networks. |
| `2205.07802_nikishin_primacy_bias.pdf` | Identifies primacy bias, where deep RL overfits early experience and underuses later data. Very relevant to non-stationary markets. |
| `2308.11958_kumar_regenerative_regularization_l2_init.pdf` | Proposes L2 regularization toward initial parameters to preserve plasticity in continual learning. Relevant for walk-forward and regime-shift training. |
| `2405.19153_juliani_ash_on_policy_plasticity_loss.pdf` | Studies plasticity loss specifically in on-policy deep RL and mitigation methods. High relevance for PPO/A2C retraining. |
| `iclr_blog_2022_ppo_implementation_details.html` | Practical list of PPO implementation details. Use as a code audit checklist. |
| `pmlr_v202_sokar23a_dormant_neuron_redo.pdf` | Describes dormant neurons in deep RL and ReDo, a method to recycle inactive neurons. Relevant for long or unstable training runs. |

## 03_trading_strategies

| file | brief explanation |
|---|---|
| `1807.02787_financial_trading_as_game_drqn.pdf` | Frames financial trading as an MDP/game and uses DRQN. Useful for FX-like recurrent Q-learning setups. |
| `1811.07522_practical_drl_stock_trading.pdf` | Practical stock-trading DRL paper from the FinRL lineage. Useful for environment design and multi-stock backtesting. |
| `1911.10107_volatility_scaled_futures_drl.pdf` | Applies DRL to liquid continuous futures with discrete and continuous actions plus volatility scaling. Strong reference for risk-adjusted futures/CFD-like trading. |
| `2002.11523_rl_algorithmic_trading_a3c.pdf` | Applies recurrent actor-critic/A3C to algorithmic trading. Useful as an A2C/A3C comparison reference. |
| `2004.06627_drl_algorithmic_trading_dqn.pdf` | Trading DQN paper with practical algorithmic trading evaluation. Useful as a Double DQN baseline inspiration. |
| `2017_deep_direct_rl_financial_signal_trading.pdf` | Uses recurrent deep direct RL for financial signal representation and trading. Useful for end-to-end signal learning. |
| `2019_deep_robust_rl_practical_algorithmic_trading.pdf` | Robust practical trading agent using DQN/A3C, SDAE, LSTM, position-controlled actions, and n-step reward. Useful for robust market representation design. |
| `2020_time_driven_feature_aware_drl_trading.pdf` | Jointly learns temporal feature representations and trading decisions. Useful for feature-aware OHLCV/state design. |
| `2022_sentiment_knowledge_drl_algorithmic_trading.pdf` | Combines price data with sentiment and knowledge-based features. Useful if adding news or alternative data later. |
| `2024_bilstm_attention_drl_algorithmic_trading.pdf` | Efficient deep SARSA with BiLSTM-Attention for algorithmic trading. Useful for sequence representation and attention-based features. |
| `2101.03867_encoder_decoder_stock_trading_rules.pdf` | Encoder-decoder RL framework for learning stock trading rules. Useful for interpretable/rule-like policies. |
| `2109.14789_bitcoin_transaction_strategy_drl.pdf` | PPO + LSTM for high-frequency Bitcoin transaction strategy. Useful for crypto and high-frequency PPO design. |
| `2201.09058_deepscalper_intraday_trading.pdf` | Risk-aware RL framework for fleeting intraday opportunities using LOB-style market information. Important for scalping/intraday project design. |
| `2406.08013_positional_context_intraday_trading.pdf` | Shows that positional context improves intraday DRL trading. Highly relevant for adding position, exposure, and holding-state features. |
| `2503.09655_drl_xlstm_automated_stock_trading.pdf` | Uses xLSTM networks in DRL stock trading. Useful for newer sequence-model alternatives to LSTM/GRU. |
| `2512.23773_fineft_risk_aware_ensemble_rl_futures_trading.pdf` | Risk-aware ensemble RL for futures trading. Useful for ensemble policies and futures-style evaluation. |
| `5587_13_8812_adaptive_quantitative_trading_idrl.pdf` | Adaptive quantitative trading with imitative DRL. Useful for learning from expert/imitative signals. |
| `s0020025520304692_adaptive_stock_trading_drl.pdf` | Adaptive stock trading strategies using DRL. Useful as a general stock-trading DRL reference. |
| `ssrn_5761402_maxai_intraday_index_futures_trading.pdf` | Intraday index futures system using a Q-agent plus genetic algorithm tuning. Useful for one-minute intraday futures-style setups. |

## 04_portfolio_management

| file | brief explanation |
|---|---|
| `1706.10059_drl_framework_financial_portfolio_management.pdf` | Foundational EIIE/PVM/OSBL crypto portfolio-management paper using DDPG-style continuous allocation. |
| `1808.09940_adversarial_drl_portfolio_management.pdf` | Compares DDPG, PPO, and policy-gradient methods for portfolio management under adversarial training. |
| `2020_sarl_augmented_portfolio_management.pdf` | SARL paper that augments portfolio state with predicted asset movement and optional alternative data such as news. |
| `2021_deeptrader_risk_return_portfolio_management.pdf` | DeepTrader balances risk and return using market condition embeddings and graph-style asset relationships. |
| `2509.14385_adaptive_regime_aware_rl_portfolio.pdf` | Regime-aware RL portfolio optimization with volatility, tail-risk, and regime-probability features. |
| `ssrn_3554486_alphaportfolio_caan_efma2021.pdf` | AlphaPortfolio directly optimizes portfolio construction with cross-asset attention and interpretable AI. |

## 05_execution_costs_and_hedging

| file | brief explanation |
|---|---|
| `1802.03042_deep_hedging_transaction_costs.pdf` | Deep hedging framework for derivative portfolios under transaction costs, risk limits, and market frictions. |
| `2020_ppo_optimal_trade_execution.pdf` | PPO-based end-to-end optimal trade execution using level-2 LOB data, LSTM/FCN inputs, and sparse terminal reward. |
| `2021_hrpm_hierarchical_portfolio_management.pdf` | Hierarchical portfolio management that separates high-level allocation from low-level execution to model slippage and trading costs. |
| `2103.10860_universal_trading_order_execution.pdf` | Order execution with oracle policy distillation. Useful for execution scheduling and policy distillation ideas. |
| `2103.16409_deep_hedging_derivatives_rl.pdf` | RL hedging for derivatives under transaction costs and stochastic volatility using cost mean/variance objectives. |
| `2510.04952_cmdp_ppo_trade_execution.pdf` | Constrained PPO/CMDP trade execution with compliance shielding and auditability. Treat as a recent/forward-looking reference. |

## 06_frameworks_and_surveys

| file | brief explanation |
|---|---|
| `2011.09607_finrl.pdf` | FinRL library paper. Useful for trading-environment design and baseline algorithms such as A2C, PPO, DDPG, SAC, and TD3. |
| `2112.06753_finrl_meta.pdf` | FinRL-Meta paper. Useful for near-real market environments, benchmark structure, and scalable DRL finance experimentation. |
| `PAPER_METADATA.csv` | Full catalog with structured metadata for all papers. Use this first for filtering by algorithm, asset class, trading task, reward, or input design. |

## Quick Guidance For Future Agents

For the current DRL trading project, start with these papers:

1. `03_trading_strategies/2406.08013_positional_context_intraday_trading.pdf`
2. `03_trading_strategies/1911.10107_volatility_scaled_futures_drl.pdf`
3. `03_trading_strategies/2201.09058_deepscalper_intraday_trading.pdf`
4. `03_trading_strategies/2004.06627_drl_algorithmic_trading_dqn.pdf`
5. `03_trading_strategies/2002.11523_rl_algorithmic_trading_a3c.pdf`
6. `03_trading_strategies/2019_deep_robust_rl_practical_algorithmic_trading.pdf`
7. `05_execution_costs_and_hedging/2020_ppo_optimal_trade_execution.pdf`
8. `05_execution_costs_and_hedging/2021_hrpm_hierarchical_portfolio_management.pdf`
9. `06_frameworks_and_surveys/2011.09607_finrl.pdf`

Those are the most directly useful for intraday XAUUSD / CFD-like experiments with OHLCV-derived features, positional features, transaction costs/spread, no overnight holding, PPO/A2C/Double DQN comparison, and risk-adjusted backtesting.

When selecting papers by topic:

- For PPO implementation and stability, use `01_rl_methods/` plus `02_rl_stability_and_training/`.
- For single-asset trading agents and feature engineering, use `03_trading_strategies/`.
- For allocation and multi-asset policy design, use `04_portfolio_management/`.
- For spread, slippage, execution cost, and hedging design, use `05_execution_costs_and_hedging/`.
- For reproducible environments and baselines, use `06_frameworks_and_surveys/`.
