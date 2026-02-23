# KENNY — AI Options Day Trading Platform
## Comprehensive Master Architecture & Development Plan

> **Document Version**: 1.0
> **Created**: February 23, 2026
> **Last Updated**: February 23, 2026
> **Status**: Fresh build. Reusing YELENA's proven components (models, data, Pine Script) with clean architecture and proper wiring from day one.
> **Philosophy**: Build step-by-step, test everything, move forward only when it works. NEVER skip steps. NEVER shorten documents. ALWAYS clarify with user before proceeding.
> **RULE**: This document should only GROW over time. Never shorten, condense, or remove sections.

---

# TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Platform Vision & Goals](#2-platform-vision--goals)
3. [Guiding Philosophy](#3-guiding-philosophy)
4. [Architecture Overview — The 5-Layer Cognitive System](#4-architecture-overview--the-5-layer-cognitive-system)
5. [Technology Stack](#5-technology-stack)
6. [Layer 1: Data Perception](#6-layer-1-data-perception)
7. [Layer 2: Feature Engineering & Market Understanding](#7-layer-2-feature-engineering--market-understanding)
8. [Layer 3: AI Prediction — The Ensemble Brain](#8-layer-3-ai-prediction--the-ensemble-brain)
9. [Layer 4: Action Planning — The Strategic Planner](#9-layer-4-action-planning--the-strategic-planner)
10. [Layer 5: Communication — The Interface](#10-layer-5-communication--the-interface)
11. [Pine Script Indicator Suite](#11-pine-script-indicator-suite)
12. [TradingView-ML Integration Architecture](#12-tradingview-ml-integration-architecture)
13. [Database Architecture](#13-database-architecture)
14. [AWS Infrastructure Design](#14-aws-infrastructure-design)
15. [Real-Time Data Strategy — Schwab + Polygon](#15-real-time-data-strategy--schwab--polygon)
16. [ML Pipeline — Colab Pro + SageMaker MLOps](#16-ml-pipeline--colab-pro--sagemaker-mlops)
17. [Risk Management System](#17-risk-management-system)
18. [Development Phases & Roadmap](#18-development-phases--roadmap)
19. [Cost Analysis](#19-cost-analysis)
20. [Testing Strategy](#20-testing-strategy)
21. [Security & Secrets Management](#21-security--secrets-management)
22. [Monitoring & Observability](#22-monitoring--observability)
23. [Future Roadmap](#23-future-roadmap)
24. [Development Rules & Alignment Principles](#24-development-rules--alignment-principles)
25. [Lessons Learned from YELENA](#25-lessons-learned-from-yelena)
26. [Appendices](#26-appendices)

---

# 1. EXECUTIVE SUMMARY

## 1.1 What is KENNY?

**KENNY** is an AI-powered options day trading platform built on a 5-layer cognitive architecture. It is the clean-slate successor to YELENA, reusing all proven components (44 trained ML models, 2.46M historical bars, 10 Pine Script indicators, Feature Engine v2) while completely rebuilding the application layer with proper wiring from day one.

KENNY is NOT a simple signal-forwarding service. It is a **predictive intelligence system** that:
- **Predicts** high-probability trade setups through multi-model AI confluence
- **Provides** complete trade theses (entry, SL, TP, confidence score, reasoning)
- **Validates** signals through TradingView indicator confluence synergy
- **Learns** from every trade outcome via reinforcement learning
- **Empowers** manual trade execution with institutional-grade AI-driven insights

## 1.2 Why KENNY? (Lessons from YELENA)

YELENA built everything but failed at the wiring layer:
- Dashboard → backend: CloudFront served static files, all `/api/*` calls returned 404
- No WebSocket infrastructure existed despite being designed
- TV-ML integration code was built but never deployed
- Direct EC2 port exposure replaced the proper API Gateway/Lambda architecture, breaking connectivity
- `main.py` imported old modules, orchestrator never called TV synergy, files were missing from server

KENNY fixes this by:
1. **Proper connectivity layer**: API Gateway (REST + WebSocket) handles all routing — dashboard NEVER talks directly to EC2
2. **Clean codebase**: No legacy v1 files, no disconnected modules, no wrong imports
3. **Wired from day one**: Every module connected, tested, and verified before moving to the next
4. **Proven components reused**: 44 ML models, 2.46M bars, Pine Script indicators, Feature Engine — all carry forward unchanged

## 1.3 What Carries Forward from YELENA (Same AWS Account)

| Asset | Location | Status |
|-------|----------|--------|
| **44 ML Models** (53.9MB) | S3: `s3://yelena-models/models_v2/` | ✅ Proven — 15min: 87.8% WR / 14.46 PF |
| **2.46M Historical Bars** | RDS: `yelena-db` (1min/5min/15min/1hr/daily) | ✅ Clean, validated data |
| **1.2M Feature Rows** | RDS: `features` table (163 features, v2) | ✅ Backfilled across 4 TFs × 11 symbols |
| **1.75M Training Data Rows** | S3: `s3://yelena-data-lake/training-data/` | ✅ Labeled with hybrid ATR+R:R |
| **Pine Script Indicators** (10) | TradingView + YELENA repo code | ✅ All optimized with real backtested params |
| **Colab Notebooks** (7+) | Google Drive: `/MyDrive/yelena/` | ✅ Full training pipeline |
| **SSM Secrets** (6 params) | AWS SSM Parameter Store: `/yelena/*` | ✅ All API keys stored |
| **VPC/Networking** | AWS VPC, subnets, security groups, IGW | ✅ Properly configured |
| **EC2 Instance** | `i-0c9a8339948b61b12` (t3.large, 52.71.0.160) | ✅ Running, will nuke ~/yelena/ and clone kenny |
| **RDS Database** | `yelena-db.ci3ykeq8mckh.us-east-1.rds.amazonaws.com` | ✅ PostgreSQL 15, db.t3.small |

## 1.4 Target Trading Profile

| Parameter | Value |
|-----------|-------|
| **Style** | Scalping (0DTE, weeklies), 2-15 minute holds |
| **Assets** | SPY, QQQ, TSLA, NVDA, META, AAPL, GOOGL, MSFT, AMZN, AMD, NFLX (11 symbols) |
| **Timeframes (v1)** | 1min, 5min, 15min, 1hr |
| **Timeframes (v1.1)** | Add 2min timeframe + SPX |
| **Risk Tolerance** | Aggressive |
| **Max Position Size** | 10% of account per trade |
| **Daily Loss Limit** | $500 |
| **Max Concurrent Positions** | 3 |
| **Trade Execution** | Manual (v1) — AI provides complete trade thesis, user executes |

## 1.5 Target Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Win Rate | ≥65% | Rolling 30 days |
| Profit Factor | ≥1.5 | Rolling 30 days |
| Max Drawdown | ≤5% | Per session |
| Average R:R | ≥1.2 | Per trade |
| Signal Latency | <1 second | End-to-end (Schwab tick → Verdict on dashboard) |
| Model Prediction Accuracy | ≥70% | Backtested |
| Sharpe Ratio | >1.0 | Backtested |

## 1.6 Proven ML Performance (From YELENA Training)

**Simple Average Ensemble @ 0.55 threshold (v2, 163 features)**:

| Timeframe | CALL WR | CALL PF | PUT WR | PUT PF |
|-----------|---------|---------|--------|--------|
| 1min | 68.2% | 4.29 | 67.8% | 4.21 |
| 5min | 74.5% | 5.82 | 74.3% | 5.79 |
| **15min** ⭐ | **87.8%** | **14.46** | **86.3%** | **12.63** |
| 1hr | 80.1% | 8.04 | 82.5% | 9.43 |

**Unanimous 3/3 Agreement** adds +4-7 percentage points WR across all timeframes.

---

# 2. PLATFORM VISION & GOALS

## 2.1 KENNY v1.0 — Minimum Viable Trading Platform

The goal: a working system Talon can trade with daily. This means:

1. **Real-time market data** flowing via Schwab WebSocket ($0/mo)
2. **Feature Engine** computing 163 features per bar in real-time
3. **ML Ensemble** generating predictions across all active timeframes (1/5/15/60min)
4. **Verdict Engine** producing GO/SKIP decisions with entry/SL/TP
5. **TradingView synergy** modulating ML confidence via Master Confluence webhooks
6. **Cross-timeframe agreement** scoring across all timeframes and all models
7. **React Dashboard** showing live verdicts with real-time WebSocket updates (proper HTTPS via API Gateway)
8. **Everything wired and working** — no disconnected modules, no 404s, no missing files

## 2.2 KENNY v1.1 — Expanded Coverage

- Add 2-minute timeframe (train new models on Colab)
- Add SPX symbol
- Options Contract Selector module (finds optimal contract for each signal)
- SageMaker Pipelines for automated weekly retraining

## 2.3 KENNY v2.0 — Autonomous Trading

- Schwab API auto-execution (system places trades automatically)
- NLP sentiment analysis from news feeds
- Portfolio optimization (Kelly criterion position sizing)
- Mobile push notifications
- Historical pattern matching ("this setup looks like...")

---

# 3. GUIDING PHILOSOPHY

## 3.1 From Reactive Analysis to Predictive Action

KENNY doesn't just identify patterns — it calculates the **probability** of those patterns leading to profitable moves and formulates the **optimal, risk-managed trade plan** to capitalize on them.

The system combines two complementary analytical structures:
- **ML Ensemble**: Identifies statistical patterns in 163 engineered features across multiple timeframes
- **TradingView Indicators**: Analyzes chart structure through 9 specialized technical analysis tools

Their agreement amplifies confidence. Their divergence triggers caution. Neither operates alone.

## 3.2 The Signal Confluence Model

A KENNY verdict is not based on a single model or indicator. It requires:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KENNY SIGNAL CONFLUENCE                           │
│                                                                     │
│  ML ENSEMBLE (Primary)                                              │
│  ├── XGBoost prediction (direction + confidence)                    │
│  ├── Transformer prediction (direction + confidence)                │
│  ├── CNN prediction (direction + confidence)                        │
│  ├── RL Agent (exit optimization)                                   │
│  └── Ensemble Agreement (3/3 unanimous = strongest)                 │
│                                                                     │
│  CROSS-TIMEFRAME AGREEMENT                                          │
│  ├── 1min direction + confidence                                    │
│  ├── 5min direction + confidence                                    │
│  ├── 15min direction + confidence                                   │
│  ├── 1hr direction + confidence                                     │
│  └── Alignment Score (how many TFs agree)                           │
│                                                                     │
│  TRADINGVIEW SYNERGY (Confidence Modulator)                         │
│  ├── Master Confluence score (-10 to +10)                           │
│  ├── Grade (A+/A/B+/B)                                              │
│  ├── Component quality signals (squeeze, BOS, volume spike, etc.)   │
│  └── TV-ML Agreement Classification (7 levels)                      │
│                                                                     │
│  ═══════════════════════════════════════════════                     │
│  FINAL VERDICT: GO / SKIP / EXIT                                    │
│  With: Entry, Stop Loss, Take Profit (3 tiers), Confidence %,       │
│        Reasoning (which models agree, which indicators fire)         │
└─────────────────────────────────────────────────────────────────────┘
```

## 3.3 Development Philosophy

1. **Step-by-step, test everything**: Complete and verify each step before moving forward
2. **Prove before integrate**: Each component proven individually before connecting to others
3. **Wiring is architecture**: The connections between components ARE the system — test them first
4. **Real data from day one**: No placeholders, no mocks in production paths
5. **Documents only grow**: Never shorten, condense, or remove — only add

---

# 4. ARCHITECTURE OVERVIEW — THE 5-LAYER COGNITIVE SYSTEM

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LAYER 5: COMMUNICATION                               │
│  React Dashboard (CloudFront CDN)                                       │
│  ↕ API Gateway REST (HTTPS) ↕ API Gateway WebSocket (WSS)              │
│  TradingView Pine Script Indicators │ Audio/Push Notifications          │
├─────────────────────────────────────────────────────────────────────────┤
│                    LAYER 4: ACTION PLANNING                             │
│  Trade Verdict Engine │ Cross-TF Agreement │ TV-ML Synergy Engine       │
│  Dynamic Risk Engine (ATR-based SL/TP) │ Position Sizing                │
├─────────────────────────────────────────────────────────────────────────┤
│                    LAYER 3: AI PREDICTION                               │
│  XGBoost │ Transformer │ CNN │ RL Agent (PPO)                           │
│  Simple Average Ensemble │ Unanimous Agreement Filter                   │
│  Colab Pro (Training) → EC2 In-Process (Inference) → SageMaker (MLOps) │
├─────────────────────────────────────────────────────────────────────────┤
│                    LAYER 2: FEATURE ENGINEERING                          │
│  163 Features: MA(22) + Momentum(18) + Volatility(12) + Volume(14)     │
│  + PriceAction(20) + Time(8) + SMC(22) + VWAP(8) + Divergence(4)      │
│  + Derived(12) + MTF(24)                                                │
│  Batch Engine (historical) + Streaming Engine (real-time)               │
├─────────────────────────────────────────────────────────────────────────┤
│                    LAYER 1: DATA PERCEPTION                             │
│  Schwab WebSocket (Real-Time, $0) │ Polygon REST (Historical, $29/mo)  │
│  TradingView Webhooks (Master Confluence alerts)                        │
│  EC2 Orchestrator → Bar Aggregation → PostgreSQL (RDS)                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Architecture Difference from YELENA

**YELENA** (broken): Dashboard → direct to EC2:8000 → 404 errors, no HTTPS, no WebSocket management

**KENNY** (fixed):
```
Dashboard (CloudFront) ──→ API Gateway REST (HTTPS) ──→ Lambda/EC2 Backend
                        ──→ API Gateway WebSocket (WSS) ──→ EC2 Orchestrator broadcast
TradingView ──→ API Gateway REST (webhook endpoint) ──→ Lambda TV Alert Handler
```

The dashboard NEVER talks directly to EC2. API Gateway handles all routing, HTTPS termination, and WebSocket connection management. This is the connectivity layer that YELENA was missing.

---

# 5. TECHNOLOGY STACK

## 5.1 Backend

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.12 | Core backend |
| API Framework | FastAPI | REST + WebSocket APIs on EC2 |
| Database | PostgreSQL 15 (RDS, native partitioning) | Time-series data storage |
| Data Processing | Pandas + NumPy + TA-Lib | Feature computation |
| Real-Time Data | Schwab WebSocket (schwab-py) | Live market data ($0/mo) |
| Historical Data | Polygon.io REST ($29/mo) | Backfill + training data |
| Task Scheduling | EventBridge | Market-hours EC2 start/stop |

## 5.2 Machine Learning

| Component | Technology | Purpose |
|-----------|------------|---------|
| Training | Google Colab Pro ($12/mo) | GPU training, experimentation |
| Inference | EC2 in-process (FastAPI ModelManager) | All 44 models in RAM, 10-50ms |
| MLOps | SageMaker (Registry + Pipelines) | Versioning, automated retraining |
| XGBoost | XGBoost library | Tabular feature synthesis |
| Transformer | PyTorch (custom YelenaTransformer) | Time-series sequence prediction |
| CNN | PyTorch (multi-scale 1D + SE attention) | Local pattern recognition |
| RL Agent | Custom PPO | Trade exit optimization (92% smart exits) |

### Why EC2 In-Process Inference (Not SageMaker Endpoints)

This decision was made during YELENA development after detailed evaluation of 5 options:

| Criteria | EC2 In-Process | SageMaker Real-Time | SageMaker Serverless |
|----------|---------------|--------------------|--------------------|
| **Latency per call** | 2-8ms | 20-50ms | 50-200ms (warm) |
| **Cold start** | 0 (models in RAM) | 0 (always-on) | 1-6 seconds |
| **Total for 6-8 calls** | 10-50ms | 120-400ms | 300-1600ms |
| **Monthly cost** | $12 (market hours) | $56-184 | $5-20 |

**Key insight**: SageMaker endpoints add 5-15ms overhead PER API CALL (HTTP serialization, auth, routing). With 6-8 model calls per trading decision = 40-120ms unnecessary latency. Models are only 54MB total — they fit trivially in EC2 RAM.

**Where SageMaker IS Used**: Model Registry ($0.50/mo) for versioning/rollback, Pipelines ($6-8/mo in Phase 2) for automated weekly retraining. NOT for inference.

## 5.3 Connectivity Layer (THE FIX)

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Gateway REST** | AWS API Gateway | HTTPS routing for all `/api/*` calls |
| **API Gateway WebSocket** | AWS API Gateway | WSS connection management for dashboard |
| **Lambda Functions** | AWS Lambda (Python 3.12) | TV webhook handler, WebSocket handlers, lightweight API |
| **CloudFront** | AWS CloudFront | CDN for React dashboard static files |

This is the layer that YELENA was missing. The dashboard talks to API Gateway (managed HTTPS, proper routing), NOT directly to EC2 ports.

## 5.4 Frontend

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | React 19 | UI framework |
| Build Tool | Vite | Fast development |
| Styling | Tailwind CSS 3.3.3 | Utility-first CSS |
| Charts | TradingView Lightweight Charts | Price charts in dashboard |
| Analytics | Recharts | Performance visualization |
| State | Zustand | Lightweight state management |
| Real-Time | WebSocket via API Gateway WSS | Live signal delivery |

## 5.5 External Services

| Service | Purpose | Cost |
|---------|---------|------|
| Polygon.io Stocks Starter | Historical backfill + training data | $29/month |
| Schwab API | Real-time WebSocket data + future execution | $0/month |
| TradingView Premium | Pine Script indicators, chart analysis | Already paid |
| Google Colab Pro | ML training with GPU | $12/month |
| GitHub | Version control | Free |

**Dropped**: Polygon.io Options Starter ($79/mo) — Schwab handles live options chains. Will add back when building Options Contract Selector ML model.

---

# 6. LAYER 1: DATA PERCEPTION

## 6.1 Real-Time Data: Schwab WebSocket ($0/mo)

**Primary real-time data source for KENNY**, replacing Polygon's delayed feed.

| Attribute | Value |
|-----------|-------|
| Library | schwab-py (`pip install schwab-py`) |
| Data Type | Real-time consolidated trades + quotes |
| Latency | Real-time (no 15-minute delay) |
| Cost | $0/month |
| Requirement | Schwab account with $500+ equity (✅ met) |
| Auth | OAuth 2.0 with refresh token (7-day expiry) |
| Status | Credentials ready, OAuth flow completed, tokens obtained |

**Schwab WebSocket → KENNY Pipeline**:
```
Schwab WebSocket (real-time ticks)
    │
    ▼
EC2 Orchestrator
    │
    ├── Bar Aggregation (1min/5min/15min/1hr candles)
    │
    ├── Feature Engine (163 features per bar)
    │
    ├── ML Ensemble (4 models × N timeframes)
    │
    ├── TV Confluence Synergy (if fresh TV signal exists)
    │
    ├── Cross-Timeframe Agreement
    │
    ├── Verdict Engine (GO/SKIP with entry/SL/TP)
    │
    ▼
API Gateway WebSocket → Dashboard (live update)
```

**7-Day Token Refresh**: Schwab refresh tokens expire after 7 days. KENNY must either:
- Auto-refresh before expiry (background task)
- Send notification to re-authenticate if auto-refresh fails
- Fallback to Polygon if Schwab connection drops (degraded mode with 15-min delay)

## 6.2 Historical Data: Polygon.io REST ($29/mo)

**Used exclusively for**: Historical backfill, training data generation, model retraining.
**NOT used for**: Real-time trading signals (that's Schwab).

| Data Type | Method | Coverage |
|-----------|--------|----------|
| 1-minute bars | REST API | ~6 months (1.07M rows) |
| 5-minute bars | REST API | ~2 years (961K rows) |
| 15-minute bars | REST API | ~2 years (336K rows) |
| 1-hour bars | REST API | ~2 years (86K rows) |
| Daily bars | REST API | ~2 years (5.4K rows) |
| **Total** | | **2.46M bars in RDS** |

## 6.3 TradingView Webhooks

Master Confluence fires webhook alerts to API Gateway when CALL/PUT signals trigger. Rich payload includes 9 component scores, direction, grade, confidence, entry/SL/TP.

```
TradingView Alert fires
    │
    ▼
API Gateway REST endpoint (/api/tv-alert)
    │
    ▼
Lambda: kenny-tv-webhook-handler
    │
    ├── Validates passphrase
    ├── Parses rich payload (9 components)
    ├── Stores in TV Signals table (RDS) + in-memory cache (DynamoDB/ElastiCache)
    │
    ▼
EC2 Orchestrator reads TV signal → TVConfluenceIntegrator → modulates ML confidence
```

## 6.4 Data Quality Controls

- Deduplication via UNIQUE constraints on (symbol, timestamp)
- Gap detection and backfill from Polygon REST
- Timezone normalization (all UTC, display in ET)
- Weekend bar filtering (6,159 known weekend bars from extended hours)
- Schwab connection health monitoring with fallback

---

# 7. LAYER 2: FEATURE ENGINEERING & MARKET UNDERSTANDING

## 7.1 Feature Engine v2 — 163 Features (Proven from YELENA)

The Feature Engine is fully built and proven. Two versions exist for different purposes:

| Engine | File | Lines | Purpose |
|--------|------|-------|---------|
| **Batch** | `services/feature_engine.py` | 1,362 | Historical backfill, training data generation |
| **Streaming** | `ml/feature_engine.py` | 530 | Real-time computation from 200-bar OHLCV window |

**CRITICAL**: These are different files serving different purposes. NEVER confuse or overwrite one with the other. (This caused an incident during YELENA development.)

## 7.2 Feature Categories (163 Total)

| Category | Count | Examples |
|----------|-------|---------|
| Moving Averages | 22 | SMA/EMA 5-200, VWMA, price vs MA ratios, slopes, ribbon count |
| Momentum | 18 | RSI, MACD, Stochastic, StochRSI, CCI, Williams %R, ROC, MFI, ADX |
| Volatility | 12 | ATR, Bollinger Bands, Keltner Channels, squeeze detection |
| Volume | 14 | Volume ratios, OBV, A/D Line, CMF, VPT, volume delta, force index |
| Price Action | 20 | Candle patterns, consecutive bars, HH/LL, gaps, engulfing, doji |
| Time-Based | 8 | Hour/minute, market open/close, power hour, day of week |
| **Smart Money Concepts** | **22** | **BOS, CHoCH, FVG, Order Blocks, displacement, premium/discount, liquidity sweeps** |
| **VWAP & Levels** | **8** | **Session VWAP (daily reset), bands, distance, position** |
| **Divergence** | **4** | **RSI/MACD bullish/bearish divergence** |
| **Additional Derived** | **12** | **Multi-period returns, MACD hist slope, EMA crosses, interaction features** |
| **Multi-Timeframe** | **24** | **Per HTF: trend, RSI, ADX, squeeze, MACD hist, volume ratio, candle dir + alignment scores** |

## 7.3 Per-Timeframe Feature Counts

Each timeframe has different feature counts after dropping self-referencing HTF features:

| Timeframe | Features | Reason |
|-----------|----------|--------|
| 1min | 163 | Full feature set (3 HTFs available: 5min, 15min, 1hr) |
| 5min | 156 | Drop htf_5min_* (self-reference) |
| 15min | 149 | Drop htf_5min_* and htf_15min_* |
| 1hr | 139 | Drop all htf_* (top-level TF, no HTFs) |

---

# 8. LAYER 3: AI PREDICTION — THE ENSEMBLE BRAIN

## 8.1 The Four Models

### 8.1.1 XGBoost — The Synthesist
- **Input**: Full 163-feature vector per bar
- **Output**: P(CALL), P(PUT), P(HOLD)
- **Strength**: Handles tabular features natively, fast inference
- **v2 15min**: CALL 75.2% WR / 6.07 PF, PUT 74.1% / 5.71 PF

### 8.1.2 Transformer — The Forecaster (BEST Individual Model)
- **Input**: 30-bar sequence of features
- **Architecture**: Custom YelenaTransformer (d_model=128, 4 heads, 3 layers, d_ff=256, dropout=0.15)
- **Strength**: Captures temporal patterns and long-range dependencies
- **v2 15min**: CALL 78.3% WR / 7.20 PF, PUT 77.5% / 6.89 PF

### 8.1.3 CNN — The Visionary
- **Input**: 30-bar sequence (same as Transformer)
- **Architecture**: Multi-scale 1D CNN (kernels 3/5/7) with SE attention, dual-conv branches
- **Strength**: Detects local patterns that tree/sequence models miss — essential ensemble diversity
- **v2 15min**: CALL 62.4% WR / 3.32 PF, PUT 61.8% / 3.22 PF

### 8.1.4 RL Agent (PPO) — The Exit Master
- **Input**: Feature vector + current position state
- **Architecture**: Custom PPOActorCritic (NOT Stable-Baselines3) with shared backbone
- **Strength**: Learns optimal trade EXITS (92-94% of exits are agent-decided, not static SL/TP)
- **v2 15min**: 65.1% WR, 1.52 PF, 94% smart exits

### 8.2 Ensemble Strategy

**Simple Average** of XGBoost + Transformer + CNN predictions, with RL Agent handling exit decisions separately. Equal weighting was chosen because it outperformed all other methods (weighted, majority vote, stacked meta-learner) — diversity beats individual strength.

**44 total model files** (53.9MB) organized as:
```
models/
├── 5min/  (11 files: 3×XGB + 3×Transformer + 3×CNN + 1×RL + scalers + configs)
├── 15min/ (11 files)
├── 1hr/   (11 files)
└── 1min/  (11 files)
```

### 8.3 Confluence Requirements for "GO" Verdict

For a trade to receive GO status, the system requires:
1. **ML Agreement**: Minimum 2/3 prediction models agree on direction (3/3 unanimous = highest quality)
2. **Confidence Threshold**: Combined ensemble confidence > 0.55 (operating sweet spot)
3. **Cross-TF Alignment**: At least 2 timeframes agree on direction
4. **Risk Parameters Met**: ATR-based SL/TP within acceptable R:R ratio (minimum 1.2:1)
5. **TV Synergy (when available)**: TV agreement boosts confidence up to +15%, divergence penalizes up to -20%

---

# 9. LAYER 4: ACTION PLANNING — THE STRATEGIC PLANNER

## 9.1 Trade Verdict Engine

The Verdict Engine consolidates all signals into a final GO/SKIP/EXIT decision:

1. **Collects** ML predictions from all active timeframes
2. **Scores** cross-timeframe agreement
3. **Integrates** TradingView confluence signals (when fresh — 10-min TTL)
4. **Calculates** ATR-based entry/SL/TP levels
5. **Sizes** position based on confidence and risk rules
6. **Generates** complete verdict with reasoning

### Verdict Output

```json
{
  "verdict": "GO",
  "direction": "CALL",
  "symbol": "SPY",
  "confidence": 0.82,
  "entry": 450.25,
  "stop_loss": 449.10,
  "take_profit_1": 451.40,
  "take_profit_2": 452.55,
  "take_profit_3": 453.70,
  "position_size": 2,
  "risk_reward": "2.1:1",
  "reasoning": {
    "ml_agreement": "3/3 unanimous CALL",
    "timeframe_alignment": "4/4 TFs agree bullish",
    "tv_synergy": "strong_agree (A grade, score +7.5)",
    "quality_signals": ["QBands squeeze firing", "QSMC BOS bullish", "QCVD institutional buying"]
  }
}
```

## 9.2 Dynamic Risk Engine

| Rule | Calculation |
|------|-------------|
| Stop Loss | Entry - (ATR_14 × 1.5) for longs |
| Take Profit 1 | Entry + (Risk × 2.0) — conservative 2:1 R:R |
| Take Profit 2 | AI predicted target from ML models |
| Take Profit 3 | Next resistance level from S/R analysis |
| Position Size | Max 10% of account, respects $500 daily loss limit |

---

# 10. LAYER 5: COMMUNICATION — THE INTERFACE

## 10.1 React Dashboard (via CloudFront + API Gateway)

### Connectivity Architecture (THE FIX)

```
React Dashboard (CloudFront CDN — kenny-dashboard bucket)
    │
    ├── REST API calls: https://api.kenny.{domain}/api/*
    │   └── API Gateway REST → Lambda/EC2 Backend
    │
    └── WebSocket: wss://ws.kenny.{domain}
        └── API Gateway WebSocket → EC2 Orchestrator broadcasts
```

**No direct EC2 access from dashboard.** Everything goes through API Gateway.

### Core Dashboard Features (v1)

| Feature | Description |
|---------|-------------|
| **Verdict Cards** | GO/SKIP with direction, confidence, entry/SL/TP, reasoning |
| **Real-Time Updates** | WebSocket push — no polling, instant signal delivery |
| **Signal History** | Full log of all verdicts with outcomes |
| **System Health** | Connection status (Schwab, ML, TV), model load status |
| **Multi-TF View** | See verdicts across all timeframes simultaneously |

### Future Dashboard Features (v1.1+)

| Feature | Description |
|---------|-------------|
| TradingView Lightweight Charts | Candlestick charts from backend data |
| Analytics Panel | Win rate, profit factor, equity curve, drawdown |
| Audio Alerts | Distinct sounds for GO verdicts |
| Desktop Notifications | Push notifications for high-conviction signals |
| Settings Panel | Ticker filters, grade filters, risk parameters |

---

# 11. PINE SCRIPT INDICATOR SUITE

## 11.1 The 10 Indicators (All Complete from YELENA)

| # | Indicator | Purpose | Lines | Status |
|---|-----------|---------|-------|--------|
| 1 | **QCloud** | Trend via multi-layer MA cloud | ~350 | ✅ Optimized |
| 2 | **QLine** | SuperTrend adaptive trend ribbon | ~350 | ✅ Optimized |
| 3 | **QWave** | ADX-based directional momentum | ~370 | ✅ Optimized |
| 4 | **QBands** | Adaptive volatility bands + squeeze | ~300 | ✅ Optimized |
| 5 | **Moneyball** | Momentum oscillator with flip signals | ~250 | ✅ Optimized |
| 6 | **QMomentum** | RSI divergence detection | ~300 | ✅ Optimized |
| 7 | **QCVD** | Cumulative volume delta | ~280 | ✅ Optimized |
| 8 | **QSMC** | Smart money concepts (BOS/CHoCH/FVG/OB) | ~350 | ✅ Optimized |
| 9 | **QGrid** | Pullback entry zones | ~300 | ✅ Optimized |
| 10 | **Master Confluence** ⭐ | Aggregator of all 9 indicators | 613 | ⚠️ Needs source wiring fix |

## 11.2 Critical Issue: Master Confluence Source Wiring

**Problem**: 0/22 input.source() connections are wired. All sources default to SPY close price (~$687) instead of indicator values (-1 to +1, 0-7, etc.). This causes PUT-only signals because raw price scores far outside expected ranges.

**Fix**: Manually wire each of 22 input.source() fields in TradingView indicator settings to correct sub-indicator X_ hidden plots. This is a manual TradingView UI task.

### 22-Source Wiring Map

| # | Field | Connect To | Expected Range |
|---|-------|-----------|---------------|
| 1 | QCloud Bull Count | QCloud → X_BullCount | 0-7 |
| 2 | QCloud Direction | QCloud → X_Direction | -1/0/+1 |
| 3 | QCloud Squeeze | QCloud → X_Squeeze | 0/1 |
| 4 | QLine Trend | QLine → X_Trend | -1/0/+1 |
| 5 | QLine Bounce Count | QLine → X_BounceCount | 0-10 |
| 6 | QWave Score | QWave → X_Score | -100 to +100 |
| 7 | QWave TrendStrength | QWave → X_TrendStrength | 0-100 |
| 8 | QBands Position | QBands → X_Position | -1 to +1 |
| 9 | QBands Squeeze | QBands → X_Squeeze | 0/1 |
| 10 | Moneyball Score | Moneyball → X_Score | -100 to +100 |
| 11 | Moneyball Trend | Moneyball → X_Trend | -1/0/+1 |
| 12 | QMomentum Momentum | QMomentum → X_Momentum | -100 to +100 |
| 13 | QMomentum Trend | QMomentum → X_Trend | -1/0/+1 |
| 14 | QCVD Volume Ratio | QCVD → X_VolumeRatio | 0+ |
| 15 | QCVD Delta | QCVD → X_Delta | -1/0/+1 |
| 16 | QSMC Structure | QSMC → X_Structure | -1/0/+1 |
| 17 | QSMC BOS/CHoCH | QSMC → X_BOS_CHoCH | 0/1/2 |
| 18 | QGrid Position | QGrid → X_GridPosition | 0-1 |
| 19 | QGrid Support Count | QGrid → X_SupportCount | 0-5 |
| 20 | QGrid Resistance Count | QGrid → X_ResistanceCount | 0-5 |
| 21 | (Reserved) | — | — |
| 22 | (Reserved) | — | — |

## 11.3 v2 Enhancement Pattern (Applied to All)

Every indicator includes:
1. Auto-optimized settings per symbol/timeframe (hardcoded lookup tables with real backtested values)
2. Enhanced analytics (dynamics, slopes, durations, quality scores)
3. Info table on chart
4. Enhanced webhook payload
5. Hidden plot exports for Master Confluence
6. Education guide (300+ lines)
7. alertcondition() entries
8. Timeframe groups: Scalp (≤1min), Intraday (5-15min), Swing (1hr+)

## 11.4 Key Pine Script Rules

- **Pine Script v6**: No `transp` (use `color.new()`), no `when`, booleans can't be `na`, `ta.*` at global scope
- **QLine MUST use built-in `ta.supertrend()`** — NEVER manual implementation (critical ratcheting bug)
- **Multi-line expressions**: Ternary chains and string concatenations MUST be on single lines (Pine v6 parser limitation)
- **All lookup tables contain REAL optimized values** from backtesting (not placeholders)

---

# 12. TRADINGVIEW-ML INTEGRATION ARCHITECTURE

## 12.1 Integration Philosophy

ML and TradingView are **complementary analytical structures**, not competing systems:
- **ML identifies statistical patterns** in 163 engineered features
- **TV analyzes chart structure** through 9 specialized technical tools
- Their agreement/divergence MODULATES confidence — TV never overrides ML direction

## 12.2 Synergy Logic (Tier 1 — Confidence Modulation)

| TV-ML Relationship | Classification | Confidence Adjustment |
|--------------------|---------------|----------------------|
| Both CALL, high TV grade | strong_agree | +15% |
| Both CALL, medium TV grade | agree | +10% |
| Both CALL, low TV grade | weak_agree | +5% |
| TV neutral | neutral | 0% |
| Different directions, low TV | weak_diverge | -5% |
| Different directions, medium TV | diverge | -10% |
| Different directions, high TV | strong_diverge | -20% |

**Grade-aware weighting**: A+ (×1.0), A (×0.8), B+ (×0.5), B (×0.3)

**Freshness decay**: TV signals have 10-minute TTL. At >5 minutes, influence halved. At >10 minutes, ignored.

## 12.3 Component Quality Signal Extraction

These are qualitative signals extracted from TV sub-indicator states:
- **QCloud/QBands squeeze = true** → Volatility expansion imminent
- **QSMC BOS/CHoCH detected** → Structural break (major move likely)
- **QCVD volume spike** → Institutional activity
- **QLine bounce count ≥ 3** → Strong S/R confirmation

## 12.4 Data Flow

```
TradingView Master Confluence → Webhook Alert (rich JSON payload)
    │
    ▼
API Gateway REST (/api/tv-alert) → Lambda: kenny-tv-webhook-handler
    │
    ├── Validates passphrase
    ├── Parses 9-component breakdown
    ├── Stores in tv_signals table (RDS) + in-memory cache
    │
    ▼
EC2 Orchestrator polls TV signal cache
    │
    ▼
TVConfluenceIntegrator.compute_adjustment()
    │
    ├── Grade-aware confidence delta
    ├── Agreement classification (7 levels)
    ├── Component quality signal extraction
    │
    ▼
Verdict Engine applies TV adjustment to ML confidence → Final Verdict
```

---

# 13. DATABASE ARCHITECTURE

## 13.1 PostgreSQL 15 with Native Partitioning (Existing RDS)

- **Instance**: `yelena-db.ci3ykeq8mckh.us-east-1.rds.amazonaws.com`
- **Class**: db.t3.small (2 vCPU, 2GB RAM)
- **Database name**: `yelena` (keeping as-is — name doesn't affect functionality)
- **TimescaleDB**: NOT available on standard RDS — using native PostgreSQL partitioning

### Existing Tables (Data Preserved)

| Table | Rows | Purpose |
|-------|------|---------|
| bars_1min | 1,074,985 | 1-minute OHLCV bars |
| bars_5min | 961,829 | 5-minute OHLCV bars |
| bars_15min | 335,824 | 15-minute OHLCV bars |
| bars_1hr | 85,879 | 1-hour OHLCV bars |
| bars_daily | 5,445 | Daily OHLCV bars |
| features | 1,204,164 | 163 v2 features per bar |
| symbols | 12 | Tracked symbols |
| **Total** | **~3.67M rows** | |

### New Tables for KENNY

```sql
-- TV signals from Master Confluence webhooks
CREATE TABLE tv_signals (
    id              SERIAL PRIMARY KEY,
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol          VARCHAR(10) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    direction       VARCHAR(10) NOT NULL,
    score           DOUBLE PRECISION,
    grade           VARCHAR(5),
    confidence      DOUBLE PRECISION,
    entry_price     DOUBLE PRECISION,
    stop_loss       DOUBLE PRECISION,
    tp1             DOUBLE PRECISION,
    tp2             DOUBLE PRECISION,
    tp3             DOUBLE PRECISION,
    components      JSONB,
    raw_signal      JSONB,
    is_rich         BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Verdicts (GO/SKIP/EXIT decisions)
CREATE TABLE verdicts (
    id              SERIAL PRIMARY KEY,
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol          VARCHAR(10) NOT NULL,
    verdict         VARCHAR(10) NOT NULL,
    direction       VARCHAR(10),
    confidence      DOUBLE PRECISION,
    entry_price     DOUBLE PRECISION,
    stop_loss       DOUBLE PRECISION,
    tp1             DOUBLE PRECISION,
    tp2             DOUBLE PRECISION,
    tp3             DOUBLE PRECISION,
    ml_agreement    VARCHAR(20),
    tf_alignment    VARCHAR(20),
    tv_agreement    VARCHAR(20),
    tv_score        DOUBLE PRECISION,
    tv_grade        VARCHAR(5),
    tv_confidence_delta DOUBLE PRECISION,
    reasoning       JSONB,
    quality_signals JSONB,
    metadata        JSONB
);

-- Predictions from ML models
CREATE TABLE predictions (
    time            TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(10) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    model_name      VARCHAR(50) NOT NULL,
    signal          VARCHAR(10) NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    probabilities   JSONB NOT NULL,
    model_version   VARCHAR(20)
);

-- Trade plans and tracking
CREATE TABLE trade_plans (
    id              SERIAL PRIMARY KEY,
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol          VARCHAR(10) NOT NULL,
    direction       VARCHAR(10) NOT NULL,
    grade           VARCHAR(5) NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    plan_json       JSONB NOT NULL,
    status          VARCHAR(20) DEFAULT 'PENDING',
    outcome         VARCHAR(10),
    actual_pnl      DOUBLE PRECISION,
    notes           TEXT
);

-- Actual trades executed
CREATE TABLE trades (
    id              SERIAL PRIMARY KEY,
    trade_plan_id   INTEGER REFERENCES trade_plans(id),
    time_entered    TIMESTAMPTZ NOT NULL,
    time_exited     TIMESTAMPTZ,
    symbol          VARCHAR(10) NOT NULL,
    direction       VARCHAR(10) NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION,
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    quantity        INTEGER NOT NULL,
    pnl             DOUBLE PRECISION,
    pnl_percent     DOUBLE PRECISION,
    outcome         VARCHAR(10),
    exit_reason     VARCHAR(20),
    notes           TEXT
);
```

---

# 14. AWS INFRASTRUCTURE DESIGN

## 14.1 Network Architecture

```
AWS Account (us-east-1) — EXISTING, shared with YELENA data
│
├── VPC: yelena-vpc (10.0.0.0/16) — EXISTING
│   ├── Public Subnet (10.0.1.0/24) — EC2
│   ├── Private Subnet 1 (10.0.2.0/24) — RDS
│   └── Private Subnet 2 (10.0.3.0/24) — RDS (multi-AZ requirement)
│
├── EC2: i-0c9a8339948b61b12 (t3.large, 52.71.0.160) — EXISTING
│   ├── ~/kenny/ (NEW clean codebase)
│   ├── FastAPI backend (port 8000)
│   ├── ML ModelManager (44 models in RAM)
│   ├── Orchestrator (Schwab WebSocket → pipeline)
│   └── Feature Engine (streaming, real-time)
│
├── RDS: yelena-db (PostgreSQL 15, db.t3.small) — EXISTING
│   └── Database: yelena (preserving name + all data)
│
├── S3 Buckets
│   ├── yelena-data-lake — EXISTING (raw data, training CSVs)
│   ├── yelena-models — EXISTING (44 v2 model files)
│   ├── yelena-backups — EXISTING
│   └── kenny-dashboard — NEW (React static files)
│
├── API Gateway — NEW
│   ├── kenny-rest-api (REST, HTTPS)
│   │   ├── /api/signals/* → EC2 backend
│   │   ├── /api/verdicts/* → EC2 backend
│   │   ├── /api/tv-alert → Lambda: kenny-tv-webhook
│   │   ├── /api/health → EC2 backend
│   │   └── /api/symbols → EC2 backend
│   └── kenny-websocket-api (WebSocket, WSS)
│       ├── $connect → Lambda: kenny-ws-connect
│       ├── $disconnect → Lambda: kenny-ws-disconnect
│       └── broadcast ← EC2 orchestrator pushes via API
│
├── Lambda Functions — NEW
│   ├── kenny-tv-webhook — Receives TV alerts, validates, stores
│   ├── kenny-ws-connect — WebSocket connection handler
│   ├── kenny-ws-disconnect — WebSocket disconnection handler
│   └── kenny-ws-broadcaster — Broadcasts verdicts to connected clients
│
├── CloudFront — NEW distribution
│   └── kenny-dashboard bucket → https://d{xxx}.cloudfront.net
│
├── SageMaker (Phase 1.5+)
│   ├── Model Registry — Version control, rollback ($0.50/mo)
│   ├── Experiments — Training run tracking (free)
│   └── Pipelines (Phase 2) — Automated weekly retraining ($6-8/mo)
│
├── SSM Parameter Store — EXISTING
│   ├── /yelena/polygon/api-key
│   ├── /yelena/schwab/api-key
│   ├── /yelena/schwab/api-secret
│   ├── /yelena/schwab/callback-url
│   ├── /yelena/webhook-passphrase
│   └── /yelena/database-url
│
├── EventBridge — NEW
│   ├── Rule: Start EC2 at 9:15 AM ET (Mon-Fri)
│   └── Rule: Stop EC2 at 4:15 PM ET (Mon-Fri)
│
└── CloudWatch
    ├── Log Groups: /kenny/*
    └── Alarms: API errors, prediction latency, EC2 health
```

## 14.2 What's NEW vs EXISTING

| Resource | Status | Action |
|----------|--------|--------|
| VPC, subnets, security groups | EXISTING | Keep as-is |
| EC2 instance | EXISTING | Nuke ~/yelena/, clone kenny repo |
| RDS database | EXISTING | Add new tables, keep all data |
| S3 data-lake + models + backups | EXISTING | Keep as-is |
| S3 kenny-dashboard | **NEW** | Create bucket for React static files |
| API Gateway REST | **NEW** | Create REST API with routes |
| API Gateway WebSocket | **NEW** | Create WebSocket API |
| Lambda functions (4) | **NEW** | TV webhook, WS connect/disconnect/broadcast |
| CloudFront distribution | **NEW** | CDN for kenny-dashboard |
| EventBridge rules | **NEW** | Market-hours EC2 scheduling |
| SSM parameters | EXISTING | Keep as-is (still /yelena/* paths) |
| IAM roles | EXISTING | May need Lambda execution role |

---

# 15. REAL-TIME DATA STRATEGY — SCHWAB + POLYGON

## 15.1 Architecture

```
SCHWAB (Real-Time, $0/mo)                    POLYGON ($29/mo, Historical)
├── WebSocket: live ticks/quotes              ├── REST: backfill historical bars
├── Real-time options chains                  ├── REST: training data (5+ years)
├── Account data (positions, buying power)    └── REST: gap detection + fill
└── Future: order execution
    │                                              │
    ▼                                              ▼
EC2 Orchestrator                              Feature Engine (batch mode)
├── Bar aggregation (1/5/15/60min)            ├── Historical feature computation
├── Feature Engine (streaming)                └── Training data generation
├── ML Ensemble predictions
├── Verdict Engine
└── Dashboard broadcast
```

## 15.2 Schwab API Details

| Attribute | Value |
|-----------|-------|
| Library | schwab-py (`pip install schwab-py`) |
| Auth | OAuth 2.0 (app key + secret from SSM) |
| Token Refresh | 7-day expiry — auto-refresh task needed |
| Data | Real-time consolidated WebSocket |
| Equity Requirement | $500 minimum (✅ exceeded) |
| Cost | $0/month |
| Status | Credentials ready, OAuth completed, tokens obtained |

## 15.3 Schwab Token Management

```python
# Token auto-refresh strategy
# schwab-py handles refresh internally, but tokens expire after 7 days of inactivity
# KENNY needs:
# 1. Token storage (encrypted in SSM or local file)
# 2. Health check endpoint that verifies Schwab connection
# 3. Notification if re-auth required (push notification or dashboard warning)
# 4. Fallback: Polygon REST polling (degraded mode, 15-min delayed)
```

## 15.4 Cost Savings vs YELENA

| Data Source | YELENA Cost | KENNY Cost | Savings |
|-------------|-------------|------------|---------|
| Polygon Stocks | $29/mo | $29/mo | $0 |
| Polygon Options | $79/mo | $0 (dropped) | $79/mo |
| Schwab Real-Time | N/A | $0 | — |
| **Total Data** | **$108/mo** | **$29/mo** | **$79/mo ($948/yr)** |

---

# 16. ML PIPELINE — COLAB PRO + SAGEMAKER MLOPS

## 16.1 Development Workflow

```
Google Colab Pro ($12/mo)           AWS SageMaker (MLOps)           EC2 FastAPI (Inference)
┌─────────────────────┐         ┌─────────────────────────┐     ┌─────────────────────────┐
│ • Jupyter notebooks  │         │ • Model Registry         │     │ • All 44 models in RAM   │
│ • GPU training       │ export  │   (version control)      │     │ • 10-50ms latency        │
│ • Experimentation    │ models  │ • Experiments tracking   │ pull│ • $12/mo (market hours)  │
│ • Hyperparameter     │───────> │   (hyperparams, metrics) │────>│ • ModelManager class     │
│   tuning (Optuna)    │ to S3   │ • Pipelines (Phase 2)   │     │ • Hot-swap via /reload   │
│ • Walk-forward       │         │   (automated retraining) │     │ • Cross-TF agreement     │
│   validation         │         │ • $0.50-8/mo             │     │                          │
└─────────────────────┘         └─────────────────────────┘     └─────────────────────────┘
```

## 16.2 Model Lifecycle

**Phase 1 (Current)**: Train on Colab → export to Google Drive → upload to S3 → pull to EC2 → hot-swap via /reload

**Phase 1.5**: + SageMaker Model Registry for versioning, rollback, approval workflows

**Phase 2**: + SageMaker Pipelines for automated weekly retraining (pull new data → features → train → evaluate → register → deploy)

## 16.3 Training Data Pipeline

```
PostgreSQL (bars + features tables)
    ↓
generate_training_data.py (pulls from Feature Engine v2 — NO inline compute)
    ↓
Hybrid ATR+R:R labeling (SL=1.5×ATR, TP=2.0×ATR)
    ↓
S3 (1.75M rows, 2.1GB across 4 timeframes)
    ↓
Colab Pro (GPU training, Optuna tuning, walk-forward validation)
    ↓
44 model files → S3 → EC2
```

---

# 17. RISK MANAGEMENT SYSTEM

## 17.1 Automated Rules

| Rule | Value | Enforcement |
|------|-------|-------------|
| Max position size | 10% of account | Calculated by Risk Engine |
| Daily loss limit | $500 | Tracked in DB, enforced before signal generation |
| Max concurrent positions | 3 | Tracked in DB |
| Minimum R:R | 1.2:1 | Required for GO verdict |
| Stop-loss required | Every trade | ATR-based calculation |
| Maximum hold time | 15 minutes (scalps) | Monitored, alert if exceeded |

## 17.2 Dynamic SL/TP

```
LONG: SL = Entry - (ATR_14 × 1.5)
SHORT: SL = Entry + (ATR_14 × 1.5)

TP1 = Entry + (Risk × 2.0)     -- 2:1 R:R conservative
TP2 = AI predicted target        -- ML model price prediction
TP3 = Next resistance/support    -- from S/R analysis
```

---

# 18. DEVELOPMENT PHASES & ROADMAP

## Phase 0: KENNY Setup (Day 1)

| Step | Task | Details |
|------|------|---------|
| 0.1 | Create GitHub repo `kenny` | Fresh repo, clean structure |
| 0.2 | Define project structure | Proper module layout from day one |
| 0.3 | Nuke ~/yelena/ on EC2 | Remove old codebase |
| 0.4 | Clone kenny repo to EC2 | `git clone` to ~/kenny/ |
| 0.5 | Create kenny-dashboard S3 bucket | For React static files |
| 0.6 | Copy Pine Script code to repo | From YELENA repo |
| 0.7 | Copy proven Python modules | Feature Engine (batch + streaming), ML model classes |
| 0.8 | Verify ML models accessible | S3 models still accessible from EC2 |
| 0.9 | Verify RDS data accessible | All 2.46M bars + features still queryable |
| 0.10 | Create new RDS tables | tv_signals, verdicts, predictions, trade_plans, trades |

**GATE**: EC2 has clean kenny repo, RDS has new tables + old data, S3 models accessible, Pine Script code in repo.

## Phase 1: Connectivity Layer (Days 2-3) — THE FIX

This phase builds what YELENA was missing.

| Step | Task | Details |
|------|------|---------|
| 1.1 | Create API Gateway REST API | kenny-rest-api with proper routes |
| 1.2 | Create Lambda: kenny-tv-webhook | TV alert handler with passphrase validation |
| 1.3 | Create API Gateway WebSocket API | kenny-websocket-api for dashboard real-time |
| 1.4 | Create Lambda: kenny-ws-connect | WebSocket connection handler |
| 1.5 | Create Lambda: kenny-ws-disconnect | WebSocket disconnection handler |
| 1.6 | Create Lambda: kenny-ws-broadcaster | Push verdicts to connected clients |
| 1.7 | Configure API Gateway → EC2 integration | REST routes that proxy to EC2:8000 |
| 1.8 | Create CloudFront distribution | kenny-dashboard bucket as origin |
| 1.9 | Test: REST API → EC2 health check | Verify HTTPS routing works |
| 1.10 | Test: WebSocket connect/disconnect | Verify WSS connections managed properly |
| 1.11 | Test: TV webhook → Lambda → RDS | Verify alert pipeline end-to-end |
| 1.12 | Create EventBridge rules | EC2 start/stop on market hours |

**GATE**: API Gateway REST returns data from EC2 backend via HTTPS. WebSocket connections established via WSS. TV webhook received and stored. CloudFront serves static content. This is the foundation everything else builds on.

## Phase 2: Backend Core (Days 3-5)

| Step | Task | Details |
|------|------|---------|
| 2.1 | FastAPI application (main.py) | Clean app with proper router mounting from day one |
| 2.2 | Config module | SSM-backed configuration (database URL, API keys) |
| 2.3 | Database module | asyncpg connection pool |
| 2.4 | Health endpoint | /health returning system status |
| 2.5 | Symbols endpoint | /api/symbols returning tracked symbols |
| 2.6 | Signals router | /api/signals — storage + retrieval |
| 2.7 | Verdicts router | /api/verdicts — storage + retrieval + stats |
| 2.8 | TV Alerts v2 router | /api/tv-alert — rich payload parsing, TVAlertStore |
| 2.9 | WebSocket manager | Broadcast verdicts to connected dashboard clients |
| 2.10 | Verify all routers mounted | main.py imports and mounts every router correctly |

**GATE**: All API endpoints respond correctly. TV alerts store to DB. WebSocket broadcasts work. No disconnected modules.

## Phase 3: ML Inference Pipeline (Days 5-7)

| Step | Task | Details |
|------|------|---------|
| 3.1 | ModelManager | Load all 44 models from S3/local, predict(), batch_predict() |
| 3.2 | Prediction Service | /predict, /batch-predict, /health on port 8001 (or integrated) |
| 3.3 | Streaming Feature Engine | Real-time 163-feature computation from 200-bar window |
| 3.4 | Verify model loading | All 44 models load correctly with proper architecture classes |
| 3.5 | Test single prediction | Feature vector → ModelManager → correct output format |
| 3.6 | Test ensemble | All 3 prediction models + RL agent producing results |
| 3.7 | Test cross-TF agreement | Multi-timeframe scoring working correctly |

**GATE**: All models loaded, predictions accurate, ensemble producing expected confidence ranges.

## Phase 4: Orchestrator + Verdict Engine (Days 7-9)

| Step | Task | Details |
|------|------|---------|
| 4.1 | Schwab WebSocket client | Connect to Schwab, receive real-time ticks |
| 4.2 | Bar aggregation | Aggregate ticks into 1/5/15/60min OHLCV bars |
| 4.3 | Orchestrator pipeline | Bar close → Features → ML → TV Synergy → Verdict |
| 4.4 | TV Confluence Integrator | Read TV signals, compute confidence adjustment |
| 4.5 | Verdict Engine | GO/SKIP/EXIT with full reasoning |
| 4.6 | Wire TV store to Verdict Engine | set_tv_store() called during initialization |
| 4.7 | WebSocket broadcast | Verdicts pushed to dashboard via API Gateway WebSocket |
| 4.8 | Systemd services | kenny-api.service + kenny-orchestrator.service (auto-start) |
| 4.9 | Test live data flow | Schwab → bars → features → ML → verdict → dashboard |

**GATE**: Complete end-to-end pipeline working with real Schwab data. Verdicts appearing on dashboard in real-time.

## Phase 5: Dashboard (Days 9-11)

| Step | Task | Details |
|------|------|---------|
| 5.1 | React app scaffold | Vite + React 19 + Tailwind 3.3.3 |
| 5.2 | WebSocket connection | Connect to API Gateway WSS endpoint |
| 5.3 | Verdict cards | Display GO/SKIP with direction, confidence, entry/SL/TP |
| 5.4 | System health indicators | Schwab connection, ML model status, last verdict time |
| 5.5 | Multi-TF view | Show verdicts across all timeframes simultaneously |
| 5.6 | Signal history | Log of recent verdicts |
| 5.7 | Build and deploy | npm run build → S3 → CloudFront → verify live |
| 5.8 | Test: Dashboard receives live verdicts | WebSocket push works through API Gateway |

**GATE**: Dashboard live on CloudFront, receiving and displaying real-time verdicts via WSS. No 404s, no connection issues.

## Phase 6: TradingView Integration (Days 11-13)

| Step | Task | Details |
|------|------|---------|
| 6.1 | Fix Master Confluence source wiring | Manual: wire all 22 input.source() to sub-indicator X_ plots |
| 6.2 | Verify MC generating correct signals | Both CALL and PUT signals, not just PUT |
| 6.3 | Configure webhook alerts | MC alert → API Gateway → Lambda → RDS |
| 6.4 | Test TV-ML synergy live | TV signal + ML prediction → confidence modulation → verdict |
| 6.5 | Master Confluence debug indicator | Deploy debug Pine to verify source values |

**GATE**: Master Confluence generating correct CALL/PUT signals. Webhooks flowing to backend. TV-ML synergy working and visible in verdicts.

## Phase 7: Paper Trading Validation (Days 13-20+)

| Step | Task | Details |
|------|------|---------|
| 7.1 | Full system integration test | All components connected, stable |
| 7.2 | Paper trading (1 week minimum) | Execute signals manually on Schwab paper |
| 7.3 | Daily trade logging | Record every verdict, every trade, every outcome |
| 7.4 | Performance tracking | Win rate, profit factor, R:R, drawdown |
| 7.5 | Bug fixes and tuning | Fix issues as they appear |
| 7.6 | Go/No-go decision | Meet minimum thresholds → live trading |

**GATE**: Win rate ≥65%, profit factor ≥1.5, system stable, latency <1 second. Ready for live trading.

---

# 19. COST ANALYSIS

## 19.1 KENNY v1 Monthly Costs

| Service | Cost | Notes |
|---------|------|-------|
| EC2 (t3.large) | $12/mo | Market-hours only via EventBridge (9:15 AM - 4:15 PM ET) |
| RDS (db.t3.small) | $30/mo | Existing, always-on |
| S3 | $5/mo | Data lake + models + dashboard |
| API Gateway | $0-3/mo | Free tier covers most usage |
| Lambda | $0/mo | Free tier |
| CloudFront | $0/mo | Free tier |
| CloudWatch | $0/mo | Free tier |
| SageMaker (Phase 1.5) | $0.50/mo | Model Registry only |
| Polygon Stocks | $29/mo | Historical backfill |
| Colab Pro | $12/mo | ML training |
| **KENNY v1 TOTAL** | **~$89-91/mo** | |

## 19.2 Cost Comparison: KENNY vs YELENA

| Category | YELENA | KENNY | Savings |
|----------|--------|-------|---------|
| Data (Polygon + Schwab) | $108/mo | $29/mo | $79/mo |
| EC2 (with scheduling) | $12/mo | $12/mo | $0 |
| RDS | $30/mo | $30/mo | $0 |
| S3 | $5/mo | $5/mo | $0 |
| API Gateway + Lambda | $0 | $0-3/mo | -$3/mo |
| CloudFront | $0 | $0 | $0 |
| SageMaker | $0 | $0.50/mo | -$0.50/mo |
| Colab Pro | $12/mo | $12/mo | $0 |
| **TOTAL** | **$167/mo** | **~$91/mo** | **~$76/mo ($912/yr)** |

---

# 20. TESTING STRATEGY

## 20.1 Testing Philosophy

**"We do NOT move forward if something doesn't work."**

Every phase has a GATE that must pass before proceeding. No exceptions.

## 20.2 Critical Tests

| Test | What It Validates | When |
|------|-------------------|------|
| API Gateway → EC2 roundtrip | Connectivity layer works (HTTPS) | Phase 1 |
| WebSocket connect + receive | Real-time push works (WSS) | Phase 1 |
| TV webhook → Lambda → RDS | Alert pipeline end-to-end | Phase 1 |
| Model loading (all 44) | ML inference ready | Phase 3 |
| Schwab → bar → feature → ML → verdict | Complete data pipeline | Phase 4 |
| Dashboard receives live verdict | Full user experience | Phase 5 |
| MC generates CALL + PUT signals | TV wiring fixed | Phase 6 |

---

# 21. SECURITY & SECRETS MANAGEMENT

## 21.1 Secrets (All in AWS SSM Parameter Store)

| Secret | SSM Path | Type |
|--------|----------|------|
| Polygon API Key | /yelena/polygon/api-key | SecureString |
| Schwab API Key | /yelena/schwab/api-key | SecureString |
| Schwab API Secret | /yelena/schwab/api-secret | SecureString |
| Schwab Callback URL | /yelena/schwab/callback-url | String |
| Webhook Passphrase | /yelena/webhook-passphrase | SecureString |
| Database URL | /yelena/database-url | SecureString |

**NEVER** store secrets in .env files, Git, or hardcoded in source code.

## 21.2 Network Security

- RDS not publicly accessible (EC2 only via security group)
- EC2 SSH from known IPs only
- API Gateway handles HTTPS termination
- Webhook passphrase validation on TV alerts
- CORS restricted to CloudFront domain

---

# 22. MONITORING & OBSERVABILITY

| Metric | Threshold | Action |
|--------|-----------|--------|
| EC2 CPU | >80% for 5 min | Email alert |
| RDS CPU | >80% for 5 min | Email alert |
| API Gateway 5xx | >5 in 5 min | Email alert |
| Schwab WebSocket disconnect | Any | Attempt reconnect, fallback to Polygon |
| Prediction latency | >100ms | Email alert |
| Schwab token expiry | <24 hours | Dashboard warning + email |

---

# 23. FUTURE ROADMAP

### v1.1 (Month 2-3)
- Add 2-minute timeframe (train new models)
- Add SPX symbol
- Options Contract Selector module
- SageMaker Pipelines for automated weekly retraining
- Re-add Polygon Options ($79/mo) for options ML training

### v2.0 (Month 3-6)
- Schwab API auto-execution
- Advanced analytics dashboard (equity curves, drawdown analysis)
- Audio alerts + desktop notifications
- TradingView Lightweight Charts in dashboard

### v3.0 (Month 6+)
- NLP sentiment analysis (FinBERT)
- Portfolio optimization (Kelly criterion)
- Mobile push notifications
- Multi-user support (Cognito)
- Historical pattern matching

---

# 24. DEVELOPMENT RULES & ALIGNMENT PRINCIPLES

## 24.1 Cardinal Rules

1. **NEVER skip steps** — Complete and verify each step before moving on
2. **NEVER make assumptions** — Always clarify with user when uncertain
3. **NEVER shorten documents** — These files only GROW. Never condense or remove.
4. **ALWAYS provide complete code** — No "..." or partial snippets
5. **ALWAYS test before marking complete** — "Code written" ≠ "Complete"
6. **Step by step, one at a time** — Complete and verify each step
7. **Wiring IS architecture** — Test connections before building more components
8. **Dashboard never talks directly to EC2** — Everything through API Gateway

## 24.2 Key Technical Rules

- **Two feature_engine.py files** — batch (1362 lines) and streaming (530 lines). NEVER confuse them.
- **QLine uses built-in `ta.supertrend()`** — NEVER manual implementation
- **Pine Script v6** — No `transp`, no `when`, booleans can't be `na`, `ta.*` at global scope
- **Feature counts vary per timeframe** — 1min: 163, 5min: 156, 15min: 149, 1hr: 139
- **TA-Lib naming MUST match between training and inference** — exact same function calls and names
- **TV-ML synergy is ADDITIVE** — ML is primary, TV modulates confidence, never overrides direction
- **Simple Average ensemble** — Equal weighting beats all other methods for diverse architectures

---

# 25. LESSONS LEARNED FROM YELENA

These are the specific failures and discoveries that shaped KENNY's architecture:

1. **Direct EC2 port exposure broke everything** — Dashboard couldn't reach backend because nothing routed `/api/*` calls. KENNY uses API Gateway for ALL routing.
2. **Code was built but never deployed** — tv_alerts_v2.py (574 lines), tv_confluence.py (473 lines) existed as files but were never copied to the server, never imported in main.py, never wired to the orchestrator.
3. **main.py imported the wrong modules** — Line 13 imported old tv_alerts instead of tv_alerts_v2. KENNY builds with correct imports from day one.
4. **Orchestrator never called set_tv_store()** — TV synergy code existed in the verdict engine but was never connected.
5. **0/22 TradingView sources were wired** — Master Confluence read SPY close price (~$687) for every input, producing garbage PUT-only signals.
6. **Feature Engine was accidentally overwritten** — Streaming version (530 lines) replaced by batch version (1362 lines). KENNY keeps them in clearly separate paths with clear naming.
7. **Architecture decisions made without proper analysis** — Removing API Gateway/Lambda to "simplify" actually broke connectivity. KENNY restores the proper serverless connectivity layer.
8. **SageMaker endpoints add latency, not reduce it** — 5-15ms per call × 6-8 calls = 40-120ms overhead for 54MB of models that fit trivially in RAM.
9. **Polygon $29 plan = 15-minute delayed data** — Cannot live trade with delayed data. Schwab provides real-time for free.
10. **Documents were shortened** — PROJECT_STATE.md went from 747 to 330 lines in one session, losing critical context. KENNY enforces "documents only grow."

---

# 26. APPENDICES

## Appendix A: KENNY Project File Structure

```
~/kenny/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI app — ALL routers properly mounted
│   │   ├── config.py             # SSM-backed configuration
│   │   ├── database.py           # asyncpg connection pool
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── signals.py        # /api/signals — storage + retrieval
│   │   │   ├── verdicts.py       # /api/verdicts — storage + retrieval + stats
│   │   │   └── tv_alerts.py      # /api/tv-alert — TV webhook handler (v2 from day one)
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── model_manager.py       # ModelManager — loads 44 models in-process
│   │   │   ├── feature_engine.py      # Streaming FeatureEngine (530 lines, real-time)
│   │   │   ├── orchestrator.py        # Schwab WebSocket → full pipeline
│   │   │   ├── verdict_engine.py      # GO/SKIP/EXIT with TV synergy WIRED
│   │   │   └── tv_confluence.py       # TVConfluenceIntegrator (from day one, not added later)
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── polygon_client.py      # REST client for historical backfill
│   │   │   ├── schwab_client.py       # NEW: Schwab WebSocket + REST client
│   │   │   └── feature_engine.py      # Batch FeatureEngine (1362 lines, historical)
│   │   └── websocket/
│   │       └── manager.py             # WebSocket broadcasting
│   ├── models/                        # 44 ML model files (synced from S3)
│   │   ├── 1min/  (11 files)
│   │   ├── 5min/  (11 files)
│   │   ├── 15min/ (11 files)
│   │   └── 1hr/   (11 files)
│   ├── scripts/
│   │   ├── backfill_historical.py
│   │   ├── backfill_features.py
│   │   ├── backfill_predictions.py
│   │   ├── generate_training_data.py
│   │   └── optimize_params_v2.py
│   ├── lambda/                        # Lambda function code
│   │   ├── tv_webhook/
│   │   │   └── handler.py
│   │   ├── ws_connect/
│   │   │   └── handler.py
│   │   ├── ws_disconnect/
│   │   │   └── handler.py
│   │   └── ws_broadcaster/
│   │       └── handler.py
│   ├── tests/
│   │   └── test_prediction_service.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── index.css
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── Sidebar.jsx
│   │   │   └── MainContent.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   └── stores/
│   │       └── appStore.js
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js
├── tradingview/
│   ├── indicators/
│   │   ├── qcloud.pine
│   │   ├── qline.pine
│   │   ├── qwave.pine
│   │   ├── qbands.pine
│   │   ├── moneyball.pine
│   │   ├── qmomentum.pine
│   │   ├── qcvd.pine
│   │   ├── qsmc.pine
│   │   ├── qgrid.pine
│   │   ├── master_confluence.pine
│   │   └── master_confluence_debug.pine
│   └── education/
│       ├── qcloud-guide.md
│       ├── qline-guide.md
│       └── ... (9 more guides)
├── infrastructure/
│   ├── sql/
│   │   └── schema.sql
│   ├── cloudformation/            # NEW: IaC for API Gateway, Lambda, CloudFront
│   │   └── kenny-stack.yaml
│   └── scripts/
│       └── deploy.sh
├── docs/
│   ├── KENNY_MASTER_PLAN.md
│   └── KENNY_PROJECT_STATE.md
├── .gitignore
└── README.md
```

## Appendix B: Key Commands

```bash
# SSH to EC2
ssh -i yelena-key.pem ubuntu@52.71.0.160

# Connect to RDS from EC2
psql -h yelena-db.ci3ykeq8mckh.us-east-1.rds.amazonaws.com -U postgres -d yelena

# Get RDS credentials from SSM
DB_URL=$(aws ssm get-parameter --name "/yelena/database-url" --with-decryption --query "Parameter.Value" --output text)

# Activate Python venv
cd ~/kenny/backend && source ../venv/bin/activate

# Run backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Sync models from S3
aws s3 sync s3://yelena-models/models_v2/ ~/kenny/backend/models/

# Deploy dashboard
cd ~/kenny/frontend && npm run build
aws s3 sync dist/ s3://kenny-dashboard/ --delete
aws cloudfront create-invalidation --distribution-id <ID> --paths "/*"

# Systemd services
sudo systemctl start kenny-api
sudo systemctl status kenny-api
journalctl -u kenny-api -f
```

## Appendix C: Symbols Tracked (v1)

| Symbol | Type | Notes |
|--------|------|-------|
| SPY | ETF | S&P 500 Index ETF |
| QQQ | ETF | Nasdaq 100 ETF |
| TSLA | Stock | High volatility |
| NVDA | Stock | High volatility |
| META | Stock | Medium volatility |
| AAPL | Stock | Medium volatility |
| GOOGL | Stock | Medium volatility |
| MSFT | Stock | Medium volatility |
| AMZN | Stock | Medium volatility |
| AMD | Stock | High volatility |
| NFLX | Stock | Medium volatility |

**v1.1 additions**: SPX (S&P 500 Index), 2min timeframe

---

*End of KENNY Master Plan v1.0 — Created February 23, 2026*
