# KENNY — AI Options Day Trading Platform
## Project State Document

> **CRITICAL**: This document maintains project continuity across chat sessions.
> Update after every work session. When starting a new chat, upload this file
> along with KENNY_MASTER_PLAN.md.
>
> **RULE**: This document should only GROW over time. Never shorten, condense, or remove sections.
> Add new information; do not replace or summarize existing information.

---

## Quick Status Dashboard

| Component | Status | Last Updated |
|-----------|--------|-------------|
| **GitHub Repo** | ⬜ Not Created — `kenny` repo | Feb 23, 2026 |
| **AWS Infrastructure (inherited)** | ✅ Running (EC2, RDS, S3, SSM, VPC) | Feb 23, 2026 |
| **EC2 Instance** | ✅ Running (t3.large, 52.71.0.160) — needs ~/yelena/ cleanup | Feb 23, 2026 |
| **RDS PostgreSQL** | ✅ Running (db.t3.small, 2.46M bars + 1.2M features) | Feb 23, 2026 |
| **S3 Buckets (data/models/backups)** | ✅ Existing — data lake, 44 models, backups | Feb 23, 2026 |
| **S3 kenny-dashboard** | ⬜ Not Created | Feb 23, 2026 |
| **API Gateway REST** | ⬜ Not Created — CRITICAL for dashboard connectivity | Feb 23, 2026 |
| **API Gateway WebSocket** | ⬜ Not Created — CRITICAL for real-time push | Feb 23, 2026 |
| **Lambda: kenny-tv-webhook** | ⬜ Not Created | Feb 23, 2026 |
| **Lambda: kenny-ws-connect** | ⬜ Not Created | Feb 23, 2026 |
| **Lambda: kenny-ws-disconnect** | ⬜ Not Created | Feb 23, 2026 |
| **Lambda: kenny-ws-broadcaster** | ⬜ Not Created | Feb 23, 2026 |
| **CloudFront (kenny-dashboard)** | ⬜ Not Created | Feb 23, 2026 |
| **EventBridge (market hours)** | ⬜ Not Created | Feb 23, 2026 |
| **FastAPI Backend** | ⬜ Not Created — clean build with proper routing | Feb 23, 2026 |
| **Schwab WebSocket Client** | ⬜ Not Created — real-time data source | Feb 23, 2026 |
| **ML ModelManager** | 🔄 Code exists from YELENA (789 lines) — needs clean integration | Feb 23, 2026 |
| **Feature Engine (Streaming)** | 🔄 Code exists from YELENA (530 lines) — needs clean integration | Feb 23, 2026 |
| **Feature Engine (Batch)** | 🔄 Code exists from YELENA (1362 lines) — needs clean integration | Feb 23, 2026 |
| **Orchestrator** | ⬜ Not Created — Schwab WebSocket replaces Polygon | Feb 23, 2026 |
| **Verdict Engine** | 🔄 Code exists from YELENA (884 lines) — needs clean integration with TV wired | Feb 23, 2026 |
| **TV Confluence Engine** | 🔄 Code exists from YELENA (473 lines) — needs clean integration | Feb 23, 2026 |
| **TV Alerts v2** | 🔄 Code exists from YELENA (574 lines) — needs clean integration | Feb 23, 2026 |
| **React Dashboard** | ⬜ Not Created — clean build targeting API Gateway (not EC2 direct) | Feb 23, 2026 |
| **Systemd Services** | ⬜ Not Created | Feb 23, 2026 |
| **Pine Script: All 10 Indicators** | ✅ Code complete from YELENA — needs repo copy | Feb 23, 2026 |
| **Pine Script: Master Confluence** | ⚠️ 0/22 sources wired — MUST fix in TradingView | Feb 23, 2026 |
| **ML Models (44 files, 53.9MB)** | ✅ On S3: s3://yelena-models/models_v2/ | Feb 23, 2026 |
| **Historical Data (2.46M bars)** | ✅ In RDS across 5 timeframes | Feb 23, 2026 |
| **Feature Data (1.2M rows, v2)** | ✅ In RDS features table | Feb 23, 2026 |
| **Training Data (1.75M rows)** | ✅ On S3: s3://yelena-data-lake/training-data/ | Feb 23, 2026 |
| **RDS: tv_signals table** | ⬜ Not Created | Feb 23, 2026 |
| **RDS: verdicts table** | ⬜ Not Created | Feb 23, 2026 |
| **RDS: predictions table** | ⬜ Not Created | Feb 23, 2026 |
| **RDS: trade_plans table** | ⬜ Not Created | Feb 23, 2026 |
| **RDS: trades table** | ⬜ Not Created | Feb 23, 2026 |
| **Paper Trading** | ⬜ Not Started — blocked on full system completion | Feb 23, 2026 |

**Status Legend**: ⬜ Not Started | 🔄 Code Exists (needs integration) | ✅ Complete | ⚠️ Needs Attention | ❌ Failed

---

## Last Updated

- **Date**: February 23, 2026
- **Last Completed Step**: Project planning and architecture finalization. Decision to create KENNY as fresh repo with clean wiring, reusing YELENA's proven assets (models, data, Pine Script).
- **Next Step**: Phase 0 — Create GitHub repo, project structure, EC2 cleanup, verify inherited assets.
- **Active Tracks**: Architecture finalization → Phase 0 setup
- **Blocking Issues**: None — ready to begin

---

## Project Overview

### What is KENNY?

KENNY is an AI-powered options day trading platform — the clean-slate successor to YELENA. It reuses all proven components (44 ML models, 2.46M bars, 10 Pine Script indicators, Feature Engine v2) while completely rebuilding the application layer with proper wiring.

### Core Architecture

```
Schwab WebSocket (real-time) → EC2 Orchestrator → Feature Engine → ML Ensemble
                                                                        ↓
TradingView Webhooks → API Gateway → Lambda → TV Signal Store    → Verdict Engine
                                                                        ↓
                                                               API Gateway WebSocket
                                                                        ↓
                                                               React Dashboard (CloudFront)
```

### Key Differences from YELENA

| Aspect | YELENA (broken) | KENNY (fixed) |
|--------|----------------|---------------|
| Dashboard connectivity | Direct EC2 → 404 errors | API Gateway HTTPS → EC2 |
| Real-time push | No WebSocket infrastructure | API Gateway WebSocket (WSS) |
| TV webhooks | Lambda existed but not deployed | Lambda from day one |
| TV-ML synergy | Code built, never wired | Wired during Phase 2 build |
| Real-time data | Polygon (15-min delayed, $108/mo) | Schwab (real-time, $29/mo total) |
| Code organization | Legacy v1 files, wrong imports | Clean from day one |

---

## Inherited Assets (From YELENA AWS Account)

### ML Models — 44 Files, 53.9MB (✅ Proven)

| Timeframe | CALL WR | CALL PF | PUT WR | PUT PF | Models |
|-----------|---------|---------|--------|--------|--------|
| 1min | 68.2% | 4.29 | 67.8% | 4.21 | XGB + TF + CNN + RL + scalers |
| 5min | 74.5% | 5.82 | 74.3% | 5.79 | XGB + TF + CNN + RL + scalers |
| **15min** ⭐ | **87.8%** | **14.46** | **86.3%** | **12.63** | XGB + TF + CNN + RL + scalers |
| 1hr | 80.1% | 8.04 | 82.5% | 9.43 | XGB + TF + CNN + RL + scalers |

**Location**: `s3://yelena-models/models_v2/` organized by timeframe (5min/, 15min/, 1hr/, 1min/)
**Total**: 44 files across XGBoost (.json), Transformer (.pt), CNN (.pt), RL Agent (.pt), scalers (.pkl), configs (.json)

### Historical Data — 2.46M Bars (✅ Clean)

| Table | Rows | Coverage |
|-------|------|----------|
| bars_1min | 1,074,985 | ~6 months |
| bars_5min | 961,829 | ~2 years |
| bars_15min | 335,824 | ~2 years |
| bars_1hr | 85,879 | ~2 years |
| bars_daily | 5,445 | ~2 years |
| **Total** | **2,463,962** | |

### Feature Data — 1.2M Rows, 163 Features (✅ Backfilled)

| Timeframe | Rows | Non-null Features |
|-----------|------|-------------------|
| 1min | 560,884 | 162/163 |
| 5min | 453,794 | 153/163 |
| 15min | 151,536 | 148/163 |
| 1hr | 37,950 | 137/163 |
| **Total** | **1,204,164** | |

### Training Data — 1.75M Rows (✅ On S3)

| Timeframe | Rows | Size | Location |
|-----------|------|------|----------|
| 1min | 1,117,038 | 1.3GB | s3://yelena-data-lake/training-data/ |
| 5min | 451,429 | 600MB | s3://yelena-data-lake/training-data/ |
| 15min | 149,226 | 193MB | s3://yelena-data-lake/training-data/ |
| 1hr | 35,662 | 45MB | s3://yelena-data-lake/training-data/ |
| **Total** | **1,753,355** | **2.1GB** | |

### Pine Script Code (✅ All 10 Indicators + Optimizer)

All indicators have real backtested parameters (not placeholders). Code lives in YELENA GitHub repo and will be copied to KENNY repo.

### AWS Infrastructure (✅ All Running)

| Resource | Identifier | Status |
|----------|-----------|--------|
| EC2 | i-0c9a8339948b61b12 (t3.large, 52.71.0.160) | ✅ Running |
| RDS | yelena-db.ci3ykeq8mckh.us-east-1.rds.amazonaws.com | ✅ Running |
| S3: yelena-data-lake | Raw data, training CSVs | ✅ |
| S3: yelena-models | 44 v2 model files | ✅ |
| S3: yelena-backups | Database backups | ✅ |
| SSM: /yelena/* | 6 parameters (API keys, DB URL, passphrase) | ✅ |
| VPC: yelena-vpc | 10.0.0.0/16, 3 subnets, IGW | ✅ |
| Security Groups | sg-ec2 (22,80,443,8000), sg-rds (5432 from sg-ec2) | ✅ |
| IAM: yelena-ec2-role | S3Full, SSMRead, SageMakerFull, CloudWatchFull | ✅ |

### Colab Notebooks (✅ On Google Drive)

| Notebook | Purpose |
|----------|---------|
| yelena_5min_full_training_v2.ipynb | 5min all 4 models + ensemble |
| yelena_1min_training_v2.ipynb | 1min all 4 models + ensemble |
| yelena_15min_1hr_training_v2.ipynb | 15min + 1hr all models |
| shap_feature_importance_v2.ipynb | SHAP analysis across TFs |
| + v1 notebooks (preserved for reference) | |

---

## Development Phases — Detailed Checklists

### PHASE 0: KENNY Setup (Day 1)

- [ ] **Step 0.1**: Create GitHub repo `kenny`
- [ ] **Step 0.2**: Define project structure (see Master Plan Appendix A)
- [ ] **Step 0.3**: Nuke ~/yelena/ on EC2
- [ ] **Step 0.4**: Clone kenny repo to EC2 (`git clone https://github.com/talonreeder/kenny.git`)
- [ ] **Step 0.5**: Create kenny-dashboard S3 bucket
- [ ] **Step 0.6**: Copy Pine Script code from YELENA repo to kenny repo
- [ ] **Step 0.7**: Copy proven Python modules (Feature Engines, ML model classes, optimizer)
- [ ] **Step 0.8**: Set up Python venv on EC2, install dependencies
- [ ] **Step 0.9**: Verify ML models accessible from EC2 (`aws s3 ls s3://yelena-models/models_v2/`)
- [ ] **Step 0.10**: Verify RDS data accessible (connect, check row counts)
- [ ] **Step 0.11**: Create new RDS tables (tv_signals, verdicts, predictions, trade_plans, trades)
- [ ] **Step 0.12**: Git push initial structure
- [ ] **GATE**: EC2 has clean kenny repo, new tables created, all inherited data accessible, models synced

### PHASE 1: Connectivity Layer (Days 2-3) — THE FIX

- [ ] **Step 1.1**: Create API Gateway REST API (`kenny-rest-api`)
  - [ ] Route: POST /api/tv-alert → Lambda: kenny-tv-webhook
  - [ ] Route: GET /api/health → EC2 integration (HTTP proxy to :8000)
  - [ ] Route: GET /api/symbols → EC2 integration
  - [ ] Route: GET /api/signals/* → EC2 integration
  - [ ] Route: GET /api/verdicts/* → EC2 integration
  - [ ] Route: POST /api/verdicts/* → EC2 integration
- [ ] **Step 1.2**: Create Lambda: kenny-tv-webhook
  - [ ] Passphrase validation
  - [ ] Rich payload parsing (9 components)
  - [ ] Store to RDS tv_signals table
  - [ ] IAM role with RDS access
- [ ] **Step 1.3**: Create API Gateway WebSocket API (`kenny-websocket-api`)
  - [ ] $connect route → Lambda: kenny-ws-connect
  - [ ] $disconnect route → Lambda: kenny-ws-disconnect
  - [ ] broadcast route ← EC2 pushes via Management API
- [ ] **Step 1.4**: Create Lambda: kenny-ws-connect
  - [ ] Store connection ID (DynamoDB connections table)
- [ ] **Step 1.5**: Create Lambda: kenny-ws-disconnect
  - [ ] Remove connection ID
- [ ] **Step 1.6**: Create Lambda: kenny-ws-broadcaster
  - [ ] Read all connection IDs
  - [ ] Post message to each via Management API
  - [ ] Handle stale connections
- [ ] **Step 1.7**: Create DynamoDB: kenny-ws-connections (for WebSocket connection tracking)
- [ ] **Step 1.8**: Create CloudFront distribution for kenny-dashboard bucket
- [ ] **Step 1.9**: Test: REST API → EC2 health check (HTTPS roundtrip)
- [ ] **Step 1.10**: Test: WebSocket connect + disconnect
- [ ] **Step 1.11**: Test: TV webhook → Lambda → RDS
- [ ] **Step 1.12**: Create EventBridge rules (EC2 start/stop market hours)
- [ ] **Step 1.13**: Configure CORS on API Gateway
- [ ] **GATE**: API Gateway REST returns data from EC2. WebSocket connections work. TV webhook stores to DB. CloudFront serves static content.

### PHASE 2: Backend Core (Days 3-5)

- [ ] **Step 2.1**: FastAPI main.py — clean app with ALL routers mounted correctly
- [ ] **Step 2.2**: config.py — SSM-backed configuration
- [ ] **Step 2.3**: database.py — asyncpg connection pool
- [ ] **Step 2.4**: /health endpoint — returns system status (DB connection, model count, Schwab status)
- [ ] **Step 2.5**: /api/symbols — returns tracked symbols from DB
- [ ] **Step 2.6**: signals.py router — /api/signals storage + retrieval
- [ ] **Step 2.7**: verdicts.py router — /api/verdicts storage + retrieval + stats
- [ ] **Step 2.8**: tv_alerts.py router — /api/tv-alert rich payload parsing + TVAlertStore
  - [ ] Pydantic models for rich Master Confluence payload
  - [ ] Per-symbol+timeframe storage with 10-min TTL
  - [ ] DB persistence to tv_signals table
  - [ ] get_multi_tf_confluence() for cross-TF TV alignment
- [ ] **Step 2.9**: WebSocket manager — broadcast verdicts to API Gateway WebSocket
- [ ] **Step 2.10**: Verify ALL routers mounted in main.py (import check, endpoint listing)
- [ ] **Step 2.11**: Test all endpoints via API Gateway
- [ ] **GATE**: All endpoints respond correctly through API Gateway. TV alerts store to DB. No disconnected modules.

### PHASE 3: ML Inference Pipeline (Days 5-7)

- [ ] **Step 3.1**: Copy/adapt ModelManager from YELENA (789 lines)
  - [ ] Load all 44 models from S3/local
  - [ ] predict() and batch_predict() with ensemble methods
  - [ ] Hot-swap via /reload endpoint
  - [ ] Handle variable n_features per timeframe (163/156/149/139)
- [ ] **Step 3.2**: Copy/adapt Streaming FeatureEngine from YELENA (530 lines)
  - [ ] Real-time computation from 200-bar OHLCV window
  - [ ] TA-Lib aligned to training code (exact naming match)
- [ ] **Step 3.3**: Verify all 44 models load correctly
  - [ ] Transformer architecture (encoder/head naming)
  - [ ] CNN dual-conv branch architecture
  - [ ] XGBoost feature_names match
  - [ ] RL Agent PPO architecture
- [ ] **Step 3.4**: Test single prediction (feature vector → ModelManager → output)
- [ ] **Step 3.5**: Test ensemble (all 3 prediction models + RL agent)
- [ ] **Step 3.6**: Test cross-timeframe agreement scoring
- [ ] **GATE**: All models loaded, predictions accurate, ensemble producing expected confidence ranges.

### PHASE 4: Orchestrator + Verdict Engine (Days 7-9)

- [ ] **Step 4.1**: Build Schwab WebSocket client (schwab-py)
  - [ ] OAuth token management (auto-refresh, 7-day expiry handling)
  - [ ] Real-time tick subscription for all 11 symbols
  - [ ] Connection health monitoring + reconnect logic
- [ ] **Step 4.2**: Bar aggregation (ticks → 1/5/15/60min OHLCV bars)
- [ ] **Step 4.3**: Build orchestrator pipeline
  - [ ] Bar close → Streaming Feature Engine → 163 features
  - [ ] Features → ModelManager.predict() → predictions for all active TFs
  - [ ] Predictions → Cross-TF Agreement scoring
  - [ ] TV Confluence check (if fresh signal exists)
  - [ ] → Verdict Engine → GO/SKIP/EXIT
- [ ] **Step 4.4**: Copy/adapt TVConfluenceIntegrator from YELENA (473 lines)
  - [ ] Grade-aware confidence adjustments
  - [ ] 7-level agreement classification
  - [ ] Freshness decay
  - [ ] Component quality signal extraction
- [ ] **Step 4.5**: Copy/adapt Verdict Engine from YELENA (884 lines)
  - [ ] **Wire set_tv_store() from day one** (the missing call in YELENA)
  - [ ] TV synergy integration in verdict generation
  - [ ] ATR-based entry/SL/TP calculation
  - [ ] Full reasoning output
- [ ] **Step 4.6**: Wire orchestrator → API Gateway WebSocket broadcast
  - [ ] On new verdict: call kenny-ws-broadcaster Lambda (or direct Management API)
  - [ ] Verify verdicts reach connected dashboard clients
- [ ] **Step 4.7**: Create systemd services (kenny-api.service, kenny-orchestrator.service)
- [ ] **Step 4.8**: Test with live Schwab data
  - [ ] Verify bars aggregating correctly
  - [ ] Verify features computing
  - [ ] Verify predictions generating
  - [ ] Verify verdicts appearing on dashboard
- [ ] **GATE**: Complete pipeline working with real Schwab data. Verdicts pushing to dashboard in real-time.

### PHASE 5: Dashboard (Days 9-11)

- [ ] **Step 5.1**: React app scaffold (Vite + React 19 + Tailwind 3.3.3)
- [ ] **Step 5.2**: WebSocket connection to API Gateway WSS endpoint
- [ ] **Step 5.3**: Verdict cards (GO/SKIP + direction + confidence + entry/SL/TP + reasoning)
- [ ] **Step 5.4**: System health indicators (Schwab connection, ML status, last verdict time)
- [ ] **Step 5.5**: Multi-timeframe view (all TFs simultaneously)
- [ ] **Step 5.6**: Signal history (recent verdicts log)
- [ ] **Step 5.7**: Build and deploy (npm run build → S3 kenny-dashboard → CloudFront)
- [ ] **Step 5.8**: Test dashboard receives live verdicts via WebSocket
- [ ] **GATE**: Dashboard live on CloudFront. Verdicts appear in real-time via WSS. No 404s.

### PHASE 6: TradingView Integration (Days 11-13)

- [ ] **Step 6.1**: Fix Master Confluence source wiring (manual in TradingView — all 22 input.source() fields)
- [ ] **Step 6.2**: Verify MC generating correct CALL and PUT signals (not PUT-only)
- [ ] **Step 6.3**: Configure webhook alerts (MC → API Gateway → Lambda → RDS)
- [ ] **Step 6.4**: Test TV-ML synergy live (TV signal + ML prediction → confidence modulation)
- [ ] **Step 6.5**: Verify TV agreement visible in verdict reasoning on dashboard
- [ ] **GATE**: Master Confluence generating balanced signals. Webhooks flowing. TV-ML synergy working in verdicts.

### PHASE 7: Paper Trading Validation (Days 13-20+)

- [ ] **Step 7.1**: Full system integration test (all components stable)
- [ ] **Step 7.2**: Paper trading (1 week minimum)
- [ ] **Step 7.3**: Daily trade logging
- [ ] **Step 7.4**: Performance tracking (win rate, profit factor, R:R, drawdown)
- [ ] **Step 7.5**: Bug fixes and tuning
- [ ] **Step 7.6**: Go/No-go decision
- [ ] **GATE**: Win rate ≥65%, profit factor ≥1.5, system stable, latency <1 second.

---

## External Accounts & Subscriptions

| Service | Status | Cost | Notes |
|---------|--------|------|-------|
| AWS Account | ✅ Active | ~$47/mo (EC2+RDS+S3) | us-east-1 |
| Polygon.io Stocks Starter | ✅ Active | $29/mo | Historical backfill only |
| Polygon.io Options Starter | ❌ Dropping | $0 (was $79/mo) | Will re-add for Options ML |
| Schwab Developer Account | ✅ Active | $0 | API key + secret, OAuth completed |
| Schwab Trading Account | ✅ Active | $0 | $500+ equity, tokens obtained |
| TradingView Premium | ✅ Active | Already paid | Pine Script v6 |
| Google Colab Pro | ✅ Active | $12/mo | GPU training |
| GitHub | ✅ Active | Free | New repo: kenny |

---

## Technology Stack

| Component | Technology | Status |
|-----------|------------|--------|
| Backend | Python 3.12 + FastAPI | ⬜ Building |
| Database | PostgreSQL 15 (RDS, native partitioning) | ✅ Inherited |
| Real-Time Data | Schwab WebSocket (schwab-py) | ⬜ Building |
| Historical Data | Polygon.io REST | ✅ Inherited |
| ML Inference | EC2 in-process (ModelManager, 44 models) | ⬜ Building |
| ML Training | Google Colab Pro ($12/mo) | ✅ Available |
| ML Lifecycle | SageMaker (Registry + Pipelines) | ⬜ Phase 1.5 |
| API Routing | API Gateway (REST + WebSocket) | ⬜ Building |
| Webhooks | Lambda (Python 3.12) | ⬜ Building |
| Frontend | React 19 + Vite + Tailwind 3.3.3 | ⬜ Building |
| CDN | CloudFront | ⬜ Building |
| Indicators | Pine Script v6 (10 indicators) | ✅ Code complete |

---

## AWS Resources

### Existing (Inherited from YELENA)

| Resource | Identifier | Status |
|----------|-----------|--------|
| EC2 | i-0c9a8339948b61b12 (t3.large, 8GB RAM, 52.71.0.160) | ✅ Running |
| RDS | yelena-db.ci3ykeq8mckh.us-east-1.rds.amazonaws.com (db.t3.small) | ✅ Running |
| S3: yelena-data-lake | Training data, raw market data | ✅ |
| S3: yelena-models | 44 v2 model files (models_v2/) | ✅ |
| S3: yelena-backups | Database backups | ✅ |
| VPC: yelena-vpc | 10.0.0.0/16 | ✅ |
| SSM Parameters | /yelena/* (6 params) | ✅ |
| IAM: yelena-ec2-role | S3Full + SSMRead + SageMaker + CloudWatch | ✅ |

### New (To Be Created for KENNY)

| Resource | Purpose | Status |
|----------|---------|--------|
| S3: kenny-dashboard | React dashboard static files | ⬜ |
| API Gateway: kenny-rest-api | HTTPS routing for /api/* | ⬜ |
| API Gateway: kenny-websocket-api | WSS for real-time push | ⬜ |
| Lambda: kenny-tv-webhook | TV alert handler | ⬜ |
| Lambda: kenny-ws-connect | WebSocket connection | ⬜ |
| Lambda: kenny-ws-disconnect | WebSocket disconnection | ⬜ |
| Lambda: kenny-ws-broadcaster | Push verdicts to clients | ⬜ |
| DynamoDB: kenny-ws-connections | WebSocket connection IDs | ⬜ |
| CloudFront: kenny distribution | CDN for dashboard | ⬜ |
| EventBridge: market-hours rules | EC2 start/stop scheduling | ⬜ |
| IAM: kenny-lambda-role | Lambda execution + RDS + DynamoDB + API Gateway | ⬜ |

---

## Database Stats (Inherited)

| Table | Row Count | Status |
|-------|-----------|--------|
| bars_1min | 1,074,985 | ✅ Inherited |
| bars_5min | 961,829 | ✅ Inherited |
| bars_15min | 335,824 | ✅ Inherited |
| bars_1hr | 85,879 | ✅ Inherited |
| bars_daily | 5,445 | ✅ Inherited |
| features | 1,204,164 | ✅ Inherited (163 features, v2) |
| symbols | 12 | ✅ Inherited |
| tv_signals | 0 | ⬜ Table not created yet |
| verdicts | 0 | ⬜ Table not created yet |
| predictions | 0 | ⬜ Table not created yet |
| trade_plans | 0 | ⬜ Table not created yet |
| trades | 0 | ⬜ Table not created yet |

**Symbols**: SPY, QQQ, TSLA, NVDA, META, AAPL, GOOGL, MSFT, AMZN, AMD, NFLX (+ SPX with daily bars only)

---

## YELENA Code to Reuse (Proven Modules)

These files were built and tested during YELENA development. They need clean integration into KENNY's structure:

| Module | YELENA Lines | Purpose | Integration Notes |
|--------|-------------|---------|-------------------|
| ModelManager | 789 | Loads 44 models, predict(), batch_predict(), hot-swap | Adapt to kenny paths |
| Feature Engine (Streaming) | 530 | Real-time 163 features from 200-bar window | Port as-is, verify TA-Lib alignment |
| Feature Engine (Batch) | 1,362 | Historical feature computation | Port as-is |
| Verdict Engine | 884 | GO/SKIP/EXIT with TV synergy upgrade | **Wire set_tv_store() from day one** |
| TV Confluence Integrator | 473 | Grade-aware confidence modulation | Port as-is |
| TV Alerts v2 | 574 | Rich webhook payload parsing + TVAlertStore | Port as-is |
| Parameter Optimizer | 1,348+ | Pine Script parameter optimization | Port as-is |
| Generate Training Data | 571 | Pull from Feature Engine, hybrid labeling | Port as-is |
| Backfill Features | 356 | MTF feature batch backfill | Port as-is |
| Prediction Service | 428 | FastAPI prediction endpoints | May merge into main app |

---

## Cost Tracking

| Month | AWS | Polygon | Schwab | Colab | Total | Notes |
|-------|-----|---------|--------|-------|-------|-------|
| Feb 2026 | ~$90 est | $108 | $0 | $12 | ~$210 | YELENA costs (includes Polygon Options $79) |
| Mar 2026 (KENNY) | ~$50 est | $29 | $0 | $12 | ~$91 | Dropped Polygon Options, added API Gateway/Lambda (free tier) |

---

## Lessons Learned (From YELENA — Apply to ALL KENNY Development)

1. **Dashboard NEVER talks directly to EC2** — Everything through API Gateway
2. **Build and deploy modules, don't just write them** — Code in a file ≠ code deployed and wired
3. **Test the WIRING first, then build more components** — Connectivity is architecture
4. **main.py must import the RIGHT modules** — Verify imports match actual file names
5. **Wire TV synergy from day one** — Don't build verdict_engine with set_tv_store() then forget to call it
6. **Two feature_engine.py files** — batch (1362 lines) and streaming (530 lines). NEVER confuse them.
7. **TradingView input.source() requires MANUAL wiring** — Default "close" reads raw price, not indicator values
8. **Pine Script v6**: No `transp`, no `when`, booleans can't be `na`, `ta.*` at global scope, single-line ternaries
9. **QLine MUST use built-in `ta.supertrend()`** — Never manual implementation
10. **Feature names must match EXACTLY between training and inference** — Even small differences produce garbage
11. **Variable n_features per timeframe**: 1min: 163, 5min: 156, 15min: 149, 1hr: 139
12. **Simple Average ensemble beats weighted** — Diversity > individual strength
13. **Unanimous 3/3 model agreement** = strongest quality filter (+4-7pts WR)
14. **15min is the best timeframe for ML** — Best signal-to-noise ratio
15. **Documents only grow** — Never shorten or condense. Losing context costs more than long files.
16. **SageMaker endpoints add latency** for small models (54MB) — In-process inference is faster
17. **Schwab refresh tokens expire after 7 days** — Need auto-refresh or notification
18. **Weekend bars exist in data** (6,159 rows) — Filter in market-hours queries
19. **SPX has 0 minute bars from Polygon** — Index data, daily only
20. **RDS db.t3.micro depletes burst credits** under heavy writes — Already upgraded to db.t3.small

---

## Session Notes

### February 23, 2026 (KENNY Project Creation)

- **Decision**: Create fresh KENNY project rather than fixing YELENA wiring
- **Rationale**: YELENA codebase has too many legacy issues (wrong imports, undeployed files, missing wiring). Starting clean is faster than debugging.
- **Key architecture decisions**:
  - Fresh GitHub repo: `kenny`
  - Same AWS account, same infrastructure (EC2, RDS, S3, VPC, SSM)
  - New: API Gateway (REST + WebSocket), Lambda functions, CloudFront distribution, kenny-dashboard S3 bucket
  - Real-time data: Schwab WebSocket ($0/mo) replaces Polygon for live trading
  - Historical data: Polygon $29/mo kept for backfill/training only
  - Polygon Options $79/mo DROPPED (Schwab handles live chains; will re-add for Options ML training)
  - ML inference: EC2 in-process (ModelManager) — SageMaker endpoints NOT used
  - SageMaker: MLOps lifecycle only (Registry + Pipelines)
  - Trade execution: Manual in v1 (AI tells user what to do)
  - Dashboard: React 19 + Vite + Tailwind 3.3.3 (same stack)
  - Timeframes v1: 1/5/15/60min; v1.1: add 2min + SPX
  - Symbols v1: 11 (SPY, QQQ, TSLA, NVDA, META, AAPL, GOOGL, MSFT, AMZN, AMD, NFLX)
- **Created**: KENNY_MASTER_PLAN.md and KENNY_PROJECT_STATE.md
- **Next**: Phase 0 — Create repo, set up project structure, verify inherited assets

---

## Quick Reference

### GitHub Repository
```
https://github.com/talonreeder/kenny.git
```

### SSH to EC2
```bash
ssh -i yelena-key.pem ubuntu@52.71.0.160
```

### Connect to RDS from EC2
```bash
psql -h yelena-db.ci3ykeq8mckh.us-east-1.rds.amazonaws.com -U postgres -d yelena
```

### Activate Python Environment (EC2)
```bash
cd ~/kenny/backend
source ../venv/bin/activate
```

### Sync Models from S3
```bash
aws s3 sync s3://yelena-models/models_v2/ ~/kenny/backend/models/
```

### Deploy Dashboard
```bash
cd ~/kenny/frontend
npm run build
aws s3 sync dist/ s3://kenny-dashboard/ --delete
aws cloudfront create-invalidation --distribution-id <ID> --paths "/*"
```

### Systemd Services
```bash
sudo systemctl start kenny-api
sudo systemctl status kenny-api
journalctl -u kenny-api -f
```

---

## How to Use This Document

### When Ending a Session:
1. Update "Last Updated" section with current date and exact step
2. Check off completed items in checklists (`[ ]` → `[x]`)
3. Update resource tables with real values (IDs, endpoints, ARNs)
4. Note any issues or blockers
5. Add session notes with what was done
6. **NEVER shorten or remove existing content — only ADD**

### When Starting a New Chat:
1. Upload KENNY_PROJECT_STATE.md AND KENNY_MASTER_PLAN.md
2. Claude will read both and continue exactly where you left off

### Key Rules:
- **Never skip steps** — complete and test each before moving on
- **Always update this document** before ending a session
- **Documents only grow** — never shorten, condense, or remove sections
- **Track real metrics** — no placeholder data
- **Wiring is architecture** — test connections before building more
