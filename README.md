# KENNY — AI Options Day Trading Platform

AI-powered options day trading platform built on a 5-layer cognitive architecture. Combines ML ensemble predictions (XGBoost, Transformer, CNN, RL Agent) with TradingView indicator confluence for high-probability trade signal generation.

## Architecture
```
Schwab WebSocket (real-time) → EC2 Orchestrator → Feature Engine (163 features)
                                    → ML Ensemble (44 models) → Verdict Engine
TradingView Webhooks → API Gateway → Lambda → TV Signal Store → TV-ML Synergy
                                                                      ↓
                                                    API Gateway WebSocket → Dashboard
```

## Tech Stack

- **Backend**: Python 3.12, FastAPI, asyncpg
- **ML**: XGBoost, PyTorch (Transformer, CNN, RL Agent), 44 models
- **Data**: Schwab WebSocket (real-time), Polygon REST (historical)
- **Frontend**: React 19, Vite, Tailwind CSS
- **Infrastructure**: AWS (EC2, RDS PostgreSQL, API Gateway, Lambda, CloudFront, S3)
- **Indicators**: TradingView Pine Script v6 (10 custom indicators)

## Documentation

- [KENNY Master Plan](docs/KENNY_MASTER_PLAN.md) — Complete architecture & development plan
- [KENNY Project State](docs/KENNY_PROJECT_STATE.md) — Live development tracker
