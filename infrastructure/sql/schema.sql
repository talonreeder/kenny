-- KENNY Database Schema — New tables (preserves existing bars_*, features, symbols)

CREATE TABLE IF NOT EXISTS tv_signals (
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
    is_rich         BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_tv_signals_symbol_time ON tv_signals (symbol, time DESC);

CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol          VARCHAR(10) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    model_name      VARCHAR(50) NOT NULL,
    signal          VARCHAR(10) NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    probabilities   JSONB NOT NULL,
    model_version   VARCHAR(20)
);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_tf_time ON predictions (symbol, timeframe, time DESC);

CREATE TABLE IF NOT EXISTS verdicts (
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
CREATE INDEX IF NOT EXISTS idx_verdicts_symbol_time ON verdicts (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_verdicts_verdict_time ON verdicts (verdict, time DESC);

CREATE TABLE IF NOT EXISTS trade_plans (
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
CREATE INDEX IF NOT EXISTS idx_trade_plans_symbol_time ON trade_plans (symbol, time DESC);

CREATE TABLE IF NOT EXISTS trades (
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
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, time_entered DESC);