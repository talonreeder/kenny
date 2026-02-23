# YELENA v2: Master Confluence — Education & Setup Guide

> **Version**: 1.0
> **Last Updated**: February 16, 2026
> **Prerequisite**: All 9 sub-indicators must be added to TradingView first

---

## What Is Master Confluence?

Master Confluence is YELENA's "Brain" — the single indicator you watch during live trading. It reads signals from all 9 sub-indicators, scores them on a -10 to +10 scale, assigns a letter grade, and outputs complete trade plans with entry, stop-loss, and three take-profit levels.

You don't need to interpret 9 different charts. Master Confluence does it for you and tells you: **take this trade** or **skip it**.

### The Decision Flow

```
9 Sub-Indicators (running silently on chart)
        │
        ▼
Master Confluence reads their hidden exports via input.source()
        │
        ▼
Scores each component → Total Score (-10 to +10)
        │
        ▼
Assigns Grade (A+, A, B+, B)
        │
        ▼
A/A+ only → Generates trade plan (Entry, SL, TP1/TP2/TP3)
        │
        ▼
Fires webhook alert to AWS pipeline
```

---

## Part 1: Prerequisites — The 9 Sub-Indicators

Before Master Confluence can work, all 9 sub-indicators must be on the same chart. Each one auto-optimizes its parameters based on the ticker and timeframe — no manual tuning needed.

### Indicator Loading Order

Add these to your chart in any order. Each exports hidden plots (prefixed `X_`) that Master Confluence reads:

| # | Indicator | Pane | What It Does | Key Exports |
|---|-----------|------|-------------|-------------|
| 1 | **QCloud** | Price overlay | Multi-MA trend structure (5 MAs forming a cloud) | Bull count, direction, squeeze |
| 2 | **QLine** | Price overlay | Dynamic support/resistance trendline | Trend, bounce score, extended |
| 3 | **QWave** | Separate pane | ADX/DI-based trend strength scoring | Wave score, trending flag |
| 4 | **QBands** | Price overlay | Bollinger + Keltner squeeze detection | Band position, squeeze fire |
| 5 | **Moneyball** | Separate pane | Smoothed ROC momentum oscillator | Value, zone |
| 6 | **QMomentum** | Separate pane | RSI + Stochastic + divergence detection | RSI, divergence |
| 7 | **QCVD** | Separate pane | Cumulative Volume Delta analysis | Trend, spike |
| 8 | **QSMC** | Price overlay | Smart Money Concepts (BOS, CHoCH, Order Blocks) | Structure, BOS, CHoCH |
| 9 | **QGrid** | Price overlay | S/R levels + VWAP framework | VWAP side, resistance, support |

### Adding Indicators to TradingView

1. Open TradingView → Pine Script editor
2. For each indicator, paste its code and click **Add to Chart**
3. Once all 9 are on the chart, add **Master Confluence** last
4. Indicators with `overlay=true` appear on the price chart; others get separate panes

> **TIP**: You can collapse/minimize the separate pane indicators (QWave, Moneyball, QMomentum, QCVD) since Master Confluence reads their data automatically. They just need to be present.

---

## Part 2: Wiring Master Confluence Sources

This is the most important setup step. Master Confluence reads 22 hidden plot values from the 9 sub-indicators via `input.source()`. You need to wire each source to the correct hidden plot **once per chart**.

### How Source Wiring Works

When you add Master Confluence, its settings panel shows grouped inputs like:

```
🔗 QCloud
  ├── Bull Count (X_BullCount)     → defaults to "close"
  ├── Direction (X_Direction)       → defaults to "close"
  └── Squeeze (X_IsSqueeze)         → defaults to "close"

🔗 QLine
  ├── Trend (X_Trend)              → defaults to "close"
  ├── Bounce Score (X_BounceScore) → defaults to "close"
  └── Extended (X_IsExtended)       → defaults to "close"

... (continues for all 9 indicators)
```

Each source defaults to `close` — you need to change it to the matching hidden plot from the correct sub-indicator.

### Step-by-Step Wiring

1. **Open MC Settings**: Click the gear icon on Master Confluence
2. **Find the 🔗 QCloud group**
3. **Click the dropdown** next to "Bull Count (X_BullCount)"
4. **Scroll through the source list** — you'll see entries from all indicators on the chart
5. **Find and select**: `YELENA v2: QCloud → X_BullCount`
6. **Repeat** for every source in every group

### Complete Source Mapping Table

Wire each MC input to exactly this hidden plot:

| MC Input Group | MC Input Name | Wire To → Indicator | Wire To → Hidden Plot |
|---------------|---------------|---------------------|----------------------|
| 🔗 QCloud | Bull Count | QCloud | X_BullCount |
| 🔗 QCloud | Direction | QCloud | X_Direction |
| 🔗 QCloud | Squeeze | QCloud | X_IsSqueeze |
| 🔗 QLine | Trend | QLine | X_Trend |
| 🔗 QLine | Bounce Score | QLine | X_BounceScore |
| 🔗 QLine | Extended | QLine | X_IsExtended |
| 🔗 QWave | Wave Score | QWave | X_QWaveScore |
| 🔗 QWave | Trending | QWave | X_QWaveTrending |
| 🔗 QBands | Band Position | QBands | X_QBandsPosition |
| 🔗 QBands | Squeeze Fire | QBands | X_QBandsSqFire |
| 🔗 Moneyball | Value | Moneyball | X_MoneyballValue |
| 🔗 Moneyball | Zone | Moneyball | X_MoneyballZone |
| 🔗 QMomentum | RSI | QMomentum | X_QMomentumRSI |
| 🔗 QMomentum | Divergence | QMomentum | X_QMomentumDiv |
| 🔗 QCVD | Trend | QCVD | X_QCVDTrend |
| 🔗 QCVD | Spike | QCVD | X_QCVDSpike |
| 🔗 QSMC | Structure | QSMC | X_QSMCStructure |
| 🔗 QSMC | BOS | QSMC | X_QSMCBOS |
| 🔗 QSMC | CHoCH | QSMC | X_QSMCCHoCH |
| 🔗 QGrid | VWAP Side | QGrid | X_QGridVWAPSide |
| 🔗 QGrid | Resistance | QGrid | X_QGridResistance |
| 🔗 QGrid | Support | QGrid | X_QGridSupport |

**Total: 22 source connections**

### Wiring Tips

- **Hidden plots are invisible** on the chart but show up in the source dropdown. Look for entries with the `X_` prefix.
- **The indicator name appears first** in the dropdown (e.g., "YELENA v2: QCloud"), followed by the plot name.
- **This is a one-time setup per chart**. Once wired, the connections persist across sessions.
- **If you add a new chart** (different ticker/timeframe), you'll need to re-wire. Consider using TradingView's chart template feature to save the wired configuration.
- **Verification**: After wiring, MC's score table should show non-zero component scores. If everything shows 0, sources aren't wired correctly.

---

## Part 3: Understanding the Scoring System

### How Each Indicator Contributes

Master Confluence scores 9 components. Each gives a directional score — positive for bullish, negative for bearish:

| Component | Max Points | Bullish Triggers | Bearish Triggers |
|-----------|-----------|-----------------|-----------------|
| **QCloud** | ±1.5 | 4-5 bullish MAs → +1.0 to +1.5 | 0-1 bullish MAs → -1.0 to -1.5 |
| **QLine** | ±1.5 | Bullish trend + strong bounce → up to +1.5 | Bearish trend + strong bounce → down to -1.5 |
| **QWave** | ±1.0 | Wave score > 30 → +0.5, > 60 → +1.0 | Wave score < -30 → -0.5, < -60 → -1.0 |
| **QBands** | ±1.0 | Lower band touch/bounce or bullish squeeze fire | Upper band touch/bounce or bearish squeeze fire |
| **Moneyball** | ±1.0 | Positive value + high zone (4-5) | Negative value + low zone (2-3) |
| **QMomentum** | ±1.0 | RSI leaving oversold or bullish divergence | RSI leaving overbought or bearish divergence |
| **QCVD** | ±1.0 | Bullish flow trend + buy spike | Bearish flow trend + sell spike |
| **QSMC** | ±1.0 | Up structure + bullish BOS/CHoCH | Down structure + bearish BOS/CHoCH |
| **QGrid** | ±1.0 | Above VWAP + near support | Below VWAP + near resistance |

**Total range: -10.0 (max bearish) to +10.0 (max bullish)**

### Reading the Score

Think of the score as a consensus vote among 9 market analysts:

- **+10.0**: Every single indicator screams bullish — extremely rare, extremely high conviction
- **+7.0**: Strong majority bullish — this is a solid CALL signal
- **+3.0**: Slightly bullish lean — not enough conviction to trade
- **0.0**: Dead neutral — equal bull/bear pressure
- **-7.0**: Strong majority bearish — solid PUT signal

---

## Part 4: Signal Grades — What to Trade

### Grade Definitions

| Grade | Score Range | Action | Alert? |
|-------|-----------|--------|--------|
| **A+ CALL** | +8.0 to +10.0 | Maximum conviction long — TAKE THIS TRADE | ✅ Webhook fires |
| **A CALL** | +6.0 to +7.9 | Strong conviction long — TAKE THIS TRADE | ✅ Webhook fires |
| **B+ CALL** | +4.0 to +5.9 | Moderate lean — NOT tradeable | ❌ No alert |
| **B** | -3.9 to +3.9 | Neutral/mixed — WAIT | ❌ No alert |
| **B+ PUT** | -4.0 to -5.9 | Moderate lean — NOT tradeable | ❌ No alert |
| **A PUT** | -6.0 to -7.9 | Strong conviction short — TAKE THIS TRADE | ✅ Webhook fires |
| **A+ PUT** | -8.0 to -10.0 | Maximum conviction short — TAKE THIS TRADE | ✅ Webhook fires |

**The golden rule: Only A and A+ grades generate alerts and are tradeable.** Everything else is noise. This filter keeps you out of low-probability setups.

### Why No B+ Trades?

B+ signals mean 4-6 indicators agree but 3-5 disagree. That's not consensus — it's a coin flip with a slight edge. The power of YELENA is patience: wait for 6+ indicators to align, and your win rate goes way up.

---

## Part 5: Confidence Percentage

The confidence percentage (0-100%) provides additional context beyond the grade. It combines three factors:

### Confidence Components

1. **Score Magnitude (0-50%)**: How far the score is from zero. A score of +10 = 50%, +5 = 25%.

2. **Indicator Agreement (0-30%)**: How many of the 9 indicators point the same direction. 9/9 bullish = 30%, 6/9 = 20%.

3. **Quality Bonuses (0-20%)**: Special conditions that boost confidence:
   - **QCloud squeeze breakout** + high score: +5%
   - **QLine 3-star bounce**: +5%
   - **QSMC Change of Character**: +5%
   - **QCVD institutional spike**: +5%

### Interpreting Confidence

- **85-100%**: Textbook setup — everything aligns perfectly
- **70-84%**: Strong setup — primary drivers agree, maybe 1-2 minor holdouts
- **60-69%**: Minimum for A-grade — tradeable but be prepared for faster exits

---

## Part 6: Trade Plans — Entry, Stop-Loss, Take-Profit

When Master Confluence fires an A or A+ signal, it generates a complete trade plan.

### Entry

Entry is at the current bar's close price when the signal fires. This is the price you're executing at.

### Stop-Loss (Smart Placement)

The stop-loss uses a two-factor approach:

1. **Base SL**: 1.5× ATR(14) from entry (adjustable in settings)
2. **S/R Awareness**: If QGrid's nearest support (for CALLs) or resistance (for PUTs) provides a tighter stop that still gives at least 0.5 ATR of room, it uses that instead

This means your stops respect actual market structure rather than just arbitrary distance.

**Example (CALL signal)**:
- Entry: $595.50
- ATR(14): $1.20
- Base SL: $595.50 - (1.5 × $1.20) = $593.70
- QGrid nearest support: $594.80
- Smart SL: $594.70 (just below support) — tighter and structure-based

### Take-Profit (3-Tier Exit)

The system calculates risk (distance from entry to SL) and scales out in thirds:

| Level | Risk:Reward | Action | Rationale |
|-------|-----------|--------|-----------|
| **TP1** | 1:1 | Take 33% profit | Lock in gains, move SL to breakeven |
| **TP2** | 2:1 | Take 33% profit | Secure majority of profit |
| **TP3** | 3:1 | Let final 33% run | Catch extended moves |

**Example (CALL, Entry $595.50, SL $594.70)**:
- Risk = $595.50 - $594.70 = $0.80
- TP1 = $595.50 + $0.80 = $596.30 (take ⅓)
- TP2 = $595.50 + $1.60 = $597.10 (take ⅓)
- TP3 = $595.50 + $2.40 = $597.90 (let ⅓ run)

### Risk Settings (Adjustable)

These defaults can be changed in MC settings under 📐 Risk:

| Setting | Default | Range | Purpose |
|---------|---------|-------|---------|
| SL ATR Multiplier | 1.5 | 0.5–5.0 | Base stop-loss distance |
| TP1 Risk:Reward | 1.0 | 0.5–5.0 | First take-profit level |
| TP2 Risk:Reward | 2.0 | 1.0–8.0 | Second take-profit level |
| TP3 Risk:Reward | 3.0 | 1.5–10.0 | Third take-profit level |
| Min SL Distance | 0.5 ATR | 0.1–2.0 | Prevents stops too close to entry |

---

## Part 7: Visual Display

### What You See on the Chart

**Signal Labels** (on signal bars only):
- Large green label: `CALL A+ (85%)` with entry/SL/TP summary
- Medium green label: `CALL A (72%)`
- Large red label: `PUT A+ (88%)`
- Medium red label: `PUT A (70%)`
- B+ and below: No labels — not tradeable

**SL/TP Lines** (drawn on signal bars):
- White line: Entry price
- Red dashed line: Stop-loss
- Green dotted line: TP1 (1:1)
- Green dashed line: TP2 (2:1)
- Green solid line: TP3 (3:1)

### Main Score Table

Located in the corner of your chart (position adjustable in settings), this table shows:

| Row | Example Display |
|-----|----------------|
| Header | **Master Confluence v2** — A CALL |
| Signal | 🟢 CALL |
| Score | +7.5 ████████░░ |
| Confidence | 78% |
| Entry | $595.50 |
| Stop Loss | $594.70 (-$0.80) |
| TP1 / TP2 / TP3 | $596.30 / $597.10 / $597.90 |

### Component Detail Table

A second table (toggle on/off in settings) breaks down each indicator's contribution:

| Component | Score | State |
|-----------|-------|-------|
| QCloud | +1.5 | 5/5 BULL |
| QLine | +1.5 | BULL ★★★ |
| QWave | +1.0 | STRONG (72) |
| QBands | +0.5 | SQ FIRE ↑ |
| Moneyball | +1.0 | Zone 5 (+65) |
| QMomentum | +1.0 | RSI leaving OS |
| QCVD | +0.5 | BULL trend |
| QSMC | +1.0 | UP + BOS ↑ |
| QGrid | -0.5 | Below VWAP |

Green rows = bullish contribution, Red = bearish, Gray = neutral. This tells you exactly which indicators agree and which dissent.

---

## Part 8: Setting Up Webhook Alerts

### Creating the Alert

1. On TradingView, right-click your chart → **Create Alert**
2. **Condition**: Select `YELENA v2: Master Confluence`
3. **Alert trigger**: Choose the appropriate alert condition:
   - `CALL A+` — Maximum bullish only
   - `CALL A` — Strong bullish only
   - `PUT A+` — Maximum bearish only
   - `PUT A` — Strong bearish only
   - `Any A-grade signal` — All tradeable signals (recommended)
4. **Webhook URL**: Enter your AWS API Gateway endpoint
5. **Alert message**: Leave as default — MC formats the JSON payload automatically

### Webhook Payload

When an A/A+ signal fires, the webhook sends this JSON to your endpoint:

```json
{
  "passphrase": "YELENA_V2",
  "ticker": "SPY",
  "timeframe": "5",
  "signal": "CALL",
  "grade": "A+",
  "score": 8.5,
  "confidence": 85,
  "entry": 595.50,
  "stop_loss": 594.20,
  "tp1": 596.80,
  "tp2": 598.10,
  "tp3": 599.40,
  "components": {
    "qcloud": {"score": 1.5, "bull_count": 5, "direction": 1, "squeeze": 0},
    "qline": {"score": 1.5, "trend": 1, "bounce": 3, "duration": 8},
    "qwave": {"score": 1.0, "wave_score": 72, "adx": 35},
    "qbands": {"score": 0.5, "position": -1, "squeeze_fire": 1},
    "moneyball": {"score": 1.0, "value": 65, "zone": 5},
    "qmomentum": {"score": 1.0, "rsi": 32, "div": 0},
    "qcvd": {"score": 0.5, "trend": 1, "spike": 0},
    "qsmc": {"score": 1.0, "structure": 1, "bos": 1},
    "qgrid": {"score": 0.5, "vwap_side": 1, "density": 3}
  },
  "timestamp": "2026-02-16T14:30:00Z"
}
```

The passphrase (configurable in MC settings) lets your backend verify the webhook is legitimate.

---

## Part 9: Auto-Optimization — How It Works Behind the Scenes

One of YELENA's key innovations: **you never manually tune indicator parameters**.

### The Problem with Manual Tuning

Traditional indicators use fixed parameters (RSI length 14, Bollinger period 20, etc.). But optimal parameters vary by ticker and timeframe:
- AMD on a 5-min chart needs different RSI sensitivity than SPY on a 15-min chart
- A fast scalping timeframe needs shorter lookback periods than a swing timeframe

### How YELENA Solves This

Each sub-indicator contains a **hardcoded lookup table** with backtested optimal parameters for every supported ticker × timeframe combination:

```
Ticker: SPY, Timeframe: 5min  → RSI length 16, Stoch length 14
Ticker: SPY, Timeframe: 15min → RSI length 16, Stoch length 12
Ticker: AMD, Timeframe: 5min  → RSI length 20, Stoch length 18
```

When the indicator loads, it detects the current chart's ticker and timeframe, looks up the optimal parameters, and applies them automatically. No user action required.

### Supported Tickers and Timeframes

| Tickers | Timeframes |
|---------|-----------|
| SPY, QQQ, TSLA, NVDA, AMD, MSFT, META, NFLX, AAPL, AMZN, GOOG | Scalp (1-5 min), Intraday (15 min), Swing (60 min) |

If you chart a ticker or timeframe not in the lookup table, indicators fall back to sensible defaults.

### Why Master Confluence Doesn't Need Optimization

MC reads pre-optimized signals from the sub-indicators. When a sub-indicator auto-selects better parameters for SPY on a 5-min chart, MC automatically receives improved signals. The scoring weights themselves are the initial design — weight optimization is a future Phase 4+ task after collecting enough live signal data.

---

## Part 10: Trading with Master Confluence — Workflow

### Pre-Market Setup (Once)

1. Open TradingView with your preferred chart layout
2. Verify all 9 sub-indicators + Master Confluence are loaded
3. Check that MC's score table is showing real values (not all zeros)
4. Confirm webhook alerts are active

### During Market Hours

1. **Watch the MC score table** — it updates every bar
2. **When an A/A+ signal fires**:
   - The chart shows a signal label with entry/SL/TP
   - The webhook fires to your AWS pipeline
   - The score table highlights the grade
3. **Execute the trade plan**:
   - Enter at the displayed entry price
   - Set stop-loss at the displayed SL
   - Scale out at TP1, TP2, TP3 (⅓ each)
4. **Use the component table** to understand WHY the signal fired — if one component gave a strong signal while most are neutral, be cautious

### What NOT to Do

- **Don't override the system** — if MC says B+, don't take the trade because "it looks good"
- **Don't ignore the stop-loss** — the SL is structure-based and calculated, not arbitrary
- **Don't trade B+ signals** — they exist to show you the market is leaning but not committed
- **Don't watch all 9 sub-indicators individually** — MC aggregates everything for you
- **Don't change indicator parameters** — they auto-optimize per ticker/timeframe

### Position Sizing Reminders

From the YELENA risk management rules:
- **Max 10%** of account per trade
- **Max 3 concurrent positions**
- **$500 daily loss limit** — if hit, stop trading for the day
- **Target ≥65% win rate** over rolling 30 days

---

## Part 11: Visual Settings Reference

All adjustable in MC settings under 🎨 Visuals:

| Setting | Default | Options | Purpose |
|---------|---------|---------|---------|
| Table Position | Top Right | Top Left/Right, Bottom Left/Right, Middle Right | Where the score table appears |
| Show Component Detail Table | On | On/Off | Toggle the per-indicator breakdown |
| Show SL/TP Lines on Signal | On | On/Off | Toggle entry/SL/TP horizontal lines |
| Show Signal Labels | On | On/Off | Toggle CALL/PUT labels on chart |
| Label Size | 2 (medium) | 1=tiny, 2=medium, 3=large | Signal label size |

---

## Part 12: Troubleshooting

### All component scores show 0.0
**Cause**: Sources aren't wired. MC is reading `close` for everything instead of the actual hidden plots.
**Fix**: Open MC settings and wire each input.source() to the correct sub-indicator hidden plot (see Part 2).

### Score table shows values but no signals fire
**Cause**: The score hasn't reached A-grade threshold (±6.0). This is normal — the market isn't always in a high-conviction state.
**Fix**: Be patient. A-grade signals are rare by design.

### Webhook alert doesn't fire
**Cause**: Alert might not be set up, or the condition doesn't match.
**Fix**: In TradingView, verify the alert exists, uses the correct condition ("Any A-grade signal"), and the webhook URL is correct.

### Indicator shows compilation error
**Cause**: TradingView Pine Script version mismatch or syntax issue.
**Fix**: Ensure you're using the exact code from the repository. Check for Pine v6 compatibility.

### Sub-indicator shows "NaN" or blank values
**Cause**: Ticker or timeframe not in the lookup table, and fallback produced invalid data.
**Fix**: Stick to supported tickers (SPY, QQQ, TSLA, NVDA, AMD, MSFT, META, NFLX, AAPL, AMZN, GOOG) and supported timeframes (1, 5, 15, 60 min).

### Chart loads slowly with all indicators
**Cause**: 10 indicators (9 sub + MC) is a lot for TradingView to process.
**Fix**: This is normal for the initial load. Once loaded, real-time updates are fast. Consider a TradingView Pro+ plan for better performance.

---

## Part 13: Understanding What Each Indicator Measures

A deeper look at the "why" behind each component:

### QCloud — Trend Structure
Five moving averages form a cloud. When all 5 are bullish-ordered (fastest on top), trend is strong. When they compress (squeeze), a big move is coming. QCloud tells MC: **"Is there a clear trend, and how strong is it?"**

### QLine — Dynamic S/R
A SuperTrend-based line that flips between support and resistance. Bounces off QLine are high-probability re-entries. QLine tells MC: **"Is price respecting a trend structure, and is it bouncing or breaking?"**

### QWave — Trend Strength
ADX + DI lines measure whether a trend has energy. High ADX = strong trend, low ADX = choppy market. QWave tells MC: **"Does this trend have enough momentum to be worth trading?"**

### QBands — Volatility & Mean Reversion
Bollinger + Keltner channel combination detects squeezes (low volatility → imminent breakout) and extreme band positions (mean reversion opportunities). QBands tells MC: **"Is volatility compressed or extended, and which direction is it firing?"**

### Moneyball — Momentum Oscillator
Smoothed Rate of Change, normalized to -100/+100. Tracks raw price momentum with zone classification. Moneyball tells MC: **"Is price momentum accelerating or decelerating, and how extreme is it?"**

### QMomentum — Overbought/Oversold + Divergence
RSI + Stochastic combo with divergence detection. Catches reversals when price makes new highs/lows but momentum doesn't. QMomentum tells MC: **"Is momentum overextended, and are there hidden divergence warnings?"**

### QCVD — Volume Flow Analysis
Cumulative Volume Delta tracks whether buyers or sellers are dominant. Spikes reveal institutional activity. QCVD tells MC: **"Is money flowing in or out, and are institutions making moves?"**

### QSMC — Smart Money Concepts
Detects Break of Structure (trend continuation), Change of Character (trend reversal), and Order Blocks (institutional entry zones). QSMC tells MC: **"What is the market structure doing, and where are the smart money levels?"**

### QGrid — Price Levels Framework
Pivot-based support/resistance levels + VWAP. Tells you where price is relative to key levels. QGrid tells MC: **"Where are the important price levels, and is price positioned favorably?"**

---

## Quick Reference Card

```
╔══════════════════════════════════════════════════╗
║        YELENA v2: MASTER CONFLUENCE              ║
║              QUICK REFERENCE                     ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  GRADES:                                         ║
║    A+  (±8 to ±10)  → TRADE IT ✅                ║
║    A   (±6 to ±7.9) → TRADE IT ✅                ║
║    B+  (±4 to ±5.9) → SKIP ❌                    ║
║    B   (±0 to ±3.9) → SKIP ❌                    ║
║                                                  ║
║  EXITS:                                          ║
║    TP1 = 1:1 R:R → take ⅓                       ║
║    TP2 = 2:1 R:R → take ⅓                       ║
║    TP3 = 3:1 R:R → let ⅓ run                    ║
║                                                  ║
║  RISK RULES:                                     ║
║    Max 10% per trade                             ║
║    Max 3 positions                               ║
║    $500 daily loss limit                         ║
║                                                  ║
║  REMEMBER:                                       ║
║    • Only trade A/A+ grades                      ║
║    • Never override the stop-loss                ║
║    • Let the system do the analysis              ║
║    • Check component table for context           ║
║                                                  ║
╚══════════════════════════════════════════════════╝
```

---

*End of Master Confluence Education Guide v1.0 — February 16, 2026*
