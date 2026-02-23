# QWave — Bull/Bear Momentum System
## Trading Education Guide (YELENA)

---

## What is QWave?

QWave is an **ADX-based directional movement oscillator** that measures whether bulls or bears are in control and how strongly. It outputs a continuous score from -100 (maximum bearish momentum) to +100 (maximum bullish momentum).

**What it answers**: "Is this move real or fake, and how strong is it?"

Unlike simple oscillators that just show overbought/oversold, QWave uses an **ADX-weighted power curve** that actively suppresses signals during choppy/ranging markets and amplifies them during real trending moves. This is critical for 0DTE scalping where entering during a fake move is the #1 cause of losses.

---

## How It Works

### The Three Raw Components

QWave is built on the DMI (Directional Movement Index) system, which has three parts:

| Component | What It Measures | Range |
|-----------|-----------------|-------|
| **DI+** (Plus Directional Indicator) | Buying pressure strength | 0-100 |
| **DI-** (Minus Directional Indicator) | Selling pressure strength | 0-100 |
| **ADX** (Average Directional Index) | Trend strength regardless of direction | 0-100 |

- When DI+ > DI- → Buyers are winning
- When DI- > DI+ → Sellers are winning
- ADX tells you HOW MUCH the winners are winning by

### The QWave Formula

```
Step 1: Direction = (DI+ - DI-) / (DI+ + DI-)     → Who's winning? (-1 to +1)
Step 2: ADX Norm  = min(ADX / 50, 1.0)             → How strong? (0 to 1)
Step 3: Amplifier = ADX Norm ^ 1.5                  → Power curve filter
Step 4: QWave     = Direction × Amplifier × 100     → Final score (-100 to +100)
Step 5: Smoothed  = EMA(QWave, smoothing)            → Optional noise reduction
```

### The Power Curve — Why It Matters

The `^ 1.5` exponent is what makes QWave special. It creates an exponential suppression curve:

| ADX Value | Trend State | Signal Passed Through | What It Means |
|-----------|-------------|----------------------|---------------|
| 10 | No trend | **9%** | Nearly zeroed out — "nothing is happening" |
| 15 | Barely trending | **16%** | Heavily suppressed — "maybe something, probably not" |
| 20 | Trend emerging | **25%** | Starting to register — "something might be starting" |
| 25 | Moderate trend | **35%** | Meaningful signal — "there's a real move here" |
| 30 | Good trend | **47%** | Strong signal — "confirmed trend, trade it" |
| 40 | Strong trend | **72%** | High conviction — "strong move in progress" |
| 50+ | Very strong | **100%** | Full signal — "maximum trend strength" |

**Real-world example**: If DI+ is dominating DI- but ADX is only 12, QWave stays near zero. This tells you "bulls might be winning but there's no real trend — skip this." But if ADX hits 35 with the same DI spread, QWave produces a strong +50 reading: "this bullish move is real, take the trade."

This single feature prevents the majority of false entries that occur during sideways, choppy markets.

---

## Auto-Optimization

QWave automatically detects the current symbol and timeframe, then selects optimal ADX length and smoothing from a built-in lookup table.

### What Gets Optimized

| Parameter | What It Controls | Impact |
|-----------|-----------------|--------|
| **ADX Length** | How far back to look for directional movement | Shorter = faster reaction, more noise. Longer = smoother, slower. |
| **Smoothing** | EMA applied to final QWave score | 1 = raw (fastest), 3-5 = smoother (fewer false flips) |

### Volatility Groups

| Group | Symbols | ADX Length | Why |
|-------|---------|-----------|-----|
| **High Volatility** | TSLA, NVDA | Shorter (10-14) | Fast-moving stocks need quicker reaction |
| **Index ETFs** | SPY, QQQ | Standard (12-16) | More predictable, standard lookback works |
| **Mega-caps** | AAPL, GOOGL, MSFT, AMZN, AMD, META, NFLX | Standard (12-16) | Balanced behavior |

### Timeframe Groups

| Group | Timeframes | Smoothing | Why |
|-------|-----------|-----------|-----|
| **Scalp** | ≤1 minute | Lower (1-2) | Need fast zero-line crossings |
| **Intraday** | 5-15 minutes | Medium (2-3) | Primary trading timeframe |
| **Swing** | 1 hour+ | Higher (3-5) | Filter out intraday noise |

---

## The 6-Zone System

QWave divides its -100 to +100 range into 6 zones:

| Zone | Score Range | Color | Meaning | Trading Action |
|------|------------|-------|---------|----------------|
| **STRONG BULL** | +60 to +100 | 🟢 Bright Green | Aggressive buying, powerful uptrend | Maximum conviction CALL |
| **BULL** | +30 to +60 | 🟢 Green | Buyers clearly in control | Favor CALL |
| **WEAK BULL** | 0 to +30 | 🟢 Dark Green | Slight buyer edge, indecisive | Cautious — reduce size or wait |
| **WEAK BEAR** | -30 to 0 | 🔴 Dark Red | Slight seller edge, indecisive | Cautious — reduce size or wait |
| **BEAR** | -60 to -30 | 🔴 Red | Sellers clearly in control | Favor PUT |
| **STRONG BEAR** | -100 to -60 | 🔴 Bright Red | Aggressive selling, powerful downtrend | Maximum conviction PUT |

### Key Zone Insights

- **±30 zones (Bull/Bear)** are the "confirmation" zones — once QWave crosses ±30, the move is real
- **±60 zones (Strong Bull/Bear)** are the "extreme" zones — maximum momentum, but also watch for exhaustion
- **±0-30 zones (Weak Bull/Bear)** are the "noise" zones — ADX power curve has already suppressed most false signals, but signals here are still lower probability
- **Zone thresholds (±30, ±60) are fixed** across all symbols and timeframes for consistency

---

## Signal Types

### 1. Zero-Line Flips (Primary Signal)

The most important signal QWave produces. When the score crosses from negative to positive (or vice versa), it means the balance of power has shifted.

| Flip | Meaning | Action |
|------|---------|--------|
| **⚡ BULL** (crosses above 0) | Bulls just took control from bears | Prepare for CALL entries |
| **⚡ BEAR** (crosses below 0) | Bears just took control from bulls | Prepare for PUT entries |

**Why flips matter**: Because the ADX power curve suppresses weak moves, a QWave zero-line flip means the directional shift has REAL trend strength behind it — not just random noise.

### 2. Zone Crossings (Strength Confirmation)

| Crossing | Meaning | Significance |
|----------|---------|-------------|
| Weak Bull → **BULL** (crosses +30) | Move strengthening | Confirms the trend is real |
| Bull → **STRONG BULL** (crosses +60) | Maximum momentum | Highest conviction, but watch for exhaustion |
| Weak Bear → **BEAR** (crosses -30) | Selling strengthening | Confirms the downtrend is real |
| Bear → **STRONG BEAR** (crosses -60) | Maximum selling | Highest conviction PUT, but watch for exhaustion |

**Strengthening pattern**: Weak Bull → Bull → Strong Bull (momentum building — stay in trade)
**Weakening pattern**: Strong Bull → Bull → Weak Bull (momentum fading — tighten stops)

### 3. ADX Threshold Events

| Event | Meaning | Action |
|-------|---------|--------|
| **ADX crosses above 20** (ADX↑ label) | Trend emerging from consolidation | Get ready — a directional move is starting |
| **ADX crosses below 20** | Trend dying into consolidation | Stop trading — chop returning |

**Key insight**: ADX rising above 20 combined with a QWave zone entry (+30 or -30) is one of the strongest setups — it means a new trend is both starting AND already has directional conviction.

### 4. Divergence Detection

| Divergence | What Happens | What It Means |
|-----------|-------------|---------------|
| **DIV ▼ (Bearish)** | Price makes higher high, QWave makes lower high | Buying momentum fading despite price rising — potential top |
| **DIV ▲ (Bullish)** | Price makes lower low, QWave makes higher low | Selling momentum fading despite price falling — potential bottom |

**Important**: Divergence is a WARNING, not a trade signal. It means momentum is diverging from price, which often precedes a reversal — but timing the exact reversal requires confirmation from other indicators.

### 5. QWave Slope

| Slope State | Meaning | Action |
|------------|---------|--------|
| **ACCEL** | QWave moving further from zero | Momentum building — stay in trade |
| **FLAT** | QWave stable | Trend pausing — watch for continuation or reversal |
| **DECEL** | QWave moving toward zero | Momentum fading — tighten stops |

---

## Trading Rules

### Rule 1: Only Trade in ±30+ Zones
- **QWave > +30 (Bull zone)**: Only CALL positions
- **QWave < -30 (Bear zone)**: Only PUT positions
- **QWave between -30 and +30**: No trade — the signal isn't strong enough

### Rule 2: Zero-Line Flips Are Entry Signals
When QWave crosses zero, this IS a directional signal. The best entries are:
1. Zero-line flip occurs (⚡ BULL or ⚡ BEAR label appears)
2. Wait for first pullback (QWave dips toward zero without crossing back)
3. Enter in the flip direction
4. Stop if QWave flips back (crosses zero the other way)

### Rule 3: Respect the Power Curve
If QWave is near zero even though price is moving, it means ADX is low — there's no real trend. Do NOT override this by switching to manual mode with different settings. The power curve is protecting you from fake moves.

### Rule 4: Divergence = Warning, Not Entry
When you see DIV ▲ or DIV ▼:
- Do NOT immediately trade the reversal
- Tighten stops on existing positions in the divergence direction
- Wait for QWave to actually flip zones before entering the other way

### Rule 5: Watch for Exhaustion in ±60 Zones
Strong Bull (>+60) and Strong Bear (<-60) are powerful, but trends don't stay there forever:
- Entering positions when QWave is ALREADY in the ±60 zone can be late
- Best entries are when QWave ENTERS the ±60 zone (🔥 S.BULL or 🔥 S.BEAR label)
- If QWave starts decelerating from ±60 back toward ±30, the trend is exhausting

---

## Entry Setups

### Setup 1: Zero-Line Flip + Zone Confirmation (Primary — Highest Win Rate)
1. QWave crosses zero (⚡ BULL or ⚡ BEAR flip)
2. Wait for QWave to reach ±30 (enters Bull or Bear zone)
3. Enter in the trend direction
4. Stop: QWave crosses back below/above zero

### Setup 2: ADX Emerging + Zone Entry (Breakout Setup)
1. ADX crosses above 20 (ADX↑ label appears)
2. QWave enters Bull (+30) or Bear (-30) zone
3. This means a NEW trend is forming with real strength
4. Enter in the zone direction
5. Stop: QWave returns to Weak zone (between -30 and +30)

### Setup 3: Strong Zone Entry (Momentum Setup)
1. QWave crosses +60 (🔥 S.BULL) or -60 (🔥 S.BEAR)
2. Maximum momentum is present
3. Enter in the trend direction with tight stop
4. Take profit quickly — extreme momentum doesn't last

### Setup 4: Divergence Reversal (Counter-Trend — Advanced)
1. Divergence detected (DIV ▲ or DIV ▼)
2. Wait for QWave to flip zones (crosses zero OR crosses ±30)
3. Enter in the NEW direction after confirmation
4. This is counter-trend trading — use smaller position size

---

## Combining with Other Indicators

QWave tells you HOW STRONG the move is. Other indicators tell you direction (QCloud), support/resistance (QLine), and timing.

| Combo | How It Works | Signal |
|-------|-------------|--------|
| **QCloud + QWave** | QCloud confirms direction, QWave confirms strength | QCloud 4-5 bull + QWave >+30 = strong CALL |
| **QLine + QWave** | QLine provides entry point, QWave confirms momentum | QLine bounce + QWave accelerating = confirmed entry |
| **QCloud + QLine + QWave** | Triple confluence | QCloud green + QLine bounce + QWave >+30 = highest conviction |
| **QWave + Moneyball** | QWave measures trend strength, Moneyball measures momentum shift | Both flipping positive = strong continuation signal |
| **QWave divergence + QLine flip** | Divergence warns, QLine flip confirms | QWave divergence followed by QLine flip = high-probability reversal |

### Example Trade Setup (Triple Confluence)
1. QCloud shows 4-5 layers bullish (green cloud) — DIRECTION confirmed
2. QWave shows +35, Bull zone, accelerating — STRENGTH confirmed
3. Price pulls back to QLine support and bounces (★★★) — ENTRY confirmed
4. Enter CALL with stop below QLine
5. Take profit at QWave deceleration or extension warning

---

## Info Table Reference

The info table on your chart shows all real-time QWave metrics:

| Row | Label | What It Shows |
|-----|-------|--------------|
| 1 | **QWave v2** | Indicator name |
| 2 | **Score** | Current QWave score (-100 to +100) in zone color |
| 3 | **Zone** | Current zone name (STRONG BULL → STRONG BEAR) |
| 4 | **ADX** | Raw ADX value + strength state (WEAK/EMERGING/MODERATE/STRONG) |
| 5 | **DI+/DI-** | Raw DI+ and DI- values — green if DI+ winning, red if DI- winning |
| 6 | **Slope** | QWave slope state (ACCEL / FLAT / DECEL) |
| 7 | **In Zone** | Bars spent in current zone |
| 8 | **Settings** | Current mode (Auto = green, Manual = orange) |
| 9 | **Params** | Active parameters: ADX length and smoothing |

---

## Understanding ADX States

The ADX value tells you about trend STRENGTH regardless of direction:

| ADX State | ADX Value | Meaning | QWave Behavior |
|-----------|----------|---------|---------------|
| **WEAK** | Below 20 | No real trend — ranging/choppy market | QWave suppressed near zero |
| **EMERGING** | 20-25 | Trend starting to form | QWave beginning to show direction |
| **MODERATE** | 25-40 | Solid trend in progress | QWave producing reliable signals |
| **STRONG** | Above 40 | Powerful trend | QWave at high conviction levels |

**Key insight**: You want to trade when ADX is MODERATE or STRONG (25+). When ADX is WEAK (<20), even if QWave shows a direction, the signal is unreliable because the power curve has suppressed it.

---

## Webhook Payload

When alerts fire, QWave sends this JSON:

```json
{
  "passphrase": "your_secret",
  "ticker": "SPY",
  "timeframe": "5",
  "indicator": "qwave",
  "alert_type": "flip_bullish",
  "qwave_score": 47.3,
  "zone": "BULL",
  "zone_num": 5,
  "adx": 32.1,
  "di_plus": 28.4,
  "di_minus": 15.2,
  "slope_state": "ACCEL",
  "bars_in_zone": 3,
  "last_flip": "bullish",
  "bars_since_flip": 8,
  "settings_mode": "Auto",
  "adx_length": 14,
  "smoothing": 2,
  "price": 685.30,
  "timestamp": "2026-02-13T10:30:00Z"
}
```

### Alert Types

| Alert Type | Trigger |
|-----------|---------|
| `flip_bullish` | QWave crossed above zero |
| `flip_bearish` | QWave crossed below zero |
| `zone_bull` | QWave entered Bull zone (+30) |
| `zone_bear` | QWave entered Bear zone (-30) |
| `strong_bull` | QWave entered Strong Bull zone (+60) |
| `strong_bear` | QWave entered Strong Bear zone (-60) |
| `trend_emerging` | ADX crossed above 20 |
| `divergence_bullish` | Price lower low, QWave higher low |
| `divergence_bearish` | Price higher high, QWave lower high |

---

## Common Mistakes

1. **Trading in the ±0-30 "noise" zones** — QWave is near zero for a reason; the power curve is telling you there's no real trend
2. **Ignoring ADX state** — A QWave reading of +25 with ADX at 35 is completely different from +25 with ADX at 12
3. **Trading divergence immediately** — Divergence is a warning, not an entry signal; wait for the actual flip
4. **Chasing Strong Bull/Bear entries** — Entering when QWave is already at +70 means the best of the move may be over
5. **Overriding the power curve** — If QWave is suppressed near zero, don't switch to Manual mode to get stronger readings; the suppression is protecting you
6. **Using QWave alone** — QWave tells you strength, not entry timing; combine with QLine for entries and QCloud for direction
7. **Confusing ADX with direction** — ADX of 40 means "strong trend" but doesn't tell you up or down; that's what QWave's sign (+ or -) tells you

---

## Hidden Plot Exports (For Master Confluence)

| Export | Value | Range |
|--------|-------|-------|
| X_QWaveScore | QWave score | -100 to +100 |
| X_QWaveZone | Zone number | 1 (Strong Bear) to 6 (Strong Bull) |
| X_QWaveADX | Raw ADX value | 0-100 |
| X_QWaveSlope | Rate of change | Varies |
| X_QWaveDIPlus | Raw DI+ value | 0-100 |
| X_QWaveDIMinus | Raw DI- value | 0-100 |
| X_QWaveFlip | Flip signal | -1 (bearish), 0 (none), 1 (bullish) |
| X_QWaveDivergence | Divergence signal | -1 (bearish), 0 (none), 1 (bullish) |

---

## Summary

| Signal | Meaning | Action |
|--------|---------|--------|
| ⚡ BULL flip | Bulls took control | Prepare for CALL entries |
| ⚡ BEAR flip | Bears took control | Prepare for PUT entries |
| Entered BULL zone (+30) | Confirmed buying pressure | Active CALL setups |
| Entered BEAR zone (-30) | Confirmed selling pressure | Active PUT setups |
| 🔥 S.BULL (+60) | Maximum bullish momentum | Highest conviction CALL |
| 🔥 S.BEAR (-60) | Maximum bearish momentum | Highest conviction PUT |
| ADX↑ (crosses 20) | Trend emerging from consolidation | Get ready for a move |
| DIV ▲ | Bullish divergence | Warning — selling may be exhausting |
| DIV ▼ | Bearish divergence | Warning — buying may be exhausting |
| ACCEL slope | Momentum building | Stay in trade |
| DECEL slope | Momentum fading | Tighten stops |
| FLAT slope | Trend pausing | Watch for continuation or reversal |
| QWave near 0 | No trend (ADX suppression) | Do NOT trade — wait for signal |