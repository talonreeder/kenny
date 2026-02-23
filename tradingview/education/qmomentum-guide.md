# QMomentum v2 — Education Guide

## What QMomentum Does

QMomentum answers: **is the current move running out of steam?**

It tracks RSI (Relative Strength Index) on a 0-100 scale and automatically detects divergences — situations where price and momentum disagree. When they disagree, something is about to change.

It also overlays Stochastic RSI for an extra layer of sensitivity, catching turns that raw RSI misses.

---

## RSI Basics

RSI measures how much of recent price movement was upward vs downward:

- **RSI 70+**: Overbought — lots of recent buying, potentially exhausted
- **RSI 50**: Balanced — no strong bias
- **RSI 30-**: Oversold — lots of recent selling, potentially exhausted

Common misconception: overbought doesn't mean "sell now." In a strong uptrend, RSI can stay above 70 for extended periods. The signal isn't the level itself — it's the behavior AT the level.

---

## Stochastic RSI

StochRSI applies the Stochastic formula to RSI itself, creating an even more sensitive oscillator. Think of it as "how oversold is the RSI compared to its own recent range?"

The K line (fast, blue) and D line (slow, orange) create crossover signals:
- **K crosses above D in oversold zone** → strong buy signal
- **K crosses below D in overbought zone** → strong sell signal

StochRSI turns faster than raw RSI, so it catches reversals earlier but is also noisier. That's why we use it as confirmation, not as the primary signal.

---

## Divergences — The Core Feature

Divergences are the most powerful signal QMomentum produces. They detect when momentum is secretly failing.

### Regular Divergences (Reversal Signals)

**Bearish Divergence** (DIV ▼):
- Price makes a **higher high** (looks bullish on the chart)
- RSI makes a **lower high** (momentum is actually weakening)
- Translation: "Price is going up, but with less and less conviction. Reversal coming."

**Bullish Divergence** (DIV ▲):
- Price makes a **lower low** (looks bearish on the chart)
- RSI makes a **higher low** (selling momentum is actually weakening)
- Translation: "Price is going down, but sellers are losing steam. Bounce coming."

### Hidden Divergences (Continuation Signals)

**Hidden Bearish** (H.DIV ▼):
- Price makes a lower high, RSI makes a higher high
- Translation: "The downtrend is still in control, this rally is a trap."

**Hidden Bullish** (H.DIV ▲):
- Price makes a higher low, RSI makes a lower low
- Translation: "The uptrend is still in control, this dip is a buying opportunity."

### Quality Scoring

Not all divergences are created equal:

| Score | Criteria | Reliability |
|-------|----------|------------|
| **★★★** | Pivots >15 bars apart, RSI gap >5 points, AND in OB/OS zone | Highest — trade with confidence |
| **★★** | Pivots >10 bars apart, RSI gap >3 points | Good — trade with confirmation |
| **★** | Basic divergence (minimum criteria) | Lowest — wait for additional confirmation |

The wider the divergence (more bars between pivots, larger RSI gap), the more significant the signal.

---

## OB/OS Duration Matters

How long RSI stays in overbought/oversold changes the interpretation:

| Duration | Meaning |
|----------|---------|
| **1-3 bars** | Quick dip into OB/OS — likely a quick reversal |
| **4-10 bars** | Moderate stay — standard OB/OS signal |
| **10+ bars** | Extended stay — strong trend, don't fade it |

This is why QMomentum tracks "OB/OS Bars" in the info table. When you see OB with 15+ bars, the trend is powerful and shorting is dangerous. Wait for the exit from the zone before acting.

---

## Zone Exit Signals

The most actionable RSI signals aren't entering OB/OS — they're **leaving**:

- **Left Overbought** (◀ OB): RSI dropped back below 70. The buying pressure has eased. This is your short entry zone.
- **Left Oversold** (◀ OS): RSI rose back above 30. The selling pressure has eased. This is your long entry zone.

Zone exits are more reliable than zone entries because they confirm the momentum shift has begun, not just that it might happen.

---

## Trading Rules

### Rule 1: Don't Fade Strong Trends Based on OB/OS Alone
RSI at 75 in a strong uptrend is normal, not a sell signal. Only act on OB/OS when combined with divergence, zone exit, or other indicator confirmation.

### Rule 2: Divergences Are Early Warnings, Not Entries
A divergence says "the trend is weakening." Wait for the zone exit or a Moneyball/QWave confirmation before entering.

### Rule 3: Quality Matters for Divergences
★★★ divergences can be traded with less confirmation. ★ divergences need multiple confirmations from other indicators.

### Rule 4: StochRSI Crosses in Extreme Zones Are Strong
A StochRSI bullish cross while RSI is below 30 is one of the strongest oversold bounce signals. Similarly for bearish crosses above 70.

### Rule 5: Duration Determines Strategy
Short OB/OS stays → trade the reversal. Long OB/OS stays → wait for the exit, then trade the mean reversion.

---

## Entry Setups

### Setup 1: Divergence + Zone Exit
**Trigger:** ★★+ divergence detected
**Wait for:** RSI exits OB (for bearish div) or OS (for bullish div)
**Confirm with:** Moneyball flip in same direction
**Stop:** Beyond the divergence pivot high/low
**Target:** RSI 50 level (price equivalent), or QBands basis

### Setup 2: StochRSI Extreme Cross
**Trigger:** StochRSI K crosses D in oversold (RSI < 30) or overbought (RSI > 70)
**Confirm with:** QWave zone matches, QCloud color matches
**Stop:** Recent swing low/high
**Target:** First target at RSI 50, extended target at opposite OB/OS zone

### Setup 3: OB/OS Zone Exit
**Trigger:** RSI exits OB or OS zone (◀ label appears)
**Filter:** OB/OS duration was 3-10 bars (not a quick spike, not a prolonged trend)
**Confirm with:** QBands band touch in same direction
**Stop:** Back inside the zone (RSI returns to OB/OS)
**Target:** QBands basis (mean reversion)

### Setup 4: Hidden Divergence Continuation
**Trigger:** Hidden divergence detected during an established trend
**Confirm with:** QCloud confirms trend direction, QLine in same direction
**Stop:** Tight — beyond the hidden divergence pivot
**Target:** Trail with QLine, exit when regular divergence appears

---

## Combining with Other Indicators

### QMomentum + Moneyball
- Both are momentum oscillators but RSI is slower, Moneyball is faster
- Moneyball flip → then QMomentum zone exit = strong confirmation sequence
- Divergence on QMomentum + divergence on Moneyball simultaneously = very strong reversal signal

### QMomentum + QBands
- RSI in OB/OS + price at QBands 2σ/3σ = double overbought/oversold confirmation
- Divergence + QBands band touch reversal (★★★) = high probability mean reversion
- Zone exit + QBands squeeze fire in same direction = explosive move

### QMomentum + QWave
- QWave tells you trend strength (ADX), QMomentum tells you if momentum is failing
- Divergence on QMomentum while QWave is still in strong zone = early warning the strong zone won't hold
- Both in extreme zones simultaneously = maximum conviction

### QMomentum + QCloud
- RSI divergence + QCloud squeeze = the reversal will come when the cloud resolves
- RSI leaving OB/OS + QCloud color flip = trend change confirmed at both momentum and trend level

---

## Common Mistakes

### Mistake 1: Shorting at RSI 70 in a Trend
In strong trends, RSI lives above 50 and regularly visits 70-80. Don't short just because RSI is "overbought." Wait for divergence, zone exit, or other confirmation.

### Mistake 2: Ignoring Divergence Quality
A ★ divergence with 6-bar pivot separation and 2-point RSI gap is much weaker than a ★★★ with 20-bar separation and 8-point gap. Treat them very differently.

### Mistake 3: Using Divergences Alone
Divergences predict reversals but don't time them precisely. A divergence can form and then another higher high (or lower low) happens before the actual reversal. Always wait for confirmation.

### Mistake 4: Overtrading StochRSI Crosses
StochRSI crosses happen frequently. Only act on crosses that occur in OB/OS zones (RSI >70 or <30). Crosses in the neutral RSI zone are noise.

### Mistake 5: Ignoring Hidden Divergences
Most traders only watch for regular divergences (reversals). Hidden divergences are equally valuable — they confirm the existing trend is healthy and pullbacks are buying/selling opportunities.
