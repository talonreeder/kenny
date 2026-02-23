# Moneyball v2 — Education Guide

## What Moneyball Does

Moneyball answers one question: **which way is momentum pointing, and how strong is it?**

It outputs a number from -100 (maximum bearish momentum) to +100 (maximum bullish momentum). Zero is the dividing line — above zero means buyers are in control, below zero means sellers are.

What makes Moneyball different from raw price movement is **volume weighting**. When a move happens on high volume, Moneyball amplifies the signal. When a move happens on low volume, it dampens it. This helps you distinguish between real institutional moves and noise.

---

## How It Works (Simple Version)

1. **Measure how fast price is changing** (Rate of Change)
2. **Amplify or dampen based on volume** — high volume = real conviction, low volume = noise
3. **Smooth it out** so it's not jerky
4. **Normalize to -100/+100** so readings are comparable across stocks and timeframes

The result is a smooth oscillator that leads price. When Moneyball flips from negative to positive, it often signals a trend change before the candles make it obvious.

---

## The Volume Weight

This is the key v2 upgrade. Consider two scenarios:

**Scenario A**: Price rises 0.5% on normal volume → Moneyball reads +40
**Scenario B**: Price rises 0.5% on 2x average volume → Moneyball reads +80

Same price move, but Scenario B has institutional conviction behind it. The volume weight (shown in the info table as "Vol Wt") tells you how much amplification is happening:

| Vol Wt | Meaning |
|--------|---------|
| 0.5x | Very low volume — signal dampened by half |
| 0.7-0.8x | Below average volume |
| 1.0x | Normal volume — no amplification |
| 1.3-1.5x | Above average volume — signal amplified |
| 2.0x | Maximum amplification (volume 2x+ above average) |

When you see a flip or zone entry with Vol Wt above 1.3x, that's high conviction.

---

## Zone System

| Zone | Range | What It Means |
|------|-------|---------------|
| **Strong Bear** | < -70 | Extreme selling momentum. Potential for oversold bounce, but don't buy yet. |
| **Bear** | -70 to -30 | Clear bearish momentum. Favor short setups. |
| **Neutral** | -30 to +30 | No strong directional bias. Wait for a breakout from this zone. |
| **Bull** | +30 to +70 | Clear bullish momentum. Favor long setups. |
| **Strong Bull** | > +70 | Extreme buying momentum. Strong trend, but watch for exhaustion. |

The zero line is the most important level. Everything above it = bullish bias, everything below = bearish bias.

---

## Key Signals

### 1. Zero-Line Flips (Most Important)
When Moneyball crosses zero, momentum direction has changed. This is your primary signal.

- **Bull Flip** (crosses above 0): Momentum shifted from sellers to buyers
- **Bear Flip** (crosses below 0): Momentum shifted from buyers to sellers

Flips are leading indicators — they often occur 2-5 bars before price confirms the direction change.

### 2. Explosive Moves (⚡)
Sudden large jumps in the oscillator, more than 3x the average bar-to-bar change. These are unusual and significant — they indicate a sudden surge of directional momentum, often from institutional activity.

An explosive move that pushes Moneyball from neutral into a directional zone is a very strong signal.

### 3. Divergences
When price and momentum disagree, something is about to change:

- **Bearish divergence**: Price makes a higher high, but Moneyball makes a lower high. The uptrend is losing steam. Prepare for a reversal down.
- **Bullish divergence**: Price makes a lower low, but Moneyball makes a higher low. The downtrend is losing steam. Prepare for a reversal up.

Divergences are early warnings — they don't trigger immediately but set up the reversal within 5-20 bars.

### 4. Slope (Acceleration)
The slope tells you whether momentum is getting stronger or weaker:

- **ACCEL**: Momentum is increasing — the current move is gaining strength
- **FLAT**: Momentum is steady — neither strengthening nor weakening
- **DECEL**: Momentum is fading — the current move may be exhausting

DECEL in a strong zone often precedes a zone exit. ACCEL entering a zone confirms the move is real.

---

## Trading Rules

### Rule 1: Trade in the Direction of Moneyball
- Moneyball > 0 → only take long setups
- Moneyball < 0 → only take short setups
- Moneyball in neutral (-30 to +30) → be cautious, wait for direction

### Rule 2: Flips Are Entry Signals, Not Exit Signals
A bull flip is an entry for longs, not an exit for shorts (your short should have already been stopped out). Similarly for bear flips.

### Rule 3: Volume Weight Confirms Quality
- Flip with Vol Wt > 1.3x → high conviction, take the trade
- Flip with Vol Wt < 0.8x → low conviction, wait for confirmation from other indicators

### Rule 4: Respect Extreme Zones
- Entering Strong Bull/Bear zone → momentum is powerful, ride it
- But DECEL in a strong zone → momentum fading, tighten stops
- Don't fade a strong zone (don't short at +80 just because it's "overbought")

### Rule 5: Divergences Need Confirmation
A divergence alone is not a trade signal. Wait for:
- The actual flip (divergence predicts it, flip confirms it)
- Or confirmation from QCloud direction change or QLine flip

---

## Entry Setups

### Setup 1: Volume-Confirmed Flip
**Trigger:** Zero-line flip (bull or bear)
**Filter:** Vol Wt > 1.2x at the flip bar
**Confirm with:** QCloud direction matches, QWave zone matches
**Stop:** Recent swing high/low
**Target:** Ride until slope goes DECEL in a strong zone

### Setup 2: Explosive Entry
**Trigger:** Explosive move (⚡) pushes Moneyball from neutral into Bull/Bear zone
**Filter:** Vol Wt > 1.0x
**Confirm with:** QBands squeeze fire in same direction (if active)
**Stop:** Tight — below the explosive bar's low (for longs)
**Target:** Trail with QLine

### Setup 3: Divergence Reversal
**Trigger:** Bullish or bearish divergence detected
**Wait for:** Zero-line flip to confirm the reversal
**Confirm with:** QWave momentum shift, QCloud color change
**Stop:** Beyond the divergence pivot point
**Target:** First target at basis (QBands), extended target at opposite band

### Setup 4: Zone Acceleration
**Trigger:** Moneyball enters Bull/Bear zone AND slope = ACCEL
**Filter:** Already in a trade from a flip — this is an ADD signal
**Confirm with:** QCloud green/red in same direction
**Stop:** Move stop to breakeven
**Target:** Hold until slope goes DECEL

---

## Combining with Other Indicators

### Moneyball + QCloud
- Moneyball flip + QCloud same color = strong trend confirmation
- Moneyball DECEL in strong zone + QCloud squeeze = potential reversal setup
- When they disagree (Moneyball bull, QCloud bear), Moneyball often leads by a few bars

### Moneyball + QWave
- Both are momentum indicators but measure differently (ROC vs ADX)
- When both are in bull zones simultaneously = very high conviction
- If one flips before the other, the first flip is the early warning

### Moneyball + QBands
- Moneyball explosive move + QBands squeeze fire = highest momentum signal
- Moneyball DECEL at QBands 2σ touch = strong mean reversion setup
- Vol Wt > 1.5x during a squeeze fire = institutional breakout

### Moneyball + QLine
- Moneyball flip + QLine flip in same bar = "double flip" — very rare, very strong
- Use QLine as your trailing stop when Moneyball confirms the direction

---

## Reading the Info Table

| Row | What to Look For |
|-----|-----------------|
| **Score** | Current -100 to +100. Watch for zero crossings. |
| **Zone** | Which zone you're in. Only trade in directional zones (Bull/Bear or stronger). |
| **Slope** | ACCEL = momentum growing. DECEL = fading. FLAT = steady. |
| **Vol Wt** | Above 1.3x = high volume conviction. Below 0.7x = low conviction. |
| **Bars** | How long in current zone. Short stays (<5) in a zone = unstable. Long stays (>15) = established. |
| **Params** | ROC length and smoothing being used. Verify Auto mode is active. |

---

## Common Mistakes

### Mistake 1: Fading Strong Momentum
Moneyball at +80 doesn't mean "overbought, time to short." Strong momentum tends to persist. Only look for reversals when you see DECEL slope or divergence.

### Mistake 2: Trading Every Flip
Not all flips lead to sustained moves. Low-volume flips (Vol Wt < 0.8x) in choppy markets often whipsaw. Wait for volume confirmation.

### Mistake 3: Ignoring the Neutral Zone
When Moneyball is between -30 and +30, there's no clear momentum. This is a waiting zone. Don't force trades — wait for a decisive move out of neutral.

### Mistake 4: Using Divergences as Instant Signals
Divergences are early warnings, not entries. A divergence says "the trend is weakening" — not "reverse right now." Always wait for the actual flip or another confirmation signal.

### Mistake 5: Not Checking Volume Weight
A flip at 0.5x volume weight is very different from a flip at 1.8x. Always check Vol Wt in the info table before acting on any Moneyball signal. The volume weight is what separates Moneyball from a generic momentum oscillator.
