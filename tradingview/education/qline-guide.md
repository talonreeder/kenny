# QLine — Adaptive Trend Ribbon
## Trading Education Guide (YELENA)

---

## What is QLine?

QLine is a **dynamic trend-following line** based on SuperTrend that automatically adjusts to market volatility using ATR (Average True Range). It acts as:
- **Support** during uptrends (green line below price — price bounces off from above)
- **Resistance** during downtrends (red line above price — price bounces off from below)

When price breaks through QLine decisively, it signals a trend reversal.

**What it answers**: "What direction should I trade, and where are the best entries?"

---

## How It Works

### Built-in SuperTrend

QLine uses Pine Script's **built-in `ta.supertrend(factor, atr_length)`** which handles all the complex ratcheting math correctly. The key behavior:

1. In an **uptrend**: QLine sits below price and can only move UP (never down). This creates a rising support floor.
2. In a **downtrend**: QLine sits above price and can only move DOWN (never up). This creates a falling resistance ceiling.
3. **Flip**: When price closes beyond the line, the trend reverses and QLine jumps to the other side.

This "ratcheting" is what makes QLine so powerful — during normal pullbacks, the line holds steady. Only a genuine trend break causes a flip.

### Direction Convention
- **direction = -1** → Uptrend (green line below price)
- **direction = 1** → Downtrend (red line above price)

---

## Auto-Optimization

QLine automatically detects the current symbol and timeframe, then selects optimal ATR length and factor.

### Why Different Settings Matter

| Stock Type | ATR Length | Factor | Why |
|-----------|-----------|--------|-----|
| **High vol** (TSLA, NVDA) | Shorter (10-12) | Wider (2.2-3.0) | Wider band prevents constant whipsaw from big candles |
| **Index ETFs** (SPY, QQQ) | Standard (12-16) | Moderate (1.8-2.5) | More predictable, tighter bands work |
| **Mega-caps** (AAPL, MSFT) | Standard (12-15) | Moderate (1.9-2.6) | Balanced approach |

### Timeframe Groups

| Group | Factor Range | Why |
|-------|-------------|-----|
| **Scalp** (≤1min) | Lower (1.8-2.2) | Tighter for quick reaction |
| **Intraday** (5-15min) | Medium (2.0-2.5) | Primary trading timeframe |
| **Swing** (1hr+) | Higher (2.5-3.0) | Wider to filter noise |

---

## Enhanced Features

### 1. Trend Duration Tracking

QLine tracks how long the current trend has been active:

| Age | Duration | Meaning | Implication |
|-----|----------|---------|------------|
| **FRESH** | ≤5 bars | Brand new trend | Highest probability entries — momentum is strong |
| **DEVELOPING** | 6-20 bars | Trend establishing | Good entries, trend gaining confidence |
| **MATURE** | 21-50 bars | Well-established trend | Still tradeable but watch for exhaustion |
| **EXTENDED** | 51+ bars | Long-running trend | Higher reversal risk, tighten stops |

### 2. Touch Counting

Every time price comes close to QLine (within touch threshold), it counts as a "touch." Touches are separated by a minimum 3-bar gap to avoid double-counting.

| Touch Count | Quality | Meaning |
|-------------|---------|---------|
| 1-2 | **STRONG** | First touches — line is untested, high probability of holding |
| 3-4 | **NORMAL** | Line has been tested — still holding but watch closely |
| 5+ | **WEAKENING** | Multiple tests weaken the level — break becomes more likely |

### 3. Bounce Quality Scoring (★ to ★★★)

When a bounce occurs (price touches QLine and reverses), it's scored 1-3 stars:

| Factor | +1 Star If... |
|--------|---------------|
| **Candle body** | Body > 50% of total range (strong conviction candle) |
| **Trend freshness** | Trend is ≤20 bars old (FRESH or DEVELOPING) |
| **Touch strength** | ≤3 previous touches (line not yet weakened) |

**★★★ = Best bounce** — strong candle, fresh trend, untested line
**★★ = Good bounce** — 2 of 3 factors present
**★ = Weak bounce** — only 1 factor, lower probability

Only ★★+ bounces generate webhook alerts and show markers on chart.

### 4. QLine Slope Analysis

| Slope | Meaning | Implication |
|-------|---------|------------|
| **ACCELERATING** | QLine moving faster in trend direction | Trend is strengthening — stay in trade |
| **FLAT** | QLine barely moving | Trend pausing — watch for continuation or reversal |
| **DECELERATING** | QLine slowing down | Trend weakening — tighten stops, prepare for possible flip |

### 5. Extension Warnings

When price moves > 2.5 ATR away from QLine, an "⚠ EXT" warning appears. This means:
- Price is overextended from its trend line
- A pullback toward QLine is statistically likely
- NOT a reversal signal — just a warning that chasing here is risky
- Wait for price to pull back toward QLine for the next entry

---

## Trading Rules

### Rule 1: Trade WITH QLine Direction
- **Green QLine (below price)**: Only CALL positions
- **Red QLine (above price)**: Only PUT positions
- Never trade against QLine color

### Rule 2: Best Entries are Bounces
1. Wait for price to pull back toward QLine
2. Look for touch + reversal candle
3. ★★+ bounces are highest probability
4. Enter in trend direction with stop just beyond QLine

### Rule 3: Respect Flips
- **Red → Green**: Exit PUTs, prepare for CALLs
- **Green → Red**: Exit CALLs, prepare for PUTs
- First pullback after a flip to QLine is often the best entry

### Rule 4: Fresh Trends > Extended Trends
- FRESH/DEVELOPING trends have the best win rates
- MATURE/EXTENDED trends are still tradeable but use tighter stops
- More touches = weaker line = higher chance of break

### Rule 5: Don't Chase Extensions
- When "⚠ EXT" appears, don't enter new positions
- Wait for price to return toward QLine
- The extension warning is about TIMING, not direction

---

## Entry Setups

### Setup 1: ★★★ Bounce Entry (Primary — Highest Win Rate)
**Trigger:** Price pulls back to QLine and a ★★★ bounce marker appears
**Filter:** Trend age is FRESH or DEVELOPING (≤20 bars), touch count ≤3 (STRONG quality)
**Confirm with:** QCloud 4-5 layers in same direction (trend filter agrees), QMomentum RSI in oversold/overbought zone matching the bounce direction
**Entry:** On the bounce bar's close. The ★★★ scoring already confirms the candle has a strong body (>50% range), the trend is young, and the line is untested.
**Stop:** 0.5 ATR beyond QLine (just past the line — if price trades through with that margin, the bounce failed)
**Target:** QGrid nearest resistance/support level in trend direction. First partial at 1:1 risk/reward, let remainder trail with QLine.

This is QLine's bread and butter. The three-star scoring system filters out weak bounces automatically. You're entering at dynamic support/resistance with the trend, at a fresh untested level, with a strong reversal candle.

### Setup 2: Flip + First Pullback (Trend Reversal Entry)
**Trigger:** QLine flips color (red → green or green → red). Wait — don't enter on the flip bar itself.
**Wait for:** First pullback to QLine after the flip (trend age will be FRESH, 1-5 bars)
**Confirm with:** Moneyball zero-line flip in same direction (momentum agrees), QSMC CHoCH or BOS in flip direction (structural confirmation)
**Entry:** When price touches QLine on the first pullback and shows any bounce (even ★). First pullback after a flip is inherently high probability because the market just demonstrated it can break through the old trend.
**Stop:** Beyond QLine + 0.5 ATR. A tighter stop works here because the flip just proved this level.
**Target:** Previous swing high/low. Flips often produce moves that test the prior trend's extremes.

The flip + first pullback is aggressive but high reward. The flip itself proves the old trend broke. The pullback gives you a low-risk entry point. Combined with Moneyball/QSMC confirmation, this catches trend reversals early.

### Setup 3: Trend Continuation at DEVELOPING Age
**Trigger:** QLine green (bullish), trend age is DEVELOPING (6-20 bars), price dips toward QLine but may not quite touch
**Filter:** Slope = ACCELERATING (QLine is moving in trend direction), QCloud 4-5/5 in same direction
**Confirm with:** QBands — price near lower band in uptrend (or upper band in downtrend) providing mean reversion context. QCVD delta showing buying (or selling) flow matching the trend.
**Entry:** When price closes back in trend direction after approaching QLine zone (within 1 ATR of the line)
**Stop:** Below QLine (long) or above QLine (short)
**Target:** QBands opposite band (the 2σ band in trend direction) or QGrid next level

This catches the "meat of the move" — the trend is established but not yet extended. You're entering during normal trend breathing, not chasing an initial breakout.

### Setup 4: Extension Fade Warning + Re-entry
**Trigger:** ⚠ EXT warning appears (price > 2.5 ATR from QLine)
**Action:** Do NOT enter new positions. If already in a position, take partial profits.
**Wait for:** Price pulls back toward QLine and the ⚠ EXT disappears
**Re-entry:** When price returns to within 1 ATR of QLine AND QLine slope is still ACCELERATING or FLAT (not DECELERATING)
**Confirm with:** QMomentum RSI reaching oversold (in uptrend) or overbought (in downtrend) during the pullback — confirms the pullback is overdone
**Stop:** Beyond QLine
**Target:** Previous extension high/low (price tends to retest its extension)

Extensions are the most common trap for new traders — they see a strong trend and chase. The extension warning protects you. Wait for the pullback, then re-enter at a better price with the trend still intact.

### Setup 5: Weakening Touch + Flip Anticipation
**Trigger:** Touch count reaches 5+ (WEAKENING quality), QLine slope = DECELERATING
**Context:** The line has been tested many times and is losing strength. A flip is becoming increasingly likely.
**Action:** Tighten stops on existing positions to just beyond QLine. Do not add to positions.
**If flip occurs:** The weakening touches were the warning. Enter the new trend direction using Setup 2 (Flip + First Pullback).
**If bounce holds:** A bounce at WEAKENING touch count is lower probability but still valid. Enter smaller position size (half normal).

This isn't a standalone entry — it's a risk management framework. When QLine shows WEAKENING, you should be preparing for a possible trend change, not blindly trusting the line.

---

## Combining with Other Indicators

### QLine + QCloud
- QCloud provides the macro trend filter (which direction to trade). QLine provides the micro entry timing (where to get in).
- **Best signal:** QCloud 4-5/5 bullish + QLine green + ★★★ bounce = the trend, the direction, and the entry all align. This is a high-conviction CALL with structural backing.
- **Flip sequence:** When QLine flips green and within 1-3 bars QCloud layers increase to 4+ = double-confirmed trend reversal. The fast indicator (QLine) and the slow indicator (QCloud) both agree.
- **Conflict resolution:** QCloud still 4/5 bullish but QLine flips red = short-term counter-trend move. Two approaches: (1) trust QCloud and wait for QLine to flip back green before entering, or (2) take a small PUT position with tight stops expecting a quick mean reversion. Approach 1 is safer.

### QLine + QWave
- QLine provides direction and entry. QWave confirms whether the trend has momentum behind it.
- **Strongest setup:** QLine ★★★ bounce + QWave in STRONG zone = the bounce is happening during a powerful trend. High conviction.
- **Caution signal:** QLine bounce (even ★★★) but QWave in WEAK zone = the trend looks right but momentum is absent. Reduce size or skip. The bounce may hold but the follow-through will be limited.
- **Transition entry:** QWave transitions from WEAK to STRONG while QLine is bullish = momentum kicking in. Look for the next QLine touch as an entry.

### QLine + QBands
- QLine for trend direction and support/resistance. QBands for volatility envelope and mean reversion context.
- **Double support:** QLine bounce + price also touching QBands lower band (in uptrend) = two independent support mechanisms agree. Very strong bounce setup.
- **Extension context:** QLine ⚠ EXT + price at QBands 3σ band = overextended by both measures. High probability pullback incoming. Don't enter.
- **Squeeze + flip:** QBands squeeze fires in same direction as a QLine flip = volatility expansion beginning right as the trend changes. Explosive setup.

### QLine + Moneyball
- QLine for price-based trend. Moneyball for volume-weighted momentum.
- **Fastest confirmation:** Moneyball flips bullish (zero-line cross) + QLine flips green within 1-5 bars = momentum and trend agree. The convergence validates both signals.
- **Momentum divergence:** QLine still green but Moneyball flipping bearish repeatedly = momentum is weakening even though QLine support holds. Each subsequent QLine bounce is lower probability. Begin tightening stops.
- **STRONG zone entry:** QLine bounce + Moneyball in STRONG BULL zone (>70) = bouncing off support with powerful momentum behind it. Enter with conviction, wider targets.

### QLine + QMomentum
- QLine for trend and entry zones. QMomentum for overbought/oversold context.
- **Bounce + OS exit:** QLine bullish bounce + QMomentum RSI leaving oversold (crossing above 30) = the pullback that created the bounce also pushed RSI to oversold, confirming the dip is overdone. Enter CALL.
- **Divergence + weakening:** QMomentum bearish divergence (★★+) + QLine touch count at WEAKENING = both indicators warn of a coming reversal. Prepare for QLine flip.
- **StochRSI cross:** QLine bounce + QMomentum StochRSI bullish cross in oversold zone = fast momentum confirmation of the bounce. Enter immediately.

### QLine + QCVD
- QLine for price-based trend. QCVD for volume flow confirmation.
- **Volume-backed bounce:** QLine bounce + QCVD buy spike or bullish CVD trend = institutional money supports the bounce. High confidence entry.
- **Volume divergence:** QLine bounce but QCVD showing ACCEL SELL momentum = volume flow disagrees with the bounce. The bounce may fail — skip or take very small size.
- **Flip validation:** QLine flips green + QCVD flips bullish = both price trend and volume flow changed direction. Strong reversal confirmation.

### QLine + QSMC
- QLine for dynamic support/resistance. QSMC for institutional structure.
- **Institutional bounce:** QLine bounce at a location that overlaps with a QSMC bullish order block = institutional demand zone meets dynamic support. The highest conviction bounce setup possible.
- **Structure + trend:** QLine green + QSMC BOS bullish = QLine's trend direction confirmed by institutional structure break. Continue riding the trend with confidence.
- **CHoCH + flip:** QSMC CHoCH in opposite direction to QLine within 5 bars of each other = structure is changing. QLine flip should follow. Prepare to reverse positions.

### QLine + QGrid
- QLine for dynamic trend line. QGrid for static level targets.
- **Target setting:** Enter on QLine bounce → set target at QGrid next resistance level (for longs) or support (for shorts). This gives concrete exit points.
- **Level confluence:** QLine value aligns with a QGrid support level = two independent support systems at the same price. A bounce here has the backing of both dynamic and static analysis.
- **Break planning:** QLine flip + QGrid level break in same direction = the dynamic trend and static levels both agree price is going to a new range. Trail with QLine, target next QGrid level.

---

## Info Table Reference

| Row | Label | What It Shows |
|-----|-------|--------------|
| 1 | **QLine** | Indicator name |
| 2 | **Trend** | BULLISH or BEARISH |
| 3 | **QLine** | Current QLine price level |
| 4 | **Dist %** | Price distance from QLine as percentage |
| 5 | **Age** | Trend age (FRESH/DEVELOPING/MATURE/EXTENDED) + bar count |
| 6 | **Touches** | Touch count + quality (STRONG/NORMAL/WEAKENING) |
| 7 | **Slope** | QLine slope (ACCELERATING/FLAT/DECELERATING) |
| 8 | **Settings** | Auto or Manual mode |
| 9 | **Params** | Active atr_length / factor values |

---

## Common Mistakes

### Mistake 1: Trading Against QLine Color
If QLine is red (bearish), do not take CALL positions expecting a reversal. Wait for the flip. The ratcheting mechanism means QLine won't flip until price genuinely breaks through — respect that.

### Mistake 2: Chasing Extended Price
When ⚠ EXT appears, the worst thing to do is enter a new position. Price is far from its trend support — you have no nearby stop level and you're buying at the worst risk/reward. Wait for the pullback.

### Mistake 3: Ignoring Touch Quality
A ★★★ bounce at touch count 1 (STRONG) is a completely different trade than a ★ bounce at touch count 6 (WEAKENING). Treat them accordingly — full size vs. half size vs. skip.

### Mistake 4: Using EXTENDED Trends Like FRESH Ones
A FRESH trend (≤5 bars) has different probability than an EXTENDED trend (51+ bars). As trends age, the probability of reversal increases. Tighten stops and reduce targets on older trends.

### Mistake 5: Not Waiting for Candle Close
A bounce "confirmation" requires the bar to close. Mid-bar, price might look like it's bouncing off QLine, but it could reverse and break through by close. Always wait for the closed bar before entering.

### Mistake 6: Ignoring Slope Changes
QLine slope going from ACCELERATING to DECELERATING is a major warning even if the trend hasn't flipped. The trend is losing strength. A flat or decelerating slope + weakening touches = high probability of imminent flip.

---

## Hidden Plot Exports (For Master Confluence)

| Export | Value | Range |
|--------|-------|-------|
| X_Trend | Trend direction | 1 (bull) or -1 (bear) |
| X_QLineValue | QLine price level | Price |
| X_DistancePct | Distance from QLine | Percentage |
| X_TrendDuration | Bars in current trend | 1+ |
| X_TouchCount | Touches in current trend | 0+ |
| X_IsExtended | Price extended? | 0 or 1 |
| X_BounceScore | Last bounce quality | 0-3 |

---

## Summary

| Signal | Meaning | Action |
|--------|---------|--------|
| QLine green | Bullish trend | Trade CALLs only |
| QLine red | Bearish trend | Trade PUTs only |
| ▲ BULL flip | Trend reversed bullish | Exit PUTs, prepare CALLs |
| ▼ BEAR flip | Trend reversed bearish | Exit CALLs, prepare PUTs |
| ★★★ bounce | High quality support/resistance bounce | Best entry signal |
| ★★ bounce | Good quality bounce | Good entry signal |
| ⚠ EXT | Price overextended | Don't chase — wait for pullback |
| FRESH age | New trend | Highest probability trades |
| WEAKENING touches | Level tested many times | Line may break soon — be cautious |