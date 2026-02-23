# QCloud — Step Moving Average Cloud
## Trading Education Guide (YELENA)

---

## What is QCloud?

QCloud is a **trend identification indicator** that creates a visual cloud from 5 stepped moving averages. The cloud color instantly communicates the trend direction and strength — no interpretation needed.

**What it answers**: "What is the current trend, and how strong is it?"

---

## How It Works

### The 5 Moving Averages

QCloud calculates 5 EMAs with progressively longer periods using a step multiplier:

| MA | Calculation (default SPY 5min) | Period |
|----|-------------------------------|--------|
| MA1 (Fastest) | Base length | 10 |
| MA2 | Base × step^1 | 15 |
| MA3 (Mid) | Base × step^2 | 22 |
| MA4 | Base × step^3 | 34 |
| MA5 (Slowest) | Base × step^4 | 50 |

The cloud is the filled area between MA1 (fastest) and MA5 (slowest).

### Layer Scoring (0 to 5)

Each "layer" is scored as bullish when TWO conditions are met:
1. Price is above that MA
2. That MA is above the next slower MA

This ensures we're measuring TRUE trend alignment, not just price position.

**5/5 bullish** = Perfect alignment = Strong uptrend
**0/5 bullish** = Perfect bearish alignment = Strong downtrend
**2-3/5** = Mixed/transitional = Choppy conditions

### Flip Smoothing

To prevent whipsaws, QCloud requires a state change to persist for N consecutive bars before confirming. In Auto mode:
- Scalp (≤1min): 2 bars smoothing
- Intraday (5-15min): 2 bars smoothing
- Swing (1hr+): 3 bars smoothing

---

## Auto-Optimization

### The Problem It Solves

TradingView indicator settings are **global** — they apply to every chart. If you optimize QCloud for SPY 5min, those exact settings are used when you switch to TSLA 1min. This makes the indicator unreliable across different instruments and timeframes.

### The Solution

QCloud **automatically detects** the current symbol and timeframe, then selects optimal parameters from a built-in lookup table. You never need to touch the settings.

### Volatility Groups

| Group | Symbols | Characteristics | Setting Approach |
|-------|---------|----------------|------------------|
| **Index ETFs** | SPY, SPX, QQQ | Lower volatility, more predictable | Standard base, tighter step |
| **High Volatility** | TSLA, NVDA | Fast moves, more noise | Shorter base, wider step |
| **Medium Volatility** | META, NFLX, AAPL, GOOGL, MSFT, AMZN, AMD | Balanced behavior | Moderate settings |

### Timeframe Groups

| Group | Timeframes | Characteristics | Setting Approach |
|-------|-----------|----------------|------------------|
| **Scalp** | ≤1 minute | Needs fast reaction | Shorter periods, less smoothing |
| **Intraday** | 5-15 minutes | Primary trading timeframe | Balanced settings |
| **Swing** | 1 hour+ | Needs to filter noise | Longer periods, more smoothing |

### How to Verify

The info table on chart always shows:
- **Settings**: "Auto" or "Manual"
- **Params**: The active base_length / step_mult / smoothing values

You can always see exactly what parameters are being used.

---

## The 7-Tier State System

| State | Layers | Cloud Color | Meaning | Trading Action |
|-------|--------|-------------|---------|---------------|
| **STRONG BULL** | 5/5 | 🟢 Dark Green | Perfect bullish alignment | Maximum conviction CALL |
| **BULLISH** | 4/5 | 🟢 Light Green | Strong bullish, minor weakness | Favor CALL, normal stops |
| **WEAK BULL** | 3/5 | 🟡 Yellow | Transitional, leaning bull | Reduce size or wait |
| **WEAK BEAR** | 2/5 | 🟠 Orange | Transitional, leaning bear | Reduce size or wait |
| **BEARISH** | 1/5 | 🔴 Red | Strong bearish | Favor PUT, normal stops |
| **STRONG BEAR** | 0/5 | 🔴 Dark Red | Perfect bearish alignment | Maximum conviction PUT |

**Key Insight**: The transition zone (WEAK BULL / WEAK BEAR at 2-3 layers) is where most false signals occur. Experienced traders either wait for clarity or reduce position size significantly.

---

## Cloud Dynamics

### What to Watch

QCloud tracks how the cloud itself is behaving, not just the layer count:

| Metric | What It Means | Why It Matters |
|--------|--------------|---------------|
| **Cloud Width %** | Distance between MA1 and MA5 as % of price | Wide = strong trend, Narrow = weak/indecisive |
| **EXPANDING** | Cloud getting wider | Trend is strengthening |
| **CONTRACTING** | Cloud getting narrower | Trend is weakening, potential reversal |
| **◆ SQUEEZE** | Width < 60% of 20-bar average | Volatility compression — BIG move coming |

### Squeeze Trading

When you see "◆ SQUEEZE" in the info table or on the chart:
1. The cloud has compressed significantly
2. This means volatility is coiling up
3. A breakout is likely imminent
4. The DIRECTION of the breakout is what matters

**Squeeze → Bullish breakout**: Price breaks above cloud + layers flip to 4-5 = Strong CALL
**Squeeze → Bearish breakout**: Price breaks below cloud + layers flip to 0-1 = Strong PUT

---

## Price Position

| Position | Meaning | Implication |
|----------|---------|------------|
| **ABOVE** | Price is above the entire cloud | Bullish — cloud acts as support below |
| **INSIDE** | Price is within the cloud | Indecisive — cloud is being tested |
| **BELOW** | Price is below the entire cloud | Bearish — cloud acts as resistance above |

**Best setups**: Price ABOVE cloud + 4-5 layers bullish = CALL. Price BELOW cloud + 0-1 layers bearish = PUT.

**Warning signals**: Price INSIDE cloud at any layer count = trend is being challenged. Wait for resolution.

---

## Trading Rules

### Rule 1: Trade WITH the Cloud
- **Green cloud (4-5 layers)**: Only look for CALL setups
- **Red cloud (0-1 layers)**: Only look for PUT setups
- **Yellow/Orange (2-3 layers)**: Wait or reduce position size

### Rule 2: Cloud Flips are High-Probability Signals
When confirmed bullish count crosses from ≤2 to ≥3 (or vice versa), this is a trend change signal.
- **Flip to Bullish**: Prepare for CALL entries
- **Flip to Bearish**: Prepare for PUT entries

### Rule 3: Full Cloud = Maximum Conviction
- **5/5 Bullish**: Highest conviction CALL — all layers perfectly aligned
- **0/5 Bearish**: Highest conviction PUT — all layers perfectly aligned

### Rule 4: Squeeze → Breakout = Action Signal
When a squeeze resolves AND the layers flip to an extreme (4-5 or 0-1), this is often one of the strongest setups.

### Rule 5: Don't Fight the Cloud
If QCloud shows 5/5 bullish, do NOT take PUT positions expecting a reversal. Wait for the cloud to actually flip first. The trend is your friend until it ends.

---

## Entry Setups

### Setup 1: Cloud Flip Entry
**Trigger:** QCloud flips from bearish (≤2 layers) to bullish (≥3 layers) or vice versa
**Confirm with:** QLine flip in same direction, Moneyball zero-line flip agreeing
**Entry:** On the flip bar's close, or on first pullback to MA3 (midline of cloud)
**Stop:** Beyond MA5 (slowest MA — the cloud's far edge)
**Target:** Trail with QLine; first target at nearest QGrid resistance level

This is QCloud's primary signal. A confirmed flip means the trend has structurally changed across multiple timeframes of the moving average stack. The smoothing filter ensures you're not reacting to a single noisy bar.

### Setup 2: Cloud Pullback (Trend Continuation)
**Trigger:** QCloud 4-5 layers bullish (or 0-1 bearish), price pulls back INTO the cloud
**Filter:** Cloud dynamics = EXPANDING or at least not CONTRACTING
**Confirm with:** QMomentum RSI entering oversold (<30) on bullish pullback, or overbought (>70) on bearish pullback. QBands band touch in the pullback direction adds conviction.
**Entry:** When price bounces off MA3-MA5 zone and closes back ABOVE MA1
**Stop:** Below MA5 (if long) or above MA5 (if short)
**Target:** Previous swing high/low, or QGrid nearest level in trend direction

The pullback to the cloud is one of the highest win-rate entries because the trend is already confirmed — you're just waiting for a better price within an established move.

### Setup 3: Squeeze Breakout
**Trigger:** QCloud shows ◆ SQUEEZE (cloud width < 60% of 20-bar average)
**Wait for:** Squeeze resolves — cloud begins EXPANDING and layers shift to 4-5 (bullish) or 0-1 (bearish)
**Confirm with:** QBands squeeze fire in same direction, QCVD delta spike confirming volume flow
**Entry:** On the first bar where layers reach 4+ (or ≤1) after the squeeze
**Stop:** Opposite side of the squeeze range (MA5 level during squeeze)
**Target:** Extended — squeezes produce large moves. Trail with QLine and exit when cloud dynamics shift to CONTRACTING

Squeezes are compression events. All 5 MAs converge, then when they separate, the move is explosive. The key is waiting for the direction to confirm before entering.

### Setup 4: Full Alignment Power Trade
**Trigger:** QCloud reaches 5/5 bullish (or 0/5 bearish) — perfect alignment
**Filter:** Price position = ABOVE cloud (for bull) or BELOW cloud (for bear)
**Confirm with:** QWave in STRONG zone (ADX > trending threshold), QSMC structure = UPTREND (for bull)
**Entry:** On next minor pullback (price dips toward MA1 but stays above cloud)
**Stop:** Below MA3 (mid-cloud — if price penetrates this deep, the alignment is broken)
**Target:** Trail aggressively with QLine; this is maximum conviction so let it run

5/5 alignment is rare and powerful. Every single layer agrees on direction. These produce the longest and smoothest runs but the entry must be disciplined — don't chase if price is already extended far from the cloud.

---

## Combining with Other Indicators

### QCloud + QLine
- QCloud provides the trend direction bias (which side to trade). QLine provides the entry timing (where to get in).
- **Best combo signal:** QCloud 4-5/5 bullish + QLine green + price bouncing off QLine support = highest conviction CALL entry
- **Flip confluence:** QCloud flip to bullish happening within 1-3 bars of QLine flipping green = double-confirmed trend reversal. Enter on first pullback to QLine after both flip.
- **Conflict resolution:** If QCloud is 4/5 bullish but QLine just flipped red, QLine is seeing a short-term reversal the cloud hasn't caught yet. Reduce size or wait for QCloud to drop below 3 before taking the QLine PUT signal.

### QCloud + QWave
- QCloud tells you the trend. QWave tells you if the trend has strength behind it.
- **Strongest signal:** QCloud 5/5 bullish + QWave in STRONG zone (ADX above trending) = the trend is both aligned AND powerful. Maximum conviction.
- **Warning signal:** QCloud 4/5 bullish but QWave in WEAK zone = the cloud looks bullish but momentum is fading. Reduce size, tighten stops.
- **Best entry:** QCloud expanding + QWave transitioning from WEAK to STRONG = trend is gaining momentum. This catches the "second wave" — often the strongest part of a trend.

### QCloud + QBands
- QCloud for direction bias, QBands for volatility context and mean reversion timing.
- **Pullback entry:** QCloud 4-5 bullish + price touches QBands lower band (2σ) = oversold within an uptrend. Enter CALL when price bounces back inside bands.
- **Squeeze confluence:** QCloud ◆ SQUEEZE + QBands squeeze fire in same direction = double volatility compression releasing. Extremely powerful breakout signal.
- **Overextension warning:** QCloud 5/5 bullish but price at QBands 3σ upper band = trend is right but price is stretched. Wait for pullback to basis or 1σ before entering.

### QCloud + Moneyball
- Both measure trend/momentum but at different speeds. QCloud is slower (5 EMAs), Moneyball is faster (volume-weighted ROC).
- **Fastest confirmation:** Moneyball flips bullish (zero-line cross) THEN QCloud flips bullish within 3-5 bars = Moneyball caught it first, QCloud confirmed. Enter on QCloud flip.
- **Divergence warning:** Moneyball flipping bearish while QCloud is still 4-5/5 bullish = early warning. Moneyball sees momentum fading before the cloud structure breaks. Tighten stops but don't exit yet — wait for QCloud to actually drop below 3 layers.
- **Explosive combo:** Moneyball in STRONG BULL zone (>70) + QCloud 5/5 = maximum momentum in a perfectly aligned trend. Ride aggressively.

### QCloud + QMomentum
- QCloud for trend, QMomentum for overbought/oversold timing within that trend.
- **Best entry:** QCloud 4-5 bullish + QMomentum RSI leaving oversold (crossing back above 30) = trend is up and the pullback is ending. Enter CALL.
- **Divergence alert:** QMomentum bearish divergence (★★+) while QCloud still 5/5 = earliest warning the trend may be exhausting. Don't exit, but stop entering new positions and watch for QCloud layer count to start dropping.
- **Avoid:** QCloud in transition (2-3 layers) + QMomentum in neutral (RSI 40-60) = no signal from either. Stay out.

### QCloud + QCVD
- QCloud for price-based trend, QCVD for volume-based confirmation.
- **Volume-confirmed trend:** QCloud 4-5 bullish + QCVD trend bullish (CVD above SMA) = both price structure and volume flow agree. High conviction.
- **Hidden distribution:** QCloud 5/5 bullish but QCVD bearish divergence (price HH, CVD LH) = institutions quietly selling into the rally. This is one of the earliest reversal warnings. Begin tightening stops.
- **Delta spike + cloud:** QCVD buy spike during QCloud squeeze = institutional money entering before the breakout. Strong predictor of bullish squeeze resolution.

### QCloud + QSMC
- QCloud for trend direction, QSMC for structural context (where institutions are positioned).
- **Structure-confirmed trend:** QCloud 4-5 bullish + QSMC structure = UPTREND + BOS bullish = triple confirmation. Maximum conviction CALL.
- **CHoCH warning:** QSMC CHoCH bearish while QCloud is still green = the faster structural analysis sees a reversal forming. Watch QCloud layers — if they start dropping to 3 then 2, the CHoCH was right.
- **OB entry:** QCloud pullback to MA3-MA5 zone that overlaps with a QSMC bullish order block = institutional demand meets cloud support. Enter with confidence.

### QCloud + QGrid
- QCloud for trend bias, QGrid for target levels.
- **Target framework:** QCloud 4-5 bullish → use QGrid resistance levels as take-profit targets and support levels as stop-loss zones.
- **VWAP alignment:** QCloud bullish + price above VWAP = session trend and cloud trend agree. Strongest intraday bias.
- **Level break confirmation:** QGrid resistance breakout confirmed by QCloud layers increasing to 5/5 = structural and level-based breakout. Enter on the first pullback to the broken level (now support).

---

## Info Table Reference

The info table on your chart shows all real-time metrics:

| Row | Label | What It Shows |
|-----|-------|--------------|
| 1 | **QCloud** | Indicator name and version |
| 2 | **State** | Current 7-tier state (STRONG BULL → STRONG BEAR) |
| 3 | **Layers** | Bullish layer count (0/5 → 5/5) |
| 4 | **Price** | Price position relative to cloud (ABOVE / INSIDE / BELOW) |
| 5 | **Width** | Cloud width as percentage of price |
| 6 | **Cloud** | Cloud dynamics (EXPANDING / CONTRACTING / ◆ SQUEEZE) |
| 7 | **Settings** | Current mode (Auto = green, Manual = orange) |
| 8 | **Params** | Active parameters: base_length / step_mult / smoothing |

---

## Common Mistakes

### Mistake 1: Trading Against the Cloud
If QCloud is 4-5/5 bullish, do not take PUT positions hoping for a reversal. The cloud represents the weight of 5 different timeframe perspectives of trend — fighting that is fighting the structural trend.

### Mistake 2: Overtrading in Yellow/Orange
2-3 layers means the moving averages are mixed — some bullish, some bearish. This is the "no man's land" of trend trading. Either wait for clarity (4+ or ≤1) or reduce position size dramatically.

### Mistake 3: Ignoring Squeeze Signals
A squeeze always resolves with a big move. If you see ◆ SQUEEZE and ignore it, you'll miss one of the best setups the platform produces. Set alerts for squeeze resolution.

### Mistake 4: Using Manual Mode Without Reason
Auto mode selects parameters optimized from backtesting against real historical data for each symbol/timeframe. Only switch to Manual if you have a specific thesis about why the optimizer's choice is wrong for current conditions.

### Mistake 5: Not Checking Price Position
5/5 bullish with price BELOW cloud is contradictory — the layers are lagging and haven't caught up to a price drop. Similarly, 0/5 bearish with price ABOVE cloud means the layers are lagging a price rally. Always cross-check layer count with price position.

### Mistake 6: Treating QCloud as an Entry Signal
QCloud tells you DIRECTION, not TIMING. Entering blindly because the cloud turned green is like buying a stock just because it's trending up — you might be buying at the top of a pullback. Use QLine bounces, QMomentum zone exits, or Moneyball flips for entry timing.

---

## Hidden Plot Exports (For Master Confluence)

QCloud exports these values via hidden plots for Master Confluence to read:

| Export Name | Value | Range |
|-------------|-------|-------|
| X_BullCount | Bullish layer count | 0-5 |
| X_TrendStrength | Trend strength % | 0-100 |
| X_CloudWidthPct | Cloud width as % of price | 0+ |
| X_IsSqueeze | Squeeze active? | 0 or 1 |
| X_Direction | Direction signal | -1, 0, or 1 |
| X_MA1 | Fastest MA value | Price level |
| X_MA5 | Slowest MA value | Price level |

---

## Summary

| Signal | Meaning | Action |
|--------|---------|--------|
| Cloud fully green (5/5) | Strong uptrend | Maximum conviction CALL |
| Cloud green (4/5) | Uptrend | Favor CALL |
| Cloud flip to bullish | Trend reversal starting | Prepare for CALL |
| Cloud flip to bearish | Trend reversal starting | Prepare for PUT |
| Cloud red (1/5) | Downtrend | Favor PUT |
| Cloud fully red (0/5) | Strong downtrend | Maximum conviction PUT |
| Cloud yellow/orange (2-3) | Uncertainty | Wait or reduce size |
| ◆ SQUEEZE | Volatility compression | Big move imminent — watch direction |
| Price ABOVE + green | Confirmed uptrend | Best CALL setups |
| Price BELOW + red | Confirmed downtrend | Best PUT setups |
| Price INSIDE any color | Trend being tested | Wait for resolution |