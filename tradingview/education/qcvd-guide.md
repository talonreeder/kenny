# QCVD v2 — Education Guide

## What QCVD Does

QCVD answers: **who's really in control — buyers or sellers?**

Price can go up on weak buying or down on weak selling. QCVD looks underneath the price to show the actual volume flow. When price rises but CVD falls, someone is quietly selling into strength (distribution). When price falls but CVD rises, someone is quietly buying the dip (accumulation).

This is your institutional activity detector.

---

## How Delta Is Calculated

Without tick-level data, we approximate buying vs selling using each candle's close position:

- **Close near the high** → Most volume was buying (buyers won the bar)
- **Close near the low** → Most volume was selling (sellers won the bar)
- **Close in the middle** → Balanced, roughly equal buying and selling

The formula: `delta = volume × (close_position - sell_position)` where close_position is how far the close is from the low relative to the range.

A bullish engulfing candle closing at the high produces strong positive delta. A bearish candle closing at the low produces strong negative delta.

---

## CVD vs Delta

These are different things:

- **Delta** (histogram bars): The net buying/selling on THIS bar only. Resets each bar.
- **CVD** (the line): Running cumulative total of all delta. Shows the overall flow direction over time.

Think of delta as "who won this bar" and CVD as "who's winning the war."

---

## Key Signals

### 1. CVD Trend Flips
When CVD crosses above its moving average → buying flow is increasing (bullish).
When CVD crosses below → selling flow is increasing (bearish).

These are analogous to Moneyball flips but measured through volume instead of price momentum.

### 2. Delta Spikes (🔺/🔻)
Unusually large single-bar delta (3x+ the average). These represent sudden institutional order flow — a large buyer or seller stepping in.

- **Buy spike**: Institutional buying. Often precedes an upward move.
- **Sell spike**: Institutional selling. Often precedes a downward move.

Spikes during established trends confirm continuation. Spikes against the trend are reversal warnings.

### 3. CVD Divergences
The most powerful signal — when price and volume flow disagree:

- **Bearish divergence** (DIV ▼): Price making higher highs but CVD making lower highs. Translation: "Price looks strong, but sellers are quietly distributing. Reversal coming."
- **Bullish divergence** (DIV ▲): Price making lower lows but CVD making higher lows. Translation: "Price looks weak, but buyers are quietly accumulating. Bounce coming."

### 4. Delta Momentum
Shows whether buying/selling pressure is accelerating or decelerating:
- **ACCEL BUY**: Buying pressure increasing
- **ACCEL SELL**: Selling pressure increasing
- **FLAT**: No significant change in pressure

---

## Trading Rules

### Rule 1: CVD Confirms Price Direction
If price is going up AND CVD is trending up → genuine move, trust it.
If price is going up BUT CVD is flat or falling → suspicious, be cautious.

### Rule 2: Spikes Are Institutional Footprints
A delta spike tells you a large player just acted. The direction of the spike is more important than the size. Trade in the spike direction unless other indicators strongly disagree.

### Rule 3: Divergences Are Early Warnings
CVD divergence from price is one of the earliest reversal signals in the entire YELENA suite. But like all divergences, they need confirmation before acting.

### Rule 4: Volume Validates Everything
QCVD is the volume validation layer. Any signal from QCloud, QLine, QWave, etc. is stronger when QCVD confirms the flow direction matches.

---

## Entry Setups

### Setup 1: CVD Trend Flip + Price Confirmation
**Trigger:** CVD flips bullish (crosses above SMA)
**Confirm with:** Moneyball flip or QWave zone entry in same direction
**Stop:** Recent swing low
**Target:** Trail with QLine

### Setup 2: Delta Spike Entry
**Trigger:** Buy or sell spike detected
**Filter:** Spike aligns with QCloud trend direction
**Confirm with:** Delta momentum shows ACCEL in same direction
**Stop:** Opposite side of the spike bar
**Target:** QBands 2σ band in spike direction

### Setup 3: CVD Divergence Reversal
**Trigger:** CVD divergence detected (accumulation or distribution)
**Wait for:** CVD trend flip to confirm
**Confirm with:** QMomentum divergence + Moneyball flip
**Stop:** Beyond the divergence pivot
**Target:** Mean reversion to QBands basis

### Setup 4: Volume-Confirmed Breakout
**Trigger:** Price breaks QBands squeeze + delta spike in breakout direction
**Filter:** CVD trend already matches breakout direction
**Confirm with:** Delta momentum = ACCEL
**Stop:** Opposite side of squeeze range
**Target:** Trail with expanding QBands

---

## Combining with Other Indicators

### QCVD + QCloud
- CVD bullish + QCloud green = genuine uptrend with volume support
- CVD divergence during QCloud squeeze = the squeeze resolution will favor the CVD direction

### QCVD + Moneyball
- Both measure momentum differently (volume flow vs price ROC)
- When both flip simultaneously = strongest confirmation
- QCVD divergence + Moneyball divergence = very high probability reversal

### QCVD + QBands
- Delta spike during squeeze fire = explosive institutional breakout
- CVD trend flip + QBands squeeze fire in same direction = maximum conviction

### QCVD + QWave
- QWave measures trend strength, QCVD measures volume commitment
- Strong QWave zone + matching CVD trend = high conviction
- QWave weakening while CVD still strong = QWave will likely recover

---

## Common Mistakes

### Mistake 1: Ignoring CVD When Price Looks Clear
Price can lie. A beautiful uptrend with declining CVD is a distribution pattern. Always check if volume flow supports what price is showing.

### Mistake 2: Treating Every Spike as a Signal
Delta spikes happen regularly. Only act on spikes that align with the broader trend (CVD direction + other indicator confluence). Isolated spikes against the trend are noise until confirmed.

### Mistake 3: Using CVD Divergence Alone for Entries
CVD divergence is an early warning, not a trigger. The divergence can persist for many bars before price catches up. Wait for CVD trend flip or other confirmation.

### Mistake 4: Comparing CVD Levels Across Sessions
CVD is cumulative from chart start. The absolute level doesn't matter — what matters is the direction and whether it diverges from price. Don't compare "CVD is at 50,000" vs "CVD is at -20,000."

### Mistake 5: Forgetting Delta Is Approximated
Without tick data, our delta calculation is an approximation based on candle close position. It works well on liquid symbols (SPY, QQQ, TSLA) but may be less reliable on low-volume stocks. Stick to the high-volume options symbols YELENA is designed for.
