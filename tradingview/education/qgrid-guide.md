# QGrid v2 — Education Guide

## What QGrid Does

QGrid answers: **where is price going and where will it stall?**

Every market has invisible walls — price levels where buyers and sellers cluster. QGrid finds these levels automatically from swing highs/lows, clusters nearby levels into stronger zones, and overlays VWAP as the dynamic "fair value" center. This gives you the complete target and risk framework for any trade.

---

## Support and Resistance Basics

**Support**: A price level where buying pressure historically absorbed selling. Price bounces off support. Think of it as a floor.

**Resistance**: A price level where selling pressure historically absorbed buying. Price reverses at resistance. Think of it as a ceiling.

The more times a level is tested and holds, the stronger it becomes. But when it finally breaks, it often becomes the opposite (broken resistance becomes support, and vice versa).

---

## Level Clustering

Not every swing high/low creates a meaningful level. QGrid clusters nearby swings (within 0.5 ATR) into a single stronger level. This reflects how the real market works — institutions don't place orders at exact prices, they place them in zones.

When you see a level with 4+ touches (★★★★), that's a zone where institutional orders have repeatedly been placed. It's far more significant than a single-touch level.

---

## Touch Count and Strength Scoring

Each level shows a star rating:
- ★ = 1 touch (weak — may not hold)
- ★★ = 2 touches (moderate)
- ★★★ = 3 touches (strong — likely to cause reaction)
- ★★★★ = 4 touches (very strong)
- ★★★★★ = 5+ touches (institutional level — extremely high probability reaction)

Higher touch counts mean the level has repeatedly proven itself. These are your highest-confidence target and entry zones.

---

## VWAP: The Fair Value Anchor

VWAP (Volume Weighted Average Price) represents the average price weighted by volume for the session. It's where most volume was transacted — the "fair value" for the day.

- **Price above VWAP**: Buyers are in control for the session. Longs are favored.
- **Price below VWAP**: Sellers are in control. Shorts are favored.

Institutional traders use VWAP heavily. Many algorithmic orders target VWAP. Crossing above or below VWAP is a session-level momentum signal.

---

## Level Expiry

Levels don't last forever. If a level hasn't been retested within 100 bars, it expires and is removed. This keeps the chart clean and focused on currently relevant levels.

Why? Markets evolve. A support level from last week may be irrelevant today if the market has moved significantly. QGrid automatically ages out stale levels.

---

## Trading Rules

### Rule 1: Levels Are Zones, Not Lines
A support level at 595.20 doesn't mean price will bounce at exactly 595.20. It means the zone around that price (±0.3 ATR) is where you should watch for reactions.

### Rule 2: More Touches = Higher Probability
A ★★★★ level demands your attention. Plan your trades around these levels — they're where institutional orders sit.

### Rule 3: VWAP Is the Session Bias
If you're long below VWAP, you're fighting the session trend. If you're short above VWAP, same problem. Use VWAP to determine your session bias and only take trades in the VWAP direction.

### Rule 4: Breakouts Need Volume
When price breaks a strong level, check QCVD. If there's a delta spike confirming the break, it's real. If volume is thin, it's likely a false breakout.

### Rule 5: S/R Flip Is Powerful
When resistance breaks and becomes support (or vice versa), the first retest of that flipped level is one of the highest-probability entries.

---

## Entry Setups

### Setup 1: Bounce at Strong Support
**Trigger:** Price approaches ★★★+ support level
**Confirm with:** QMomentum oversold, Moneyball bullish flip, QCVD buy delta
**Entry:** Bounce confirmation candle (close back above support)
**Stop:** Below the support zone
**Target:** Nearest resistance level or VWAP

### Setup 2: Rejection at Strong Resistance
**Trigger:** Price approaches ★★★+ resistance level
**Confirm with:** QMomentum overbought, QBands upper band touch, QCVD sell delta
**Entry:** Rejection candle (close back below resistance)
**Stop:** Above the resistance zone
**Target:** Nearest support level or VWAP

### Setup 3: Level Breakout
**Trigger:** Price breaks above resistance or below support
**Filter:** QCVD delta spike confirms, QSMC BOS in same direction
**Entry:** On the break or on first retest of the broken level
**Stop:** Back through the level
**Target:** Next level in the breakout direction

### Setup 4: VWAP Cross Entry
**Trigger:** Price crosses above/below VWAP
**Filter:** QCloud color matches, QWave not in weak zone
**Entry:** At the VWAP cross
**Stop:** Opposite side of VWAP
**Target:** Nearest S/R level in the cross direction

---

## Combining with Other Indicators

### QGrid + QBands
- QGrid provides the target levels, QBands provides the volatility context
- Price at strong S/R + at QBands 2σ = double rejection zone
- QBands squeeze fire in direction of nearest level break = explosive move

### QGrid + QSMC
- QSMC order blocks near QGrid levels = institutional confluence zone
- BOS/CHoCH at a strong S/R level = structurally confirmed breakout/reversal
- FVG between S/R levels = path price is likely to take

### QGrid + QWave
- QWave in strong zone + approaching S/R = level more likely to break
- QWave in weak zone + approaching S/R = level more likely to hold (mean reversion)

### QGrid + QCVD
- Level break + delta spike = genuine breakout with institutional flow
- Level test + delta in opposite direction = false breakout warning

---

## Common Mistakes

### Mistake 1: Treating Levels as Exact Prices
Levels are zones, not lines. Don't set limit orders at the exact level price. Instead, use the zone (level ± 0.3 ATR) as your area of interest.

### Mistake 2: Fading Breaks of Strong Levels
When a ★★★★★ level breaks with volume, don't try to fade it. The institutional orders are done. Trade in the breakout direction.

### Mistake 3: Ignoring VWAP Bias
VWAP is the institutional benchmark. Fighting VWAP direction is fighting smart money. Always know which side of VWAP you're on.

### Mistake 4: Overloading the Chart
QGrid can show up to 16 levels. For clean trading, focus on the 3-4 nearest levels to current price. The ones far away don't matter for your current trade.

### Mistake 5: Not Using Level Density
When level_density is high (many levels within 1 ATR), price is in a congestion zone. Expect choppy, range-bound action. Wait for a clean breakout before entering.
