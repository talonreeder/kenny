# QSMC v2 — Education Guide

## What QSMC Does

QSMC answers: **where are the institutions positioned?**

Smart money (hedge funds, market makers, prop firms) can't hide their footprints. They leave structural signatures in price action: breaks of structure, order blocks, and fair value gaps. QSMC automates detecting all of these so you can trade with institutional flow instead of against it.

---

## Market Structure: The Foundation

All price action exists within a structure — either an uptrend or downtrend.

**Uptrend**: Higher highs AND higher lows. Each swing point is above the previous one.
**Downtrend**: Lower lows AND lower highs. Each swing point is below the previous one.

Structure tells you the bias. Everything else is context.

---

## Break of Structure (BOS)

A BOS occurs when price breaks a swing point in the SAME direction as the current trend. It's the market saying "yes, this trend continues."

- **Bullish BOS**: In an uptrend, price breaks above a previous swing high → trend continuation confirmed
- **Bearish BOS**: In a downtrend, price breaks below a previous swing low → trend continuation confirmed

BOS is a green light to look for entries in the trend direction.

---

## Change of Character (CHoCH)

A CHoCH occurs when price breaks a swing point in the OPPOSITE direction of the current trend. It's the earliest structural reversal signal.

- **Bullish CHoCH**: In a downtrend, price breaks above a swing high → downtrend may be over, uptrend beginning
- **Bearish CHoCH**: In an uptrend, price breaks below a swing low → uptrend may be over, downtrend beginning

CHoCH is displayed with larger labels because it's a more significant event. A single CHoCH can change your entire trading bias.

---

## Order Blocks (OB)

An order block is the last opposing candle before a strong impulsive move. This is where institutional orders were placed.

**Bullish OB** (green box): The last bearish (red) candle before a strong bullish impulse. Institutions were buying into that selling. When price returns to this zone, the remaining unfilled orders act as demand — price bounces.

**Bearish OB** (red box): The last bullish (green) candle before a strong bearish impulse. Institutions were selling into that buying. When price returns, remaining sell orders act as supply — price rejects.

### Order Block Lifecycle
1. **Created**: OB detected after impulsive move
2. **Active**: Price hasn't returned to the zone yet (unmitigated)
3. **Tested**: Price returns to the zone → entry opportunity
4. **Mitigated**: Price trades completely through the zone → OB removed

Only act on OBs that haven't been mitigated. Once price trades through, the institutional orders are filled and the zone is dead.

### Strength Filter
Not every opposing candle before a move is a real OB. QSMC requires the impulsive move to be at least `ob_strength × ATR` to qualify. This filters out weak, insignificant "order blocks."

---

## Fair Value Gaps (FVG)

An FVG is a three-candle pattern where the middle candle's move creates a gap in price coverage.

**Bullish FVG** (blue box): Candle 1's high is below Candle 3's low → there's a gap where no price was traded. This represents buying imbalance — more demand than supply.

**Bearish FVG** (orange box): Candle 1's low is above Candle 3's high → selling imbalance.

Price tends to return and "fill" these gaps because the market seeks equilibrium. FVGs act as magnets for price.

### FVG Fill
When price returns to the midpoint of the FVG, it's considered filled and the box disappears. Partially filled FVGs remain visible.

---

## Trading Rules

### Rule 1: Trade With Structure
If structure is UPTREND, only look for longs. If DOWNTREND, only shorts. CHoCH is your signal to flip bias.

### Rule 2: BOS = Continuation, CHoCH = Reversal
BOS confirms your existing bias. CHoCH challenges it. Treat CHoCH with more weight — it signals a potential regime change.

### Rule 3: OBs Are Entry Zones, Not Signals
An active order block tells you WHERE to enter, not WHEN. Wait for price to return to the OB zone, then look for entry confirmation from other indicators.

### Rule 4: Respect Mitigation
Once an OB is mitigated (price traded through it), it's dead. Don't expect it to act as support/resistance again.

### Rule 5: FVGs Are Magnets
When an FVG exists between current price and a target, expect price to fill it before continuing. Plan entries and exits around FVG levels.

---

## Entry Setups

### Setup 1: CHoCH + OB Entry
**Trigger:** CHoCH detected (structure reversal)
**Wait for:** Price to pull back to an order block aligned with new structure
**Confirm with:** QMomentum divergence, Moneyball flip
**Stop:** Beyond the order block
**Target:** Next swing high/low

### Setup 2: BOS + OB Retest
**Trigger:** BOS confirms trend continuation
**Wait for:** Price returns to the most recent OB in trend direction
**Confirm with:** QBands band touch, QWave still in strong zone
**Stop:** Beyond OB zone
**Target:** New swing extension (measured move)

### Setup 3: FVG Fill Entry
**Trigger:** Active FVG exists between price and a key level
**Wait for:** Price enters the FVG zone
**Confirm with:** QCVD delta spike in FVG direction, QCloud color matches
**Stop:** Opposite side of FVG
**Target:** Extension beyond the FVG (continuation)

### Setup 4: OB + FVG Overlap
**Trigger:** Order block and FVG overlap at the same price zone
**Meaning:** Double institutional interest — very high probability zone
**Entry:** Price enters the overlap zone
**Stop:** Below/above the entire zone
**Target:** Previous structure high/low

---

## Combining with Other Indicators

### QSMC + QCloud
- Structure (QSMC) + trend direction (QCloud) alignment = maximum trend confidence
- CHoCH + QCloud color flip = double-confirmed reversal
- OB test during QCloud squeeze = setup for explosive move

### QSMC + QBands
- OB zones near QBands 2σ/3σ = extra strong support/resistance
- FVG fill + QBands band touch = dual reason for reversal

### QSMC + QCVD
- BOS with delta spike = structurally confirmed with institutional volume
- CHoCH + CVD divergence = strongest reversal signal in the suite
- OB test with volume confirmation = highest probability entry zone

### QSMC + QLine
- BOS in QLine direction = continue riding
- CHoCH against QLine = tighten stops, prepare for QLine flip

---

## Common Mistakes

### Mistake 1: Trading Every BOS
In ranging markets, you'll get false BOS signals as price chops around. BOS is most reliable in trending environments (confirmed by QWave in strong zone).

### Mistake 2: Entering at OBs Without Confirmation
An OB is a zone, not a signal. Price entering an OB means "get ready," not "enter now." Wait for a rejection candle, a Moneyball flip, or another confirmation.

### Mistake 3: Ignoring CHoCH
Many traders keep looking for trend entries after a CHoCH. A CHoCH is the market telling you the trend is over. Respect it — at minimum, stop taking entries in the old direction.

### Mistake 4: Expecting All FVGs to Fill
While FVGs tend to get filled, strong trends can leave unfilled gaps behind. Not every FVG is a trade opportunity. Prioritize FVGs that align with the current structure direction.

### Mistake 5: Using Too Small a Swing Length
If swing_length is too small, you'll detect micro-structure that doesn't represent real institutional activity. The optimizer will find the right balance, but generally 4-6 works well for intraday scalping.
