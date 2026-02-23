# QBands v2 — Education Guide

## What QBands Does

QBands tells you two things:

1. **Is a big move loading?** (Squeeze detection)
2. **How far has price stretched from normal?** (Band levels)

It combines Bollinger Bands (volatility-based) with Keltner Channels (ATR-based) to detect when volatility is compressing — the setup for explosive moves.

---

## The Spring Analogy

Think of price action like a spring:

- **Normal state:** Price bounces freely within its bands. BB bands are wide, sitting outside the KC bands. No squeeze.
- **Compression:** Price consolidates into a tight range. BB contracts because volatility drops. When BB gets SO tight it fits INSIDE KC → **squeeze is ON**. The spring is being compressed.
- **Release:** BB suddenly expands back outside KC → **squeeze fires**. The spring releases. Price explodes in one direction.

The longer the squeeze holds, the bigger the expected move. A 5-bar squeeze might give you a small pop. A 20-bar squeeze can produce a massive breakout.

---

## Multi-Level Bands (1σ, 2σ, 3σ)

The bands create concentric zones around price, like target rings on a bullseye:

| Level | Meaning | Statistical Expectation |
|-------|---------|------------------------|
| **1σ (inner)** | Normal trading range | Price stays here ~68% of time |
| **2σ (standard)** | Overbought/Oversold boundary | Price stays here ~95% of time |
| **3σ (outer)** | Extreme extension | Price stays here ~99.7% of time |

When price touches 2σ, it's stretched. When it hits 3σ, it's extremely stretched and very likely to snap back. This gives you graduated readings rather than a single line.

---

## Band Touch Quality

Not all band touches are equal. QBands scores each touch:

| Score | Type | What It Means | Reversal Probability |
|-------|------|---------------|---------------------|
| **★★★** | Wick-only touch | Price poked the band but closed back inside. Strong rejection. | Highest |
| **★★** | Body penetration | Candle body crossed the band but didn't close beyond it. Moderate rejection. | Medium |
| **★** | Close beyond band | Price closed outside the band. Band may be failing. | Lowest |

**Key rule:** 3-star touches at 2σ or 3σ are your best mean-reversion signals. The wick shows that sellers (at upper band) or buyers (at lower band) stepped in hard and pushed price back.

---

## The Squeeze Cycle

### Phase 1: Squeeze Building
- Background turns orange
- "◆ SQZ ON" label appears
- BB bands are visibly tighter than KC bands
- **Action:** Get ready. Don't trade yet — direction is unknown.

### Phase 2: Squeeze Fires
- "◆ FIRE ▲" (bullish) or "◆ FIRE ▼" (bearish) label appears
- Background returns to normal
- BB bands start expanding rapidly
- **Action:** This is your entry signal. The direction of the fire tells you which way to trade.

### Phase 3: Expansion
- Bandwidth increases
- Volatility state shows "EXPANDING"
- Price trends in the fire direction
- **Action:** Hold your position. Expanding bands mean the trend has momentum.

### Phase 4: Exhaustion
- Price reaches 2σ or 3σ bands
- Band touches start appearing (★ labels)
- Bandwidth begins to contract
- **Action:** Watch for exit signals. Mean reversion back to basis is likely.

---

## Reading the Info Table

| Row | What It Shows |
|-----|---------------|
| **Squeeze** | ON (with bar count) or OFF |
| **Last Fire** | Direction of most recent squeeze release (BULL/BEAR) |
| **BW %** | Bandwidth as percentage of price — lower = tighter = more compressed |
| **Volatility** | EXPANDING, CONTRACTING, or NORMAL relative to 20-bar average |
| **Position** | Where price sits within 2σ bands: LOWER/LOW/MID/HIGH/UPPER (0 to 1 scale) |
| **From Basis** | Distance from the EMA center line as a percentage |
| **Params** | Current BB length, multiplier, and KC multiplier |

---

## Trading Rules

### Rule 1: Only Trade Squeeze Fires with Confirmation
A squeeze fire alone isn't enough. Combine with:
- **QCloud direction** — fire should agree with cloud color
- **QWave zone** — fire direction should match QWave ±30+ zone
- A squeeze fire INTO a QWave strong zone is high conviction

### Rule 2: Respect the Bands for Exits
- If you're long and price hits upper 2σ with a ★★★ touch → tighten stop or exit
- If you're long and price hits upper 3σ → strong exit signal
- Mean reversion to basis is the natural target for band-touch trades

### Rule 3: Don't Fight Band Expansion
- When bandwidth is EXPANDING and price is trending → stay in the trade
- Bands expanding means momentum is real
- Only look for exits when expansion slows (bandwidth flattens or contracts)

### Rule 4: Squeeze Duration Matters
- Short squeezes (< 8 bars) → smaller moves, less reliable
- Medium squeezes (8-20 bars) → good setups
- Long squeezes (20+ bars) → potentially explosive moves, highest conviction

### Rule 5: Band Position for Entries
- Position 0.0-0.2 (LOWER zone) → oversold, look for long entries
- Position 0.4-0.6 (MID zone) → neutral, wait for direction
- Position 0.8-1.0 (UPPER zone) → overbought, look for short entries

---

## Entry Setups

### Setup 1: Squeeze Fire Entry
**Trigger:** Squeeze fires after 8+ bars of compression
**Direction:** Follow the fire (BULL = long, BEAR = short)
**Confirm with:** QCloud color agreement, QWave in matching ±30+ zone
**Stop:** Opposite side of the squeeze range (the tight consolidation area)
**Target:** 2σ band in the fire direction, or trail with QLine

### Setup 2: Band Touch Reversal
**Trigger:** ★★★ (wick-only) touch at 2σ band
**Direction:** Opposite to the touch (upper touch = short, lower touch = long)
**Confirm with:** QWave momentum fading (score moving toward zero)
**Stop:** Beyond the 3σ band
**Target:** Basis (EMA center line) for first target, opposite 1σ band for extended target

### Setup 3: 3σ Extreme Reversal
**Trigger:** Price hits 3σ band (any touch quality)
**Direction:** Mean reversion back toward basis
**Confirm with:** Any other indicator showing exhaustion
**Stop:** Tight — if price sustains beyond 3σ, the move is parabolic and you should respect it
**Target:** 2σ band for first target, basis for full target

### Setup 4: Bandwidth Expansion Trend
**Trigger:** Bandwidth state switches from CONTRACTING to EXPANDING
**Direction:** Follow price direction at the expansion start
**Confirm with:** Squeeze fire or QCloud/QLine direction agreement
**Stop:** Basis (EMA center line)
**Target:** Trail using 1σ band as dynamic stop

---

## Combining with Other Indicators

### QBands + QCloud
- Squeeze fire direction should match QCloud color (green cloud = bullish fire)
- Band touches at 2σ during established cloud trends are pullback entries
- A squeeze forming while QCloud is in a strong state → the fire will likely continue the trend

### QBands + QLine
- QLine provides your stop/trail level; QBands provides your target zones
- Squeeze fire + QLine flip in same direction = very strong signal
- When price is between QLine (support) and upper 2σ (target), you have a defined trade range

### QBands + QWave
- QWave tells you HOW STRONG the move is; QBands tells you HOW FAR price has stretched
- Squeeze fire into QWave ±60+ zone (strong bull/bear) = highest conviction setup
- Band touch + QWave divergence = strong reversal signal

### Triple Confluence (All Four)
The strongest possible setup:
1. QBands squeeze fires bullish
2. QCloud is green (bullish direction)
3. QLine is green/below price (bullish support)
4. QWave crosses above +30 (entering bull zone)

All four agreeing = maximum conviction trade.

---

## Common Mistakes

### Mistake 1: Trading During the Squeeze
The squeeze itself is NOT a signal. It means "big move coming" but doesn't tell you direction. Wait for the fire.

### Mistake 2: Fading Strong Trends at 2σ
In a strong trend, price can "ride the band" — staying at or above 2σ for extended periods. Don't blindly short every upper band touch. Check QWave strength first. If QWave is in strong bull (60+), the band riding is normal.

### Mistake 3: Ignoring Touch Quality
A ★ touch (close beyond band) means the band is failing, not that price will reverse. Only ★★★ and ★★ touches are reliable reversal signals.

### Mistake 4: Short Squeeze Trades
Squeezes under 5 bars often produce false fires. The move hasn't had enough compression to generate real momentum. Wait for 8+ bars minimum.

### Mistake 5: Not Using Basis as a Target
The basis (EMA center line) is a natural magnet. After any extreme move away from basis, price tends to return. Use it as your first profit target on mean-reversion trades, not some arbitrary price level.

---

## Key Concepts Summary

| Concept | What to Remember |
|---------|-----------------|
| **Squeeze** | BB inside KC = volatility compressed = big move loading |
| **Squeeze Fire** | BB expands outside KC = the move is here, trade the direction |
| **Bandwidth** | Lower = tighter = more compressed. Watch for contraction → expansion cycle |
| **Band Position** | 0 = lower band, 1 = upper band. Extremes (< 0.2 or > 0.8) mean stretched |
| **Touch Quality** | ★★★ wick > ★★ body > ★ close. Higher stars = better reversal odds |
| **Mean Reversion** | Price always tends back toward basis. It's your natural first target |
