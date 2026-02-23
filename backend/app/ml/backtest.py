"""
YELENA v2 — Backtesting Engine
Replays historical predictions through the Verdict Engine
and measures real win/loss rates using actual price movement.

Usage:
    python backtest.py --days 30 --symbols SPY,QQQ
    python backtest.py --start 2025-01-01 --end 2025-02-20
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import psycopg2
import psycopg2.extras
import numpy as np

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("yelena.backtest")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class HistoricalPrediction:
    """A prediction record from the database."""
    id: int
    time: datetime
    symbol: str
    timeframe: str
    direction: str
    confidence: float
    probability: float
    grade: str
    models_agreeing: int
    unanimous: bool


@dataclass
class HistoricalBar:
    """A 1min price bar from the database."""
    time: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BacktestVerdict:
    """A simulated verdict with outcome tracking."""
    symbol: str
    direction: str
    confidence: float
    entry_price: float
    target_price: float
    stop_price: float
    risk_reward: float
    opened_at: datetime
    # Outcome
    closed_at: Optional[datetime] = None
    result: Optional[str] = None    # WIN, LOSS, EXPIRED
    exit_price: Optional[float] = None
    pnl_pct: float = 0.0
    pnl_dollar: float = 0.0        # Assuming $1000 position
    hold_minutes: int = 0
    # Context
    tf_alignment: Dict[str, str] = field(default_factory=dict)
    reason: str = ""


@dataclass
class BacktestConfig:
    """Backtesting parameters — mirrors VerdictConfig."""
    anchor_tf: str = "15min"
    anchor_min_confidence: float = 65.0
    confirming_tfs: List[str] = field(default_factory=lambda: ["1min", "5min"])
    min_confirming: int = 1
    confirming_min_confidence: float = 55.0
    min_go_confidence: float = 70.0
    
    # Trade plan (ATR-based)
    atr_target_mult: float = 1.5
    atr_stop_mult: float = 0.6
    min_risk_reward: float = 1.5
    
    # Lifecycle
    max_duration_min: int = 15
    cooldown_min: int = 3
    prediction_max_age_sec: float = 120.0
    
    # Position sizing
    position_size: float = 1000.0   # $ per trade


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BacktestEngine:
    def __init__(self, db_url: str, config: BacktestConfig = None):
        self.db_url = db_url
        self.config = config or BacktestConfig()
        self.conn = None
        
        # Results
        self.verdicts: List[BacktestVerdict] = []
        self.skipped: int = 0
        
    def connect(self):
        self.conn = psycopg2.connect(self.db_url)
        logger.info("Connected to database")
    
    def close(self):
        if self.conn:
            self.conn.close()
    
    # ----------------------------------------------------------------
    # DATA LOADING
    # ----------------------------------------------------------------
    
    def load_predictions(
        self, symbols: List[str], start: datetime, end: datetime
    ) -> List[HistoricalPrediction]:
        """Load ensemble predictions from the database."""
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        placeholders = ",".join(["%s"] * len(symbols))
        cur.execute(f"""
            SELECT id, time, symbol, timeframe, signal, confidence, probabilities
            FROM predictions
            WHERE model_name = 'ensemble_v2'
              AND symbol IN ({placeholders})
              AND time >= %s AND time <= %s
            ORDER BY time ASC
        """, symbols + [start, end])
        
        predictions = []
        for row in cur.fetchall():
            probs = row["probabilities"]
            if isinstance(probs, str):
                probs = json.loads(probs)
            
            predictions.append(HistoricalPrediction(
                id=row["id"],
                time=row["time"],
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                direction=row["signal"],
                confidence=row["confidence"],
                probability=probs.get("probability", 0),
                grade=probs.get("grade", ""),
                models_agreeing=probs.get("models_agreeing", 0),
                unanimous=probs.get("unanimous", False),
            ))
        
        cur.close()
        logger.info(f"Loaded {len(predictions)} predictions ({start.date()} to {end.date()})")
        return predictions
    
    def load_bars(
        self, symbol: str, start: datetime, end: datetime
    ) -> List[HistoricalBar]:
        """Load 1min bars for price tracking."""
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Extend end by 30 min to capture outcomes after last prediction
        extended_end = end + timedelta(minutes=30)
        
        cur.execute("""
            SELECT time, symbol, open, high, low, close, volume
            FROM bars_1min
            WHERE symbol = %s AND time >= %s AND time <= %s
            ORDER BY time ASC
        """, (symbol, start, extended_end))
        
        bars = []
        for row in cur.fetchall():
            bars.append(HistoricalBar(
                time=row["time"],
                symbol=row["symbol"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))
        
        cur.close()
        logger.info(f"Loaded {len(bars)} bars for {symbol}")
        return bars
    
    # ----------------------------------------------------------------
    # ATR CALCULATION
    # ----------------------------------------------------------------
    
    def compute_atr(self, bars: List[HistoricalBar], idx: int, period: int = 14) -> float:
        """Compute ATR from bars ending at index idx."""
        if idx < 2:
            return bars[idx].close * 0.001 if bars else 1.0
        
        start = max(0, idx - period)
        true_ranges = []
        for i in range(start + 1, idx + 1):
            tr = max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - bars[i - 1].close),
                abs(bars[i].low - bars[i - 1].close),
            )
            true_ranges.append(tr)
        
        return sum(true_ranges) / len(true_ranges) if true_ranges else 1.0
    
    # ----------------------------------------------------------------
    # VERDICT SIMULATION
    # ----------------------------------------------------------------
    
    def run(self, symbols: List[str], start: datetime, end: datetime):
        """Run the full backtest."""
        logger.info("=" * 60)
        logger.info(f"BACKTEST: {start.date()} to {end.date()}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Config: anchor={self.config.anchor_tf}, "
                     f"min_confidence={self.config.min_go_confidence}%, "
                     f"min_rr={self.config.min_risk_reward}")
        logger.info("=" * 60)
        
        # Load all predictions
        predictions = self.load_predictions(symbols, start, end)
        if not predictions:
            logger.warning("No predictions found in date range")
            return
        
        # Load bars per symbol
        bars_by_symbol: Dict[str, List[HistoricalBar]] = {}
        bar_index: Dict[str, Dict[str, int]] = {}  # {symbol: {time_str: bar_idx}}
        
        for symbol in symbols:
            bars = self.load_bars(symbol, start, end)
            bars_by_symbol[symbol] = bars
            # Build time index for quick lookup
            bar_index[symbol] = {}
            for i, bar in enumerate(bars):
                key = bar.time.strftime("%Y-%m-%d %H:%M")
                bar_index[symbol][key] = i
        
        # Group predictions by symbol and timestamp
        # {symbol: {time_rounded: {timeframe: prediction}}}
        pred_groups: Dict[str, Dict[str, Dict[str, HistoricalPrediction]]] = {}
        for p in predictions:
            if p.symbol not in pred_groups:
                pred_groups[p.symbol] = {}
            time_key = p.time.strftime("%Y-%m-%d %H:%M")
            if time_key not in pred_groups[p.symbol]:
                pred_groups[p.symbol][time_key] = {}
            pred_groups[p.symbol][time_key][p.timeframe] = p
        
        # Simulate verdict engine per symbol
        for symbol in symbols:
            self._simulate_symbol(
                symbol,
                pred_groups.get(symbol, {}),
                bars_by_symbol.get(symbol, []),
                bar_index.get(symbol, {}),
            )
        
        self._print_results()
    
    def _simulate_symbol(
        self,
        symbol: str,
        pred_groups: Dict[str, Dict[str, HistoricalPrediction]],
        bars: List[HistoricalBar],
        bar_idx: Dict[str, int],
    ):
        """Simulate verdict engine for one symbol."""
        cooldown_until: Optional[datetime] = None
        
        # Process predictions in time order
        for time_key in sorted(pred_groups.keys()):
            preds = pred_groups[time_key]
            
            # Need anchor TF
            anchor = preds.get(self.config.anchor_tf)
            if not anchor:
                continue
            
            if anchor.direction == "HOLD":
                continue
            
            if anchor.confidence < self.config.anchor_min_confidence:
                continue
            
            # Check cooldown
            if cooldown_until and anchor.time < cooldown_until:
                continue
            
            anchor_dir = anchor.direction
            
            # Check confirming TFs
            confirming = []
            tf_alignment = {self.config.anchor_tf: anchor_dir}
            
            for ctf in self.config.confirming_tfs:
                cp = preds.get(ctf)
                if not cp:
                    tf_alignment[ctf] = "HOLD"
                    continue
                tf_alignment[ctf] = cp.direction
                if cp.direction == anchor_dir and cp.confidence >= self.config.confirming_min_confidence:
                    confirming.append(ctf)
            
            if len(confirming) < self.config.min_confirming:
                continue
            
            # Calculate composite confidence
            composite = anchor.confidence * 0.5
            confirm_weight = 0.5 / max(len(confirming), 1)
            for ctf in confirming:
                composite += preds[ctf].confidence * confirm_weight
            
            # 1hr boost
            hr = preds.get("1hr")
            if hr and hr.direction == anchor_dir:
                composite += 5.0
            
            composite = min(composite, 99.0)
            
            # Find entry price from bars
            entry_idx = bar_idx.get(time_key)
            if entry_idx is None:
                # Try nearest bar
                continue
            
            entry_bar = bars[entry_idx]
            entry_price = entry_bar.close
            
            # Compute ATR and trade plan
            atr = self.compute_atr(bars, entry_idx)
            
            if anchor_dir == "CALL":
                target = entry_price + atr * self.config.atr_target_mult
                stop = entry_price - atr * self.config.atr_stop_mult
            else:
                target = entry_price - atr * self.config.atr_target_mult
                stop = entry_price + atr * self.config.atr_stop_mult
            
            target_dist = abs(target - entry_price)
            stop_dist = abs(stop - entry_price)
            rr = target_dist / stop_dist if stop_dist > 0 else 0
            
            # GO/SKIP decision
            if composite < self.config.min_go_confidence or rr < self.config.min_risk_reward:
                self.skipped += 1
                continue
            
            # Simulate outcome — check future bars
            result, exit_price, exit_time, hold_min = self._simulate_outcome(
                bars, entry_idx, anchor_dir, target, stop
            )
            
            pnl_pct = 0.0
            if exit_price and entry_price > 0:
                if anchor_dir == "CALL":
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
            
            pnl_dollar = pnl_pct / 100 * self.config.position_size
            
            verdict = BacktestVerdict(
                symbol=symbol,
                direction=anchor_dir,
                confidence=composite,
                entry_price=entry_price,
                target_price=round(target, 2),
                stop_price=round(stop, 2),
                risk_reward=round(rr, 1),
                opened_at=anchor.time,
                closed_at=exit_time,
                result=result,
                exit_price=exit_price,
                pnl_pct=round(pnl_pct, 3),
                pnl_dollar=round(pnl_dollar, 2),
                hold_minutes=hold_min,
                tf_alignment=tf_alignment,
                reason=f"{self.config.anchor_tf} {anchor_dir} {anchor.confidence:.0f}% + {'+'.join(confirming)}",
            )
            
            self.verdicts.append(verdict)
            
            # Set cooldown
            cooldown_until = anchor.time + timedelta(minutes=self.config.cooldown_min + self.config.max_duration_min)
    
    def _simulate_outcome(
        self,
        bars: List[HistoricalBar],
        entry_idx: int,
        direction: str,
        target: float,
        stop: float,
    ) -> Tuple[str, Optional[float], Optional[datetime], int]:
        """
        Check future bars to determine if target or stop was hit first.
        Returns (result, exit_price, exit_time, hold_minutes).
        """
        max_bars = self.config.max_duration_min  # 1 bar = 1 min
        
        for i in range(1, min(max_bars + 1, len(bars) - entry_idx)):
            bar = bars[entry_idx + i]
            
            if direction == "CALL":
                # Check if target hit (high >= target)
                if bar.high >= target:
                    return "WIN", target, bar.time, i
                # Check if stop hit (low <= stop)
                if bar.low <= stop:
                    return "LOSS", stop, bar.time, i
            else:  # PUT
                # Check if target hit (low <= target)
                if bar.low <= target:
                    return "WIN", target, bar.time, i
                # Check if stop hit (high >= stop)
                if bar.high >= stop:
                    return "LOSS", stop, bar.time, i
        
        # Expired — exit at last bar's close
        if entry_idx + max_bars < len(bars):
            exit_bar = bars[entry_idx + max_bars]
            # Determine if expired trade was profitable
            if direction == "CALL":
                profitable = exit_bar.close > bars[entry_idx].close
            else:
                profitable = exit_bar.close < bars[entry_idx].close
            return "EXPIRED", exit_bar.close, exit_bar.time, max_bars
        
        return "EXPIRED", None, None, 0
    
    # ----------------------------------------------------------------
    # RESULTS
    # ----------------------------------------------------------------
    
    def _print_results(self):
        """Print comprehensive backtest results."""
        if not self.verdicts:
            logger.info("No verdicts generated in backtest period")
            return
        
        total = len(self.verdicts)
        wins = [v for v in self.verdicts if v.result == "WIN"]
        losses = [v for v in self.verdicts if v.result == "LOSS"]
        expired = [v for v in self.verdicts if v.result == "EXPIRED"]
        
        total_pnl = sum(v.pnl_dollar for v in self.verdicts)
        win_pnl = sum(v.pnl_dollar for v in wins)
        loss_pnl = sum(v.pnl_dollar for v in losses)
        expired_pnl = sum(v.pnl_dollar for v in expired)
        
        avg_win = win_pnl / len(wins) if wins else 0
        avg_loss = loss_pnl / len(losses) if losses else 0
        avg_hold = sum(v.hold_minutes for v in self.verdicts) / total if total else 0
        
        # By symbol
        symbols = set(v.symbol for v in self.verdicts)
        
        # By confidence band
        high_conf = [v for v in self.verdicts if v.confidence >= 80]
        med_conf = [v for v in self.verdicts if 70 <= v.confidence < 80]
        
        print("\n" + "=" * 70)
        print("YELENA v2 — BACKTEST RESULTS")
        print("=" * 70)
        
        print(f"\n📊 SUMMARY")
        print(f"  Total GO verdicts:    {total}")
        print(f"  Skipped (below threshold): {self.skipped}")
        print(f"  Wins:                 {len(wins)} ({len(wins)/total*100:.1f}%)")
        print(f"  Losses:               {len(losses)} ({len(losses)/total*100:.1f}%)")
        print(f"  Expired:              {len(expired)} ({len(expired)/total*100:.1f}%)")
        
        print(f"\n💰 P&L (${self.config.position_size:.0f} per trade)")
        print(f"  Total P&L:            ${total_pnl:+.2f}")
        print(f"  Win P&L:              ${win_pnl:+.2f}")
        print(f"  Loss P&L:             ${loss_pnl:+.2f}")
        print(f"  Expired P&L:          ${expired_pnl:+.2f}")
        print(f"  Avg Win:              ${avg_win:+.2f}")
        print(f"  Avg Loss:             ${avg_loss:+.2f}")
        print(f"  Avg Hold Time:        {avg_hold:.1f} min")
        
        if losses:
            profit_factor = abs(win_pnl / loss_pnl) if loss_pnl != 0 else float('inf')
            print(f"  Profit Factor:        {profit_factor:.2f}")
        
        # Per symbol
        print(f"\n📈 BY SYMBOL")
        for sym in sorted(symbols):
            sv = [v for v in self.verdicts if v.symbol == sym]
            sw = [v for v in sv if v.result == "WIN"]
            sp = sum(v.pnl_dollar for v in sv)
            wr = len(sw) / len(sv) * 100 if sv else 0
            print(f"  {sym}: {len(sv)} trades, {wr:.1f}% win rate, ${sp:+.2f}")
        
        # Per confidence band
        print(f"\n🎯 BY CONFIDENCE")
        for label, group in [("80%+", high_conf), ("70-80%", med_conf)]:
            if group:
                gw = [v for v in group if v.result == "WIN"]
                gp = sum(v.pnl_dollar for v in group)
                wr = len(gw) / len(group) * 100
                print(f"  {label}: {len(group)} trades, {wr:.1f}% win rate, ${gp:+.2f}")
        
        # Per direction
        calls = [v for v in self.verdicts if v.direction == "CALL"]
        puts = [v for v in self.verdicts if v.direction == "PUT"]
        print(f"\n📉 BY DIRECTION")
        for label, group in [("CALL", calls), ("PUT", puts)]:
            if group:
                gw = [v for v in group if v.result == "WIN"]
                gp = sum(v.pnl_dollar for v in group)
                wr = len(gw) / len(group) * 100
                print(f"  {label}: {len(group)} trades, {wr:.1f}% win rate, ${gp:+.2f}")
        
        # Daily breakdown
        print(f"\n📅 DAILY BREAKDOWN")
        days = {}
        for v in self.verdicts:
            day = v.opened_at.strftime("%Y-%m-%d")
            if day not in days:
                days[day] = {"trades": 0, "wins": 0, "pnl": 0}
            days[day]["trades"] += 1
            if v.result == "WIN":
                days[day]["wins"] += 1
            days[day]["pnl"] += v.pnl_dollar
        
        for day in sorted(days.keys()):
            d = days[day]
            wr = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0
            print(f"  {day}: {d['trades']} trades, {wr:.0f}% WR, ${d['pnl']:+.2f}")
        
        # Best and worst trades
        sorted_by_pnl = sorted(self.verdicts, key=lambda v: v.pnl_dollar, reverse=True)
        print(f"\n🏆 BEST TRADES")
        for v in sorted_by_pnl[:5]:
            print(f"  {v.opened_at.strftime('%m/%d %H:%M')} {v.symbol} {v.direction} "
                  f"{v.confidence:.0f}% → {v.result} ${v.pnl_dollar:+.2f} ({v.hold_minutes}m)")
        
        print(f"\n💀 WORST TRADES")
        for v in sorted_by_pnl[-5:]:
            print(f"  {v.opened_at.strftime('%m/%d %H:%M')} {v.symbol} {v.direction} "
                  f"{v.confidence:.0f}% → {v.result} ${v.pnl_dollar:+.2f} ({v.hold_minutes}m)")
        
        print("\n" + "=" * 70)
    
    def export_csv(self, filepath: str):
        """Export results to CSV."""
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "opened_at", "symbol", "direction", "confidence",
                "entry_price", "target_price", "stop_price", "risk_reward",
                "result", "exit_price", "pnl_pct", "pnl_dollar",
                "hold_minutes", "reason"
            ])
            for v in self.verdicts:
                writer.writerow([
                    v.opened_at.isoformat(), v.symbol, v.direction,
                    v.confidence, v.entry_price, v.target_price,
                    v.stop_price, v.risk_reward, v.result, v.exit_price,
                    v.pnl_pct, v.pnl_dollar, v.hold_minutes, v.reason
                ])
        logger.info(f"Results exported to {filepath}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="YELENA v2 Backtesting Engine")
    parser.add_argument("--symbols", default="SPY,QQQ", help="Comma-separated symbols")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--min-confidence", type=float, default=70.0, help="Min GO confidence")
    parser.add_argument("--min-rr", type=float, default=1.5, help="Min risk/reward ratio")
    parser.add_argument("--max-duration", type=int, default=15, help="Max hold minutes")
    parser.add_argument("--position-size", type=float, default=1000.0, help="$ per trade")
    parser.add_argument("--csv", default=None, help="Export results to CSV file")
    parser.add_argument("--db-url", default=None, help="Database URL (or uses SSM)")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Get DB URL
    db_url = args.db_url
    if not db_url:
        try:
            import boto3
            ssm = boto3.client("ssm", region_name="us-east-1")
            resp = ssm.get_parameter(Name="/yelena/database-url", WithDecryption=True)
            db_url = resp["Parameter"]["Value"]
        except Exception as e:
            logger.error(f"Failed to get DB URL from SSM: {e}")
            logger.error("Provide --db-url or set up AWS SSM")
            sys.exit(1)
    
    # Parse dates
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start = datetime.now(timezone.utc) - timedelta(days=args.days)
    
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(
            hour=23, minute=59, tzinfo=timezone.utc
        )
    else:
        end = datetime.now(timezone.utc)
    
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # Configure
    config = BacktestConfig(
        min_go_confidence=args.min_confidence,
        min_risk_reward=args.min_rr,
        max_duration_min=args.max_duration,
        position_size=args.position_size,
    )
    
    # Run
    engine = BacktestEngine(db_url, config)
    engine.connect()
    
    try:
        engine.run(symbols, start, end)
        
        if args.csv:
            engine.export_csv(args.csv)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
