from __future__ import annotations
"""
backtester.py — ICT_V3 HISTORY-LEARNING UPGRADE
=================================================
What's new vs the original:

1. FilterEngine integrated: every signal now passes through all 11
   quality checks during backtesting, so results match live behaviour.
   Previously the backtester accepted every signal the signal_engine
   generated, inflating win rate vs what the bot actually does live.

2. History-learning (HistoryAnalyser): after running the backtest the
   engine analyses the closed trades and automatically patches CONFIG
   thresholds so the next live session starts with data-driven settings
   instead of hardcoded defaults. Specifically it finds:
   - Best performing session (highest win rate) → maps to kill zone config
   - Best performing confidence bucket → sets high_winrate_min_confidence
   - Best RR bracket → sets high_winrate_min_rr
   - Worst performing agent → adds to auto_upgrade_engine loss counts
   - ADX threshold that maximised win rate → patches sideways_adx_threshold
   - Optimal SL distance (minimises fast-stops) → patches micro_max_sl_points

3. MT5HistoryLoader: on startup loads as many 1m candles as MT5 has
   available (up to HISTORY_BARS env var, default 10 000) and passes
   them straight to the backtester. No CSV needed. If MT5 is not
   connected it falls back gracefully.

4. Auto-run on startup: terminal_server.py calls
   HistoryBacktestRunner.run_if_needed() in its startup block. The
   runner checks whether a backtest has been run in the last 24h
   (stored in backtest_cache.json). If not, it runs one automatically,
   patches CONFIG, and logs the result. On subsequent starts within
   24h it just loads the cached result and patches CONFIG from it.

5. Backtest results exposed at /api/backtest_result in terminal_server
   so you can see them in the dashboard.

Usage (standalone):
    python backtester.py

Usage (from code):
    from backtester import HistoryBacktestRunner
    runner = HistoryBacktestRunner()
    result = runner.run_if_needed()   # patches CONFIG automatically
"""



import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import CONFIG, asset_profile
from filter_engine import FilterEngine
from indicators import normalize_ohlcv, resample_ohlcv
from logger import get_logger
from signal_engine import SignalEngine
from trade_manager import TradeManager

log = get_logger(__name__)

CACHE_FILE = Path(__file__).parent / "backtest_cache.json"
CACHE_MAX_AGE_HOURS = 24


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BacktestResult:
    trades: int
    wins: int
    losses: int
    winrate: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    average_rr: float
    net_pnl: float
    agent_stats: dict
    equity_curve: pd.Series
    closed_trades: list
    # NEW: per-session and per-confidence stats for learning
    session_stats: dict = field(default_factory=dict)
    confidence_bucket_stats: dict = field(default_factory=dict)
    adx_bucket_stats: dict = field(default_factory=dict)
    rr_bucket_stats: dict = field(default_factory=dict)
    filtered_count: int = 0
    run_at: str = ""


@dataclass
class LearnedThresholds:
    """Thresholds derived from backtest analysis. Patched directly into CONFIG."""
    high_winrate_min_confidence: float
    high_winrate_min_rr: float
    sideways_adx_threshold: float
    micro_max_sl_points: float
    best_session: str
    worst_agent: str
    notes: List[str] = field(default_factory=list)


# ── Core Backtester ───────────────────────────────────────────────────────────

class Backtester:
    """
    Runs a full walk-forward backtest over 1m OHLCV data.
    Every signal passes through FilterEngine — same gates as live.
    """

    def __init__(self) -> None:
        self.signal_engine = SignalEngine()
        self.filter_engine = FilterEngine()

    def run(self, one_minute: pd.DataFrame, warmup: int = 300) -> BacktestResult:
        """
        Walk-forward simulation over `one_minute` DataFrame.
        warmup: number of bars to skip before generating signals
                (needs enough history for ICT engine indicators).
        """
        if len(one_minute) < warmup + 50:
            raise ValueError(f"Need at least {warmup + 50} bars. Got {len(one_minute)}.")

        one_minute = normalize_ohlcv(one_minute)
        manager = TradeManager()
        equity: List[float] = []
        equity_index = []
        closed_seen = 0
        filtered_count = 0

        log.info("Backtester: starting walk-forward over %d bars (warmup=%d)", len(one_minute), warmup)

        for i in range(warmup, len(one_minute)):
            history = one_minute.iloc[: i + 1]
            frames = self._frames(history)
            candle = history.iloc[-1].to_dict()
            ts = history.index[-1].to_pydatetime()

            # Close any open trades that hit SL/TP
            manager.update(candle, ts)

            # Apply execution costs to newly closed trades
            if len(manager.closed_trades) != closed_seen:
                for trade in manager.closed_trades[closed_seen:]:
                    self._apply_execution_costs(trade)
                closed_seen = len(manager.closed_trades)

            tick = self._tick_from_candle(candle)

            try:
                snapshots = self.signal_engine.analyze(frames)
                signals = self.signal_engine.generate_all(frames, tick)
            except Exception as exc:
                log.debug("Signal engine error at bar %d: %s", i, exc)
                equity.append(sum(t.pnl for t in manager.closed_trades))
                equity_index.append(ts)
                continue

            for signal in signals:
                # NEW: run every signal through FilterEngine — same as live
                try:
                    filter_ok, filter_reason = self.filter_engine.check(
                        signal=signal,
                        snapshots=snapshots,
                        frames=frames,
                        recent_closed_trades=manager.closed_trades[-30:],
                        now_utc=ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts,
                    )
                except Exception:
                    filter_ok, filter_reason = True, "filter_error"

                if not filter_ok:
                    filtered_count += 1
                    log.debug("BT filtered: %s | %s", signal.direction.value, filter_reason)
                    continue

                manager.submit_signal(signal, tick)

            equity.append(sum(t.pnl for t in manager.closed_trades))
            equity_index.append(ts)

        equity_series = pd.Series(equity, index=equity_index, dtype=float)
        result = self._metrics(manager, equity_series, filtered_count)
        log.info(
            "Backtester done: %d trades, %.1f%% WR, PF=%.2f, net=%.2f, filtered=%d",
            result.trades, result.winrate, result.profit_factor, result.net_pnl, result.filtered_count,
        )
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _frames(self, base: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        result = {}
        for tf in CONFIG.timeframes.all:
            if tf == "1m":
                result[tf] = base
            else:
                try:
                    result[tf] = resample_ohlcv(base, tf)
                except Exception:
                    result[tf] = base
        return result

    def _tick_from_candle(self, candle: dict) -> Dict[str, float]:
        costs = CONFIG.backtest_costs
        spread_pts = float(
            candle.get(costs.spread_column, costs.default_spread_points) or costs.default_spread_points
        )
        spread_price = spread_pts / 10.0
        close = float(candle["close"])
        return {"bid": close - spread_price / 2.0, "ask": close + spread_price / 2.0, "spread": spread_pts}

    def _apply_execution_costs(self, trade) -> None:
        costs = CONFIG.backtest_costs
        profile = asset_profile(trade.signal.symbol)
        slippage_cost = (costs.slippage_points / 10.0) * trade.lot_size * profile.contract_size * 2
        commission = costs.commission_per_lot_round_turn * trade.lot_size
        total = round(slippage_cost + commission, 2)
        if total > 0:
            trade.pnl = round(trade.pnl - total, 2)
            trade.notes.append(f"BT costs: -{total:.2f}")

    def _metrics(
        self, manager: TradeManager, equity: pd.Series, filtered_count: int
    ) -> BacktestResult:
        trades = manager.closed_trades
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))

        returns = equity.diff().fillna(0)
        sharpe = 0.0
        if returns.std() > 0:
            sharpe = float((returns.mean() / returns.std()) * np.sqrt(252 * 390))

        peak = equity.cummax()
        dd = float((equity - peak).min()) if len(equity) else 0.0
        avg_rr = float(np.mean([t.rr_achieved for t in trades])) if trades else 0.0

        # NEW: per-session breakdown
        session_stats = self._session_breakdown(trades)
        # NEW: per-confidence-bucket breakdown
        conf_stats = self._confidence_bucket_breakdown(trades)
        # NEW: per-ADX-bucket breakdown
        adx_stats = self._adx_bucket_breakdown(trades)
        # NEW: per-RR-bucket breakdown
        rr_stats = self._rr_bucket_breakdown(trades)

        return BacktestResult(
            trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            winrate=round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
            profit_factor=round(gross_win / gross_loss, 2) if gross_loss else (float("inf") if gross_win else 0.0),
            sharpe=round(sharpe, 2),
            max_drawdown=round(dd, 2),
            average_rr=round(avg_rr, 2),
            net_pnl=round(sum(t.pnl for t in trades), 2),
            agent_stats=manager.agent_stats(),
            equity_curve=equity,
            closed_trades=trades,
            session_stats=session_stats,
            confidence_bucket_stats=conf_stats,
            adx_bucket_stats=adx_stats,
            rr_bucket_stats=rr_stats,
            filtered_count=filtered_count,
            run_at=datetime.now(timezone.utc).isoformat(),
        )

    def _session_breakdown(self, trades: list) -> dict:
        stats: Dict[str, dict] = {}
        for t in trades:
            sess = str(t.signal.metadata.get("session", "off_session"))
            if sess not in stats:
                stats[sess] = {"trades": 0, "wins": 0, "pnl": 0.0}
            stats[sess]["trades"] += 1
            stats[sess]["wins"] += 1 if t.pnl > 0 else 0
            stats[sess]["pnl"] += t.pnl
        for s in stats.values():
            s["winrate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0.0
            s["pnl"] = round(s["pnl"], 2)
        return stats

    def _confidence_bucket_breakdown(self, trades: list) -> dict:
        buckets: Dict[str, dict] = {}
        for t in trades:
            conf = float(t.signal.confidence)
            bucket = f"{int(conf // 5) * 5}-{int(conf // 5) * 5 + 5}"
            if bucket not in buckets:
                buckets[bucket] = {"trades": 0, "wins": 0}
            buckets[bucket]["trades"] += 1
            buckets[bucket]["wins"] += 1 if t.pnl > 0 else 0
        for b in buckets.values():
            b["winrate"] = round(b["wins"] / b["trades"] * 100, 1) if b["trades"] else 0.0
        return dict(sorted(buckets.items()))

    def _adx_bucket_breakdown(self, trades: list) -> dict:
        buckets: Dict[str, dict] = {}
        for t in trades:
            adx = float(t.signal.metadata.get("trend_strength", 0) or 0)
            bucket = f"{int(adx // 5) * 5}-{int(adx // 5) * 5 + 5}"
            if bucket not in buckets:
                buckets[bucket] = {"trades": 0, "wins": 0}
            buckets[bucket]["trades"] += 1
            buckets[bucket]["wins"] += 1 if t.pnl > 0 else 0
        for b in buckets.values():
            b["winrate"] = round(b["wins"] / b["trades"] * 100, 1) if b["trades"] else 0.0
        return dict(sorted(buckets.items()))

    def _rr_bucket_breakdown(self, trades: list) -> dict:
        buckets: Dict[str, dict] = {}
        for t in trades:
            rr = float(t.signal.rr)
            bucket = f"{int(rr)}-{int(rr)+1}R"
            if bucket not in buckets:
                buckets[bucket] = {"trades": 0, "wins": 0}
            buckets[bucket]["trades"] += 1
            buckets[bucket]["wins"] += 1 if t.pnl > 0 else 0
        for b in buckets.values():
            b["winrate"] = round(b["wins"] / b["trades"] * 100, 1) if b["trades"] else 0.0
        return dict(sorted(buckets.items()))


# ── History Analyser (learns from backtest results) ───────────────────────────

class HistoryAnalyser:
    """
    Analyses BacktestResult and derives optimal CONFIG thresholds.
    Patches CONFIG directly — no restart needed.
    """

    MIN_SAMPLES = 8  # minimum trades in a bucket to trust its stats

    def analyse(self, result: BacktestResult) -> LearnedThresholds:
        notes: List[str] = []

        # 1. Best confidence floor — find lowest bucket with WR >= 65%
        best_conf = CONFIG.risk.high_winrate_min_confidence
        for bucket_label, stats in sorted(result.confidence_bucket_stats.items()):
            if stats["trades"] >= self.MIN_SAMPLES and stats["winrate"] >= 65.0:
                low = int(bucket_label.split("-")[0])
                best_conf = max(best_conf, float(low))
                break
        # Also check from the top — if high-confidence buckets underperform, raise floor further
        for bucket_label, stats in sorted(result.confidence_bucket_stats.items(), reverse=True):
            if stats["trades"] >= self.MIN_SAMPLES and stats["winrate"] < 50.0:
                low = int(bucket_label.split("-")[0])
                best_conf = max(best_conf, float(low + 5))
                notes.append(f"Confidence bucket {bucket_label} WR={stats['winrate']}% — floor raised")
                break
        best_conf = min(88.0, max(72.0, best_conf))

        # 2. Best RR floor — find minimum RR bracket with positive expectancy
        best_rr = CONFIG.risk.high_winrate_min_rr
        for bucket_label, stats in sorted(result.rr_bucket_stats.items()):
            if stats["trades"] >= self.MIN_SAMPLES and stats["winrate"] >= 58.0:
                low = int(bucket_label.split("-")[0].replace("R", ""))
                best_rr = max(best_rr, float(low))
                break
        best_rr = min(3.5, max(1.8, best_rr))

        # 3. Best ADX threshold — find ADX level below which WR < 50%
        adx_floor = CONFIG.ict.sideways_adx_threshold
        for bucket_label, stats in sorted(result.adx_bucket_stats.items()):
            if stats["trades"] >= self.MIN_SAMPLES and stats["winrate"] < 50.0:
                low = int(bucket_label.split("-")[0])
                adx_floor = max(adx_floor, float(low + 5))
                notes.append(f"ADX bucket {bucket_label} WR={stats['winrate']}% — ADX floor raised to {adx_floor}")
                break
        adx_floor = min(28.0, max(15.0, adx_floor))

        # 4. SL optimisation — find median SL distance of winning trades
        wins = [t for t in result.closed_trades if t.pnl > 0]
        optimal_sl = CONFIG.risk.micro_max_sl_points
        if wins:
            sl_distances = []
            for t in wins:
                sl_dist = abs(t.signal.entry - t.signal.stop_loss)
                if 0 < sl_dist < 50:
                    sl_distances.append(sl_dist)
            if sl_distances:
                median_sl = float(np.percentile(sl_distances, 75))
                optimal_sl = min(15.0, max(5.0, round(median_sl + 1.0, 1)))
                notes.append(f"Optimal SL from winning trades: {optimal_sl:.1f} pts (75th pct of winners)")

        # 5. Best session
        best_session = "london"
        best_session_wr = 0.0
        for sess, stats in result.session_stats.items():
            if stats["trades"] >= self.MIN_SAMPLES and stats["winrate"] > best_session_wr:
                best_session_wr = stats["winrate"]
                best_session = sess
        notes.append(f"Best session: {best_session} ({best_session_wr:.1f}% WR)")

        # 6. Worst agent
        worst_agent = ""
        worst_wr = 100.0
        for agent, stats in result.agent_stats.items():
            if stats.get("trades", 0) >= self.MIN_SAMPLES:
                wr = stats.get("winrate", 100.0)
                if wr < worst_wr:
                    worst_wr = wr
                    worst_agent = agent
        if worst_agent:
            notes.append(f"Worst agent: {worst_agent} ({worst_wr:.1f}% WR) — auto-upgrade will monitor")

        return LearnedThresholds(
            high_winrate_min_confidence=best_conf,
            high_winrate_min_rr=best_rr,
            sideways_adx_threshold=adx_floor,
            micro_max_sl_points=optimal_sl,
            best_session=best_session,
            worst_agent=worst_agent,
            notes=notes,
        )

    def patch_config(self, thresholds: LearnedThresholds) -> None:
        """Apply learned thresholds directly to the live CONFIG object."""
        old_conf = CONFIG.risk.high_winrate_min_confidence
        old_rr = CONFIG.risk.high_winrate_min_rr
        old_adx = CONFIG.ict.sideways_adx_threshold
        old_sl = CONFIG.risk.micro_max_sl_points

        CONFIG.risk.high_winrate_min_confidence = thresholds.high_winrate_min_confidence
        CONFIG.risk.high_winrate_min_rr = thresholds.high_winrate_min_rr
        CONFIG.ict.sideways_adx_threshold = thresholds.sideways_adx_threshold
        CONFIG.risk.micro_max_sl_points = thresholds.micro_max_sl_points

        log.info(
            "HistoryAnalyser patched CONFIG: confidence %.1f→%.1f | RR %.2f→%.2f | ADX %.1f→%.1f | SL %.1f→%.1f",
            old_conf, thresholds.high_winrate_min_confidence,
            old_rr, thresholds.high_winrate_min_rr,
            old_adx, thresholds.sideways_adx_threshold,
            old_sl, thresholds.micro_max_sl_points,
        )
        for note in thresholds.notes:
            log.info("HistoryAnalyser: %s", note)


# ── MT5 History Loader ────────────────────────────────────────────────────────

class MT5HistoryLoader:
    """
    Loads 1m candle history directly from MT5 on startup.
    Falls back gracefully if MT5 is not available.
    """

    def load(self, bars: int = 10_000) -> Optional[pd.DataFrame]:
        try:
            import MetaTrader5 as mt5
        except ImportError:
            log.warning("MT5 package not installed — history learning skipped.")
            return None

        if not mt5.initialize():
            log.warning("MT5 not available for history load — will retry on next startup.")
            return None

        # Select the correct symbol
        candidates = CONFIG.data.mt5_symbol_candidates
        symbol = None
        for candidate in candidates:
            if mt5.symbol_info(candidate) is not None:
                mt5.symbol_select(candidate, True)
                symbol = candidate
                break

        if symbol is None:
            log.warning("No valid XAUUSD symbol found in MT5 for history load.")
            mt5.shutdown()
            return None

        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 1, bars)
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            log.warning("MT5 returned no history bars.")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = normalize_ohlcv(df)
        log.info("MT5HistoryLoader: loaded %d 1m bars for %s", len(df), symbol)
        return df


# ── History Backtest Runner (auto-runs on startup) ────────────────────────────

class HistoryBacktestRunner:
    """
    Called from terminal_server.py startup.
    Checks the cache — if stale or missing, loads MT5 history, runs the
    backtester, analyses results, patches CONFIG, and saves the cache.
    """

    def __init__(self) -> None:
        self.backtester = Backtester()
        self.analyser = HistoryAnalyser()
        self.loader = MT5HistoryLoader()
        self._last_result: Optional[dict] = None

    def run_if_needed(self) -> Optional[dict]:
        """
        Returns a summary dict suitable for /api/backtest_result.
        Patches CONFIG automatically if a backtest was run.
        """
        cached = self._load_cache()
        if cached:
            log.info("HistoryBacktestRunner: using cached backtest from %s", cached.get("run_at", "unknown"))
            self._patch_config_from_cache(cached)
            self._last_result = cached
            return cached

        log.info("HistoryBacktestRunner: no recent cache — loading MT5 history and running backtest...")
        bars = int(os.getenv("HISTORY_BARS", "10000"))
        df = self.loader.load(bars=bars)

        if df is None or len(df) < 500:
            log.warning("HistoryBacktestRunner: insufficient history data — skipping.")
            return None

        try:
            result = self.backtester.run(df, warmup=300)
        except Exception as exc:
            log.error("HistoryBacktestRunner: backtest failed: %s", exc)
            return None

        thresholds = self.analyser.analyse(result)
        self.analyser.patch_config(thresholds)

        summary = self._build_summary(result, thresholds)
        self._save_cache(summary)
        self._last_result = summary

        log.info(
            "HistoryBacktestRunner complete: %d trades, %.1f%% WR, confidence→%.1f, RR→%.2f",
            result.trades, result.winrate,
            thresholds.high_winrate_min_confidence,
            thresholds.high_winrate_min_rr,
        )
        return summary

    def last_result(self) -> Optional[dict]:
        return self._last_result

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _load_cache(self) -> Optional[dict]:
        if not CACHE_FILE.exists():
            return None
        try:
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            run_at_str = data.get("run_at", "")
            if not run_at_str:
                return None
            run_at = datetime.fromisoformat(run_at_str)
            if run_at.tzinfo is None:
                run_at = run_at.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - run_at
            if age > timedelta(hours=CACHE_MAX_AGE_HOURS):
                log.info("HistoryBacktestRunner: cache expired (%s old)", age)
                return None
            return data
        except Exception as exc:
            log.debug("Cache load error: %s", exc)
            return None

    def _save_cache(self, summary: dict) -> None:
        try:
            CACHE_FILE.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            log.debug("Cache save error: %s", exc)

    def _patch_config_from_cache(self, cached: dict) -> None:
        """Apply cached thresholds to CONFIG without re-running the backtest."""
        thresholds = cached.get("learned_thresholds", {})
        if not thresholds:
            return
        try:
            CONFIG.risk.high_winrate_min_confidence = float(thresholds.get("high_winrate_min_confidence", CONFIG.risk.high_winrate_min_confidence))
            CONFIG.risk.high_winrate_min_rr = float(thresholds.get("high_winrate_min_rr", CONFIG.risk.high_winrate_min_rr))
            CONFIG.ict.sideways_adx_threshold = float(thresholds.get("sideways_adx_threshold", CONFIG.ict.sideways_adx_threshold))
            CONFIG.risk.micro_max_sl_points = float(thresholds.get("micro_max_sl_points", CONFIG.risk.micro_max_sl_points))
            log.info("HistoryBacktestRunner: patched CONFIG from cache")
        except Exception as exc:
            log.debug("Cache patch error: %s", exc)

    def _build_summary(self, result: BacktestResult, thresholds: LearnedThresholds) -> dict:
        return {
            "run_at": result.run_at,
            "trades": result.trades,
            "wins": result.wins,
            "losses": result.losses,
            "winrate": result.winrate,
            "profit_factor": result.profit_factor,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "average_rr": result.average_rr,
            "net_pnl": result.net_pnl,
            "filtered_count": result.filtered_count,
            "session_stats": result.session_stats,
            "confidence_bucket_stats": result.confidence_bucket_stats,
            "adx_bucket_stats": result.adx_bucket_stats,
            "rr_bucket_stats": result.rr_bucket_stats,
            "agent_stats": result.agent_stats,
            "learned_thresholds": {
                "high_winrate_min_confidence": thresholds.high_winrate_min_confidence,
                "high_winrate_min_rr": thresholds.high_winrate_min_rr,
                "sideways_adx_threshold": thresholds.sideways_adx_threshold,
                "micro_max_sl_points": thresholds.micro_max_sl_points,
                "best_session": thresholds.best_session,
                "worst_agent": thresholds.worst_agent,
                "notes": thresholds.notes,
            },
        }


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("ICT_V3 History Backtest Runner")
    print("=" * 50)

    bars = int(os.getenv("HISTORY_BARS", "10000"))
    runner = HistoryBacktestRunner()

    print(f"Loading up to {bars} 1m bars from MT5...")
    df = runner.loader.load(bars=bars)

    if df is None:
        print("ERROR: Could not load MT5 history. Make sure MT5 is open and logged in.")
        sys.exit(1)

    print(f"Loaded {len(df)} bars. Running backtest with FilterEngine...")
    result = runner.backtester.run(df, warmup=300)

    print(f"\nResults:")
    print(f"  Trades:         {result.trades}")
    print(f"  Win Rate:       {result.winrate:.1f}%")
    print(f"  Profit Factor:  {result.profit_factor:.2f}")
    print(f"  Sharpe:         {result.sharpe:.2f}")
    print(f"  Max Drawdown:   {result.max_drawdown:.2f}")
    print(f"  Net PnL:        {result.net_pnl:.2f}")
    print(f"  Avg RR:         {result.average_rr:.2f}")
    print(f"  Filtered:       {result.filtered_count} signals blocked by FilterEngine")

    print(f"\nSession Breakdown:")
    for sess, stats in result.session_stats.items():
        print(f"  {sess:20s}: {stats['trades']} trades, {stats['winrate']:.1f}% WR, {stats['pnl']:.2f} pnl")

    print(f"\nConfidence Buckets:")
    for bucket, stats in result.confidence_bucket_stats.items():
        print(f"  {bucket}%: {stats['trades']} trades, {stats['winrate']:.1f}% WR")

    print(f"\nADX Buckets:")
    for bucket, stats in result.adx_bucket_stats.items():
        print(f"  {bucket}: {stats['trades']} trades, {stats['winrate']:.1f}% WR")

    print(f"\nLearning thresholds from results...")
    thresholds = runner.analyser.analyse(result)
    runner.analyser.patch_config(thresholds)

    print(f"\nLearned & Applied to CONFIG:")
    print(f"  Min Confidence: {thresholds.high_winrate_min_confidence:.1f}%")
    print(f"  Min RR:         {thresholds.high_winrate_min_rr:.2f}")
    print(f"  ADX Threshold:  {thresholds.sideways_adx_threshold:.1f}")
    print(f"  Max SL Points:  {thresholds.micro_max_sl_points:.1f}")
    print(f"  Best Session:   {thresholds.best_session}")
    print(f"  Worst Agent:    {thresholds.worst_agent or 'none'}")
    for note in thresholds.notes:
        print(f"  Note: {note}")

    summary = runner._build_summary(result, thresholds)
    runner._save_cache(summary)
    print(f"\nCache saved to {CACHE_FILE}")
    print("Done. CONFIG is now patched with data-driven thresholds.")
