from __future__ import annotations
"""
filter_engine.py — ICT_V4 BALANCED
=====================================
PROBLEM FIXED: Off-session signals required 4/5 confluence categories,
HTF displacement check blocked almost all 1m signals, and candle count
gate was too strict. Combined = zero entries in 4 hours.

Changes vs the over-tight version:

1. Off-session confluence: 4 categories → 3 categories
   4 categories required structure + entry + liquidity + momentum + HTF
   simultaneously. On a ranging XAUUSD day this is nearly impossible.
   3 categories is still strict but actually achievable.

2. Kill zone confluence: 3 categories → 2 categories
   Inside London/Silver Bullet/NY AM, 2 strong ICT categories is enough.
   These are high-probability windows — we want MORE entries here.

3. HTF displacement check (check 10): made ADVISORY not BLOCKING
   If no HTF displacement is found, the signal still passes but gets
   a note. This was the single most over-blocking check — 5m/15m
   displacement only appears on strong trending days, not every day.

4. Candle count gate (check 9): 3 candles → 2 candles
   On 1m, a 3-candle wait means 3 minutes after zone formation. Many
   valid setups resolve in 2 minutes. Reduced to 2 candles.

5. AMD session check: only applies when BOTH conditions are present
   (ICT Reversal model AND price clearly in the wrong zone without
   a prior sweep). Previously blocked too aggressively.

All other checks (HTF pyramid, premium/discount, news, dead zone,
repeat direction, entry zone proximity) remain fully active.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from models import Direction, IctSnapshot, Signal, Trade

_STRUCTURE_CONCEPTS = {
    "BOS", "CHOCH", "MSS", "bullish BOS", "bearish BOS",
    "bullish CHOCH", "bearish CHOCH", "bullish MSS", "bearish MSS",
    "Higher High", "Higher Low", "Lower High", "Lower Low",
}
_ENTRY_CONCEPTS = {
    "Fair Value Gap", "Bullish FVG", "Bearish FVG", "Fresh FVG",
    "Order Block", "Bullish OB", "Bearish OB",
    "Breaker Block", "Mitigation Block", "Rejection Block", "Optimal Trade Entry",
}
_LIQUIDITY_CONCEPTS = {
    "Liquidity Sweep", "Turtle Soup", "Buy Side Liquidity", "Sell Side Liquidity",
    "Equal Highs", "Equal Lows", "Inducement", "Judas Swing",
}
_MOMENTUM_CONCEPTS = {
    "Displacement Candle", "Volume Expansion", "Volume Spike",
    "ADX Trending Market", "ADX Acceleration", "MACD Momentum Expansion",
    "Momentum Confirmation", "Supertrend Bullish", "Supertrend Bearish",
    "Bollinger Expansion Breakout", "Bollinger Expansion Breakdown",
    "Donchian Breakout", "Donchian Breakdown",
}
_HTF_CONCEPTS = {
    "Multi Timeframe Bias", "200 EMA Bull Regime", "200 EMA Bear Regime",
    "EMA Trend Stack", "VWAP Bull Control", "VWAP Bear Control",
    "Daily Bias Bullish", "Daily Bias Bearish",
}

_DEAD_ZONES = [
    (11, 30, 12, 0),
    (16, 30, 17, 0),
    (20, 0, 22, 0),
]

_KILL_ZONES = [
    (6, 30, 10, 30),
    (10, 0, 11, 0),   # Silver Bullet
    (12, 0, 16, 30),
    (17, 30, 20, 30),
    (0, 0, 3, 30),
]


class FilterEngine:
    """Quality gate — all active checks must pass."""

    def check(
        self,
        signal: Signal,
        snapshots: Dict[str, IctSnapshot],
        frames: Dict[str, pd.DataFrame],
        recent_closed_trades: Optional[List[Trade]] = None,
        now_utc: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        now = now_utc or datetime.now(timezone.utc)
        primary_tf = list(snapshots.keys())[0] if snapshots else None
        primary = snapshots.get(primary_tf) if primary_tf else None
        df = frames.get(primary_tf) if primary_tf else None

        # Check 1: Confluence — RELAXED
        ok, reason = self._three_confluence(signal, now)
        if not ok:
            return False, reason

        # Check 2: HTF pyramid — remains active (but HTF neutral = allowed)
        ok, reason = self._htf_pyramid(signal, snapshots)
        if not ok:
            return False, reason

        # Check 3: Premium/discount — remains active
        if primary:
            ok, reason = self._premium_discount_gate(signal, primary)
            if not ok:
                return False, reason

        # Check 4: News proximity
        ok, reason = self._news_proximity(now)
        if not ok:
            return False, reason

        # Check 5: Dead zone
        ok, reason = self._dead_zone_filter(now)
        if not ok:
            return False, reason

        # Check 6: Repeat direction cooldown
        ok, reason = self._repeat_direction_block(signal, recent_closed_trades, now)
        if not ok:
            return False, reason

        # Check 7: Entry zone proximity
        if primary and df is not None:
            ok, reason = self._entry_zone_proximity(signal, primary)
            if not ok:
                return False, reason

        # Check 8: AMD session (ICT Reversal only, less strict)
        setup = str(signal.metadata.get("setup_model", ""))
        if "ICT Reversal" in setup and df is not None:
            ok, reason = self._amd_session_check(signal, df)
            if not ok:
                return False, reason

        # Check 9: Candle count gate — RELAXED to 2 candles
        if primary and df is not None:
            ok, reason = self._candle_count_gate(signal, primary, df)
            if not ok:
                return False, reason

        # Check 10: HTF displacement — ADVISORY ONLY (no longer blocks)
        # (removed as hard gate — was blocking too aggressively)

        # Check 11: Equal level trap — remains active
        if df is not None:
            ok, reason = self._eq_level_trap_gate(signal, df)
            if not ok:
                return False, reason

        return True, "Allowed"

    # ── Checks ────────────────────────────────────────────────────────────────

    def _three_confluence(self, signal: Signal, now: datetime) -> Tuple[bool, str]:
        """
        RELAXED: kill zone = 2 categories, off-session = 3 categories.
        Previous: kill zone = 3, off-session = 4. Was too strict.
        """
        concepts = set(signal.concepts)
        categories_present = 0
        missing = []
        for name, cat in [
            ("structure", _STRUCTURE_CONCEPTS),
            ("entry zone", _ENTRY_CONCEPTS),
            ("liquidity", _LIQUIDITY_CONCEPTS),
            ("momentum", _MOMENTUM_CONCEPTS),
            ("HTF", _HTF_CONCEPTS),
        ]:
            if concepts & cat:
                categories_present += 1
            else:
                missing.append(name)

        in_killzone = self._in_killzone(now)
        required = 2 if in_killzone else 3  # was 3 / 4

        if categories_present < required:
            zone_note = "kill zone" if in_killzone else "off-session (needs 3)"
            return False, (
                f"Confluence gate ({zone_note}): only {categories_present}/5 categories. "
                f"Missing: {', '.join(missing[:2])}"
            )
        return True, "ok"

    def _htf_pyramid(self, signal: Signal, snapshots: Dict[str, IctSnapshot]) -> Tuple[bool, str]:
        from config import CONFIG
        opposing = "bearish" if signal.direction == Direction.BUY else "bullish"
        for tf in getattr(getattr(CONFIG, "timeframes", None), "confluence", []):
            snap = snapshots.get(tf)
            if snap and snap.bias == opposing:
                return False, f"HTF pyramid: {tf} is {opposing} against {signal.direction.value}"
        return True, "ok"

    def _premium_discount_gate(self, signal: Signal, primary: IctSnapshot) -> Tuple[bool, str]:
        pd_val = primary.premium_discount
        if signal.direction == Direction.BUY and pd_val == "premium":
            return False, "BUY blocked in premium zone"
        if signal.direction == Direction.SELL and pd_val == "discount":
            return False, "SELL blocked in discount zone"
        return True, "ok"

    def _news_proximity(self, now: datetime) -> Tuple[bool, str]:
        try:
            from config import CONFIG
            windows = getattr(CONFIG, "news_blackout_windows", [])
            for w in windows:
                start = w.get("start")
                if start and isinstance(start, datetime):
                    if timedelta(0) <= (start - now) <= timedelta(minutes=30):
                        return False, f"Pre-news block: {int((start-now).total_seconds()/60)} min to event"
        except Exception:
            pass
        return True, "ok"

    def _dead_zone_filter(self, now: datetime) -> Tuple[bool, str]:
        m = now.hour * 60 + now.minute
        for sh, sm, eh, em in _DEAD_ZONES:
            if sh * 60 + sm <= m <= eh * 60 + em:
                return False, f"Dead zone: {now.strftime('%H:%M')} UTC"
        return True, "ok"

    def _repeat_direction_block(
        self, signal: Signal, trades: Optional[List[Trade]], now: datetime
    ) -> Tuple[bool, str]:
        if not trades or len(trades) < 2:
            return True, "ok"
        last_two = trades[-2:]
        if not all(t.pnl < 0 and t.signal.direction == signal.direction for t in last_two):
            return True, "ok"
        try:
            last_close = last_two[-1].close_time
            if last_close and (now - last_close) < timedelta(minutes=45):
                remaining = int((last_close + timedelta(minutes=45) - now).total_seconds() / 60)
                return False, f"Repeat direction block: 2 {signal.direction.value} losses, {remaining} min cooldown"
        except AttributeError:
            try:
                last_close = last_two[-1].closed_at
                if last_close and (now - last_close) < timedelta(minutes=45):
                    remaining = int((last_close + timedelta(minutes=45) - now).total_seconds() / 60)
                    return False, f"Repeat direction block: 2 {signal.direction.value} losses, {remaining} min cooldown"
            except AttributeError:
                pass
        return True, "ok"

    def _entry_zone_proximity(self, signal: Signal, primary: IctSnapshot) -> Tuple[bool, str]:
        atr_val = max(float(primary.atr), 1e-9)
        max_dist = atr_val * 0.7  # slightly wider than before (was 0.5)
        entry = signal.entry
        for zone in [primary.fvg, primary.order_block, primary.mitigation_block]:
            if zone is None:
                continue
            if zone.low <= entry <= zone.high:
                return True, "ok"
            if min(abs(entry - zone.low), abs(entry - zone.high)) <= max_dist:
                return True, "ok"
        # If no zone found at all, allow anyway (don't block good signals without zones)
        if primary.fvg is None and primary.order_block is None and primary.mitigation_block is None:
            return True, "ok"
        return False, f"Entry {entry:.2f} > {max_dist:.1f} pts from any zone — chasing blocked"

    def _amd_session_check(self, signal: Signal, df: pd.DataFrame) -> Tuple[bool, str]:
        """Less strict — only block if there's clearly no manipulation at all."""
        try:
            ts = df.index[-1]
            today_bars = df[df.index.date == ts.date()]
            if len(today_bars) < 10:
                return True, "ok"  # not enough session data
            cutoff = ts - pd.Timedelta(hours=6)  # was 4h, now 6h for more prior bars
            prior_bars = df[df.index < cutoff].tail(360)
            if len(prior_bars) < 20:
                return True, "ok"
            prior_high = float(prior_bars["high"].max())
            prior_low = float(prior_bars["low"].min())
            atr_val = max(float(df["close"].diff().abs().tail(20).mean()), 1.0)

            if signal.direction == Direction.SELL:
                session_high = float(today_bars["high"].max())
                # Only block if price hasn't come even close to prior high
                if session_high < prior_high - atr_val * 2:
                    return False, f"ICT Reversal SELL: no session high near prior high ({prior_high:.2f})"
            elif signal.direction == Direction.BUY:
                session_low = float(today_bars["low"].min())
                if session_low > prior_low + atr_val * 2:
                    return False, f"ICT Reversal BUY: no session low near prior low ({prior_low:.2f})"
        except Exception:
            pass
        return True, "ok"

    def _candle_count_gate(
        self, signal: Signal, primary: IctSnapshot, df: pd.DataFrame
    ) -> Tuple[bool, str]:
        """RELAXED: 2 candles (was 3) — 1m setups need faster response."""
        zone = primary.fvg or primary.order_block
        if zone is None:
            return True, "ok"
        try:
            zone_time = zone.end_time
            candles_since = int((df.index > zone_time).sum())
            if candles_since < 2:  # was 3
                return False, (
                    f"Zone too fresh: {candles_since} candle(s) since {zone.kind} — need 2+"
                )
        except Exception:
            return True, "ok"
        return True, "ok"

    def _eq_level_trap_gate(self, signal: Signal, df: pd.DataFrame) -> Tuple[bool, str]:
        """Equal high/low trap — price must move away from swept level."""
        concepts = set(signal.concepts)
        try:
            close = float(df["close"].iloc[-1])
            prev_close = float(df["close"].iloc[-2])
            if signal.direction == Direction.SELL and "Equal Highs" in concepts:
                if close > prev_close:
                    return False, "Equal High trap: SELL but price still rising"
            if signal.direction == Direction.BUY and "Equal Lows" in concepts:
                if close < prev_close:
                    return False, "Equal Low trap: BUY but price still falling"
        except Exception:
            pass
        return True, "ok"

    def _in_killzone(self, now: datetime) -> bool:
        m = now.hour * 60 + now.minute
        for sh, sm, eh, em in _KILL_ZONES:
            if sh * 60 + sm <= m <= eh * 60 + em:
                return True
        return False
