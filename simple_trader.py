"""
simple_trader.py — Standalone XAUUSD Signal Bot
==================================================

WHY THIS EXISTS
----------------
The ICT_V4 repo has ~10 stacked filters (confidence scoring, HTF pyramid,
regime gates, confluence categories, activity fallback timers) that
interact in ways that are hard to predict. Loosen one gate to get more
trades, and another gate silently closes instead. That's what happened:
3 trades in 10 hours -> "fixed" -> 0 trades in 10 hours.

This script throws that whole approach out. It uses ONE classic, simple,
well-understood strategy with a small number of plain conditions you can
read top to bottom. No confidence scores, no regime classification, no
HTF pyramid, no confluence categories.

STRATEGY: EMA Trend + RSI Pullback (a standard, widely-used approach)
-----------------------------------------------------------------------
1. Trend filter:   EMA50 vs EMA200 on 5m candles defines the trend.
                    Price above both, EMA50 > EMA200  -> uptrend (BUY only)
                    Price below both, EMA50 < EMA200  -> downtrend (SELL only)
2. Pullback entry:  RSI(14) dips below 45 in an uptrend (pulls back) then
                    crosses back above 45 -> buy the resumption.
                    RSI rises above 55 in a downtrend then crosses back
                    below 55 -> sell the resumption.
3. Fixed risk:      SL placed beyond the recent swing low/high (with ATR
                    buffer so it's never inside spread). TP at 1.5x risk
                    (lower RR than ICT_V4's 2.0+, by design — pullback
                    continuation trades win more often at a lower RR).
4. One open trade at a time. No duplicate-direction blocking, no cooldown
   timers, no "repeat-loss block" — if the setup reappears, it trades it.

This WILL trade more often than ICT_V4 and WILL have losses — that's
normal for any strategy. Nothing trades with 100% win rate. What this
script guarantees is that it ACTUALLY TAKES TRADES when the (simple,
visible) conditions are met, instead of going silent for 10 hours.

HOW TO RUN
----------
1. Open MetaTrader5, log in to your broker, open an XAUUSD chart.
2. pip install MetaTrader5 pandas numpy
3. python simple_trader.py
4. Trades print to console and log to trades_log.csv in this folder.

This connects to MT5 exactly like ICT_V4 does (same symbol discovery
logic), but the decision logic is completely independent and self-contained
in this one file — nothing from ICT_V4 is imported.
"""

from __future__ import annotations
import csv
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


# ── Configuration — edit these directly, no env-var indirection ──────────

SYMBOL_CANDIDATES = ["XAUUSD", "XAUUSDm", "GOLD", "XAUUSD.pro", "XAUUSD.a"]
TIMEFRAME = "M5"                 # 5-minute candles
EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14
RSI_PULLBACK_UP = 45             # RSI below this in uptrend = pullback zone
RSI_PULLBACK_DOWN = 55           # RSI above this in downtrend = pullback zone
ATR_PERIOD = 14
ATR_SL_BUFFER_MULT = 0.5         # extra buffer beyond swing low/high
MIN_SL_POINTS = 8.0              # XAUUSD: never risk less than this (clears spread)
MAX_SL_POINTS = 40.0
RISK_REWARD = 1.5                # TP = 1.5x risk (lower than ICT_V4's 2.0+ on purpose)
FIXED_LOT_SIZE = 0.01
MAX_CONCURRENT_TRADES = 1
POLL_SECONDS = 5                 # how often to check for a new closed candle
LOG_FILE = Path(__file__).resolve().parent / "trades_log.csv"


# ── Indicator helpers (no external dependency beyond pandas/numpy) ───────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ── Signal logic ───────────────────────────────────────────────────────────

@dataclass
class SimpleSignal:
    direction: str          # "BUY" or "SELL"
    entry: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: datetime


def compute_signal(df: pd.DataFrame) -> Optional[SimpleSignal]:
    """
    df must have columns: open, high, low, close, and be indexed by time,
    most recent candle LAST. Uses only the most recently CLOSED candle
    (df.iloc[-1]) — never an in-progress candle.
    """
    if len(df) < EMA_SLOW + 5:
        return None  # not enough history yet

    df = df.copy()
    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_slow"] = ema(df["close"], EMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["atr"] = atr(df, ATR_PERIOD)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["close"])
    ema_f = float(last["ema_fast"])
    ema_s = float(last["ema_slow"])
    rsi_now = float(last["rsi"])
    rsi_prev = float(prev["rsi"])
    atr_val = max(float(last["atr"]), 1e-6)

    uptrend = close > ema_f > ema_s
    downtrend = close < ema_f < ema_s

    direction = None
    reason = ""

    # Pullback-resumption logic: RSI crosses back through the threshold
    if uptrend and rsi_prev < RSI_PULLBACK_UP <= rsi_now:
        direction = "BUY"
        reason = (
            f"Uptrend (close>{EMA_FAST}EMA>{EMA_SLOW}EMA) + RSI pullback resumption "
            f"({rsi_prev:.1f} -> {rsi_now:.1f}, crossed {RSI_PULLBACK_UP})"
        )
    elif downtrend and rsi_prev > RSI_PULLBACK_DOWN >= rsi_now:
        direction = "SELL"
        reason = (
            f"Downtrend (close<{EMA_FAST}EMA<{EMA_SLOW}EMA) + RSI pullback resumption "
            f"({rsi_prev:.1f} -> {rsi_now:.1f}, crossed {RSI_PULLBACK_DOWN})"
        )

    if direction is None:
        return None

    # Stop loss: beyond recent swing low/high + ATR buffer, bounded to sane range
    lookback = df.tail(10)
    if direction == "BUY":
        swing = float(lookback["low"].min())
        risk_points = max(close - swing + atr_val * ATR_SL_BUFFER_MULT, MIN_SL_POINTS)
        risk_points = min(risk_points, MAX_SL_POINTS)
        sl = round(close - risk_points, 2)
        tp = round(close + risk_points * RISK_REWARD, 2)
    else:
        swing = float(lookback["high"].max())
        risk_points = max(swing - close + atr_val * ATR_SL_BUFFER_MULT, MIN_SL_POINTS)
        risk_points = min(risk_points, MAX_SL_POINTS)
        sl = round(close + risk_points, 2)
        tp = round(close - risk_points * RISK_REWARD, 2)

    return SimpleSignal(
        direction=direction,
        entry=round(close, 2),
        stop_loss=sl,
        take_profit=tp,
        reason=reason,
        timestamp=last.name if hasattr(last, "name") else datetime.now(timezone.utc),
    )


# ── MT5 connection helpers ────────────────────────────────────────────────

def find_symbol() -> str:
    for candidate in SYMBOL_CANDIDATES:
        info = mt5.symbol_info(candidate)
        if info is not None:
            mt5.symbol_select(candidate, True)
            return candidate
    raise RuntimeError(
        f"No matching XAUUSD symbol found among {SYMBOL_CANDIDATES}. "
        "Open the GOLD/XAUUSD chart in MT5 first."
    )


def fetch_candles(symbol: str, n: int = 500) -> pd.DataFrame:
    tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15}
    rates = mt5.copy_rates_from_pos(symbol, tf_map[TIMEFRAME], 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError("Failed to fetch candles from MT5 — check connection/login.")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def log_trade(signal: SimpleSignal):
    is_new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "direction", "entry", "stop_loss", "take_profit", "reason"])
        writer.writerow([
            signal.timestamp, signal.direction, signal.entry,
            signal.stop_loss, signal.take_profit, signal.reason,
        ])


# ── Main loop ──────────────────────────────────────────────────────────────

def run():
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed. Run: pip install MetaTrader5")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    symbol = find_symbol()
    print(f"Connected. Trading symbol: {symbol} | Timeframe: {TIMEFRAME}")
    print(f"Strategy: EMA{EMA_FAST}/EMA{EMA_SLOW} trend + RSI({RSI_PERIOD}) pullback")
    print(f"Risk: SL {MIN_SL_POINTS}-{MAX_SL_POINTS} pts, RR {RISK_REWARD}x, lot {FIXED_LOT_SIZE}")
    print("-" * 70)

    last_seen_candle_time = None
    open_positions = 0

    while True:
        try:
            df = fetch_candles(symbol, n=EMA_SLOW + 50)
            latest_closed_time = df.index[-1]

            # only evaluate once per NEW closed candle (avoid repeat signals on same bar)
            if latest_closed_time != last_seen_candle_time:
                last_seen_candle_time = latest_closed_time

                # check open positions count for this symbol
                positions = mt5.positions_get(symbol=symbol)
                open_positions = len(positions) if positions else 0

                if open_positions < MAX_CONCURRENT_TRADES:
                    signal = compute_signal(df)
                    if signal:
                        print(f"[{signal.timestamp}] SIGNAL: {signal.direction} @ {signal.entry} "
                              f"SL={signal.stop_loss} TP={signal.take_profit}")
                        print(f"   Reason: {signal.reason}")
                        log_trade(signal)

                        # Send order to MT5
                        order_type = mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL
                        tick = mt5.symbol_info_tick(symbol)
                        price = tick.ask if signal.direction == "BUY" else tick.bid
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": FIXED_LOT_SIZE,
                            "type": order_type,
                            "price": price,
                            "sl": signal.stop_loss,
                            "tp": signal.take_profit,
                            "deviation": 20,
                            "magic": 778899,
                            "comment": "simple_trader EMA/RSI",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result = mt5.order_send(request)
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print(f"   ORDER FAILED: {result.retcode} {result.comment}")
                        else:
                            print(f"   ORDER PLACED: ticket={result.order}")
                    else:
                        print(f"[{latest_closed_time}] No setup this candle (trend/RSI conditions not met)")
                else:
                    print(f"[{latest_closed_time}] Skipping — {open_positions} trade(s) already open")

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"Error: {e}. Retrying in 10s...")
            time.sleep(10)

    mt5.shutdown()


if __name__ == "__main__":
    run()
