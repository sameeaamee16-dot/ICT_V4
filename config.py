from __future__ import annotations
"""
config.py — ICT_V4 BALANCED
============================
PROBLEM FIXED: Tool ran 4 hours with zero entries because filters were
stacked too tightly. Every gate was at maximum strictness simultaneously,
making it impossible for any signal to pass all checks.

KEY CHANGES vs the over-tight version:

1. high_winrate_mode: True → False
   HIGH_WINRATE_MODE was demanding 74% confidence + HTF alignment +
   entry score + timing score ALL simultaneously. With no trade history
   to calibrate against, the confidence calibrator was also blocking
   trades. Disabled for now — re-enable after 30+ trades are recorded.

2. high_winrate_min_confidence: 74.0 → 68.0
   74% is the right target eventually, but with zero trade history and
   fresh filters the signal engine rarely reaches 74%. 68% is the
   realistic floor that still filters weak signals.

3. minimum_activity_minutes: 25 → 8
   The fallback was set to 25 minutes but FilterEngine was also blocking
   the fallback. Combined = no fallback ever fires. 8 minutes is enough
   to avoid overtrading while ensuring at least one entry per session.

4. minimum_activity_enabled: True (kept)
   Fallback is still enabled so the bot doesn't sit idle all day.

5. sideways_adx_threshold: 18.0 → 16.0
   18 ADX is too aggressive for 1m XAUUSD. Many valid ICT setups form
   with ADX between 16–18. Lowered back to 16.

6. htf_min_aligned: 2 → 1
   Requiring 2 HTFs to agree was blocking most signals because 1h
   often disagrees with 5m direction on ranging days. 1 HTF is enough.

7. agent_max_consecutive_losses: 3 → 5
   With zero history, the agent guard was potentially blocking all agents
   before they built any track record. Loosened to 5.

8. min_agent_trades_for_guard: 10 → 20
   Agent guard now waits for 20 trades before judging an agent — avoids
   false blocks on new/fresh agents.

9. micro_sl_points: 5.0 → 6.0 and micro_tp_points: 14.0 → 15.0
   Slightly wider to reduce noise-stop-outs on 1m XAUUSD.

10. FilterEngine off-session rule:
    Set MINIMUM_ACTIVITY_ENABLED=true and MINIMUM_ACTIVITY_MINUTES=8
    in your .env or PowerShell so the fallback fires regularly.

HOW TO RESTORE HIGH WIN-RATE MODE AFTER 30+ TRADES:
    Set high_winrate_mode = True
    Set high_winrate_min_confidence = 74.0
    Set minimum_activity_minutes = 25
    Set htf_min_aligned = 2
"""

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
import os

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"


@dataclass(frozen=True)
class TimeframeConfig:
    primary: str = "1m"
    execution: str = "1m"
    confluence: List[str] = field(default_factory=lambda: ["5m", "15m", "1h"])
    all: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])


@dataclass
class RiskConfig:
    account_equity: float = 100_000.0
    risk_per_trade_pct: float = 0.5
    max_daily_drawdown_pct: float = 3.0
    max_concurrent_trades: int = 1
    max_spread_points: float = 55.0
    min_rr: float = 1.8

    partial_tp_ratio: float = 0.0       # disabled — too small on 0.01 lot
    partial_tp_at_r: float = 2.0
    break_even_at_r: float = 1.8
    trail_after_r: float = 2.8
    atr_sl_mult: float = 0.85

    fixed_lot_size: float = 0.01
    use_micro_scalp_exits: bool = True
    micro_min_rr: float = 1.8          # was 2.0 — allows more valid setups
    micro_sl_points: float = 6.0       # was 5.0 — slightly wider for 1m noise
    micro_min_sl_points: float = 3.0
    micro_max_sl_points: float = 14.0  # was 12 — more room
    micro_tp_points: float = 15.0      # was 14
    micro_min_tp_points: float = 10.0
    micro_max_tp_points: float = 35.0  # was 30

    fixed_profit_target_usd: float = 0.0  # disabled

    # Agent guard — loosened while building history
    min_agent_trades_for_guard: int = 20  # was 10 — wait longer before judging
    agent_min_winrate_pct: float = 40.0   # was 45 — more forgiving early on
    agent_max_recent_loss: float = -25.0  # was -15
    agent_recent_window: int = 20         # was 15
    agent_max_consecutive_losses: int = 5  # was 3 — don't block too early
    agent_loss_window: int = 10
    agent_max_losses_in_window: int = 7   # was 6

    min_lot: float = 0.01
    max_lot: float = 10.0
    lot_step: float = 0.01

    calibration_min_samples: int = 20     # was 15 — don't calibrate too early
    calibration_warn_winrate_pct: float = 40.0  # was 45

    # HIGH WIN-RATE MODE — DISABLED until 30+ trades recorded
    # Re-enable by setting high_winrate_mode = True after you have history
    high_winrate_mode: bool = False       # was True — MAIN FIX
    target_winrate_pct: float = 60.0
    high_winrate_min_confidence: float = 68.0   # was 74.0
    high_winrate_min_rr: float = 1.8            # was 2.0
    high_winrate_min_entry_score: float = 60.0  # was 65.0
    high_winrate_min_timing_score: float = 58.0 # was 62.0
    mtf_alignment_floor: float = 0.40           # was 0.50
    htf_min_aligned: int = 1                    # was 2 — only need 1 HTF
    protect_win_streak: int = 8
    protect_streak_min_confidence: float = 75.0
    protect_streak_min_rr: float = 2.0
    protect_streak_min_entry_score: float = 65.0


@dataclass(frozen=True)
class BacktestCostConfig:
    default_spread_points: float = 25.0
    slippage_points: float = 3.0
    commission_per_lot_round_turn: float = 7.0
    spread_column: str = "spread"


@dataclass(frozen=True)
class AssetProfile:
    name: str
    symbols: tuple
    contract_size: float
    max_spread_points: float
    min_rr: float
    min_confidence: float
    atr_sl_mult: float
    htf_bias_lock: bool = True
    max_same_setup_open: int = 1
    duplicate_entry_atr: float = 0.55


@dataclass(frozen=True)
class IctConfig:
    swing_left: int = 3
    swing_right: int = 2
    equal_level_atr_tolerance: float = 0.18
    displacement_atr_mult: float = 0.9
    fvg_min_atr: float = 0.08
    ob_lookback: int = 20
    mitigation_lookback: int = 80
    premium_discount_lookback: int = 120
    inducement_lookback: int = 45
    min_confirmations: int = 3
    min_confidence: float = 60.0        # was 62 — slightly more forgiving
    sideways_adx_threshold: float = 16.0  # was 18 — too aggressive for 1m
    low_atr_percentile: float = 0.10    # was 0.12


@dataclass(frozen=True)
class SessionConfig:
    timezone: str = "UTC"
    kill_zones: Dict[str, tuple] = field(
        default_factory=lambda: {
            "london": ("06:30", "10:30"),
            "silver_bullet": ("10:00", "11:00"),
            "new_york_am": ("12:00", "16:30"),
            "new_york_pm": ("17:30", "20:30"),
            "asia": ("00:00", "03:30"),
        }
    )


@dataclass(frozen=True)
class DataConfig:
    symbol: str = field(default_factory=lambda: os.getenv("TRADING_SYMBOL", "XAUUSD"))
    tradingview_symbol: str = field(default_factory=lambda: os.getenv("TRADINGVIEW_SYMBOL", "OANDA:XAUUSD"))
    mt5_symbol_candidates: List[str] = field(
        default_factory=lambda: [
            item.strip()
            for item in os.getenv(
                "MT5_SYMBOL_CANDIDATES",
                "XAUUSD,XAUUSDm,GOLD,XAUUSD.pro,GOLDmicro,XAUUSD.a",
            ).split(",")
            if item.strip()
        ]
    )
    news_blackout_utc: str = field(default_factory=lambda: os.getenv("NEWS_BLACKOUT_UTC", ""))
    history_bars: int = field(default_factory=lambda: int(os.getenv("HISTORY_BARS", "1500")))
    poll_seconds: float = field(default_factory=lambda: float(os.getenv("POLL_SECONDS", "0.25")))
    closed_candle_refresh_seconds: float = field(
        default_factory=lambda: float(os.getenv("CLOSED_CANDLE_REFRESH_SECONDS", "1.0"))
    )
    aggressive_intrabar_mode: bool = field(
        default_factory=lambda: os.getenv("AGGRESSIVE_INTRABAR_MODE", "true").lower() in {"1", "true", "yes", "on"}
    )
    execution_countdown_seconds: int = field(
        default_factory=lambda: int(os.getenv("EXECUTION_COUNTDOWN_SECONDS", "3"))
    )
    execution_countdown_mode: str = field(
        default_factory=lambda: os.getenv("EXECUTION_COUNTDOWN_MODE", "visual").lower()
    )
    dashboard_refresh_ms: int = field(default_factory=lambda: int(os.getenv("DASHBOARD_REFRESH_MS", "500")))
    # MAIN FIX: 8 minutes idle before fallback (was 25 — too long when filters block everything)
    minimum_activity_minutes: int = field(
        default_factory=lambda: int(os.getenv("MINIMUM_ACTIVITY_MINUTES", "8"))
    )
    minimum_activity_enabled: bool = field(
        default_factory=lambda: os.getenv("MINIMUM_ACTIVITY_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    )


@dataclass(frozen=True)
class MySQLConfig:
    host: str = field(default_factory=lambda: os.getenv("MYSQL_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("MYSQL_PORT", "3307")))
    user: str = field(default_factory=lambda: os.getenv("MYSQL_USER", "root"))
    password: str = field(default_factory=lambda: os.getenv("MYSQL_PASSWORD", "Admin"))
    database: str = field(default_factory=lambda: os.getenv("MYSQL_DATABASE", "ict"))


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    timeframes: TimeframeConfig = field(default_factory=TimeframeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest_costs: BacktestCostConfig = field(default_factory=BacktestCostConfig)
    ict: IctConfig = field(default_factory=IctConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    mysql: MySQLConfig = field(default_factory=MySQLConfig)
    asset_profiles: Dict[str, AssetProfile] = field(
        default_factory=lambda: {
            "XAU": AssetProfile(
                name="XAU",
                symbols=("XAU", "GOLD"),
                contract_size=100.0,
                max_spread_points=55.0,
                min_rr=1.8,
                min_confidence=60.0,
                atr_sl_mult=1.0,
                htf_bias_lock=True,
                max_same_setup_open=1,
                duplicate_entry_atr=0.6,
            ),
            "DEFAULT": AssetProfile(
                name="DEFAULT",
                symbols=(),
                contract_size=100.0,
                max_spread_points=55.0,
                min_rr=1.8,
                min_confidence=60.0,
                atr_sl_mult=1.0,
            ),
        }
    )


CONFIG = AppConfig()


def asset_profile(symbol: str | None = None) -> AssetProfile:
    target = (symbol or CONFIG.data.symbol).upper()
    for profile in CONFIG.asset_profiles.values():
        if profile.name == "DEFAULT":
            continue
        if any(term in target for term in profile.symbols):
            return profile
    return CONFIG.asset_profiles["DEFAULT"]


def active_news_blackout(now: Optional[datetime] = None) -> str | None:
    raw = CONFIG.data.news_blackout_utc.strip()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    if raw:
        for window in raw.split(";"):
            if "/" not in window:
                continue
            start_raw, end_raw = [part.strip() for part in window.split("/", 1)]
            try:
                start = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                end = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            if start <= now <= end:
                return f"{start.isoformat()} to {end.isoformat()}"

    csv_path = os.getenv("ECONOMIC_NEWS_CSV", "").strip()
    if csv_path:
        before = int(os.getenv("NEWS_BLACKOUT_BEFORE_MIN", "20"))
        after = int(os.getenv("NEWS_BLACKOUT_AFTER_MIN", "20"))
        try:
            from datetime import timedelta
            for line in Path(csv_path).read_text(encoding="utf-8").splitlines():
                if not line.strip() or line.lower().startswith("time"):
                    continue
                parts = [part.strip() for part in line.split(",")]
                try:
                    event_time = datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
                except (IndexError, ValueError):
                    continue
                impact = parts[2].lower() if len(parts) > 2 else "high"
                if impact not in {"high", "red", "major"}:
                    continue
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)
                start = event_time - timedelta(minutes=before)
                end = event_time + timedelta(minutes=after)
                if start <= now <= end:
                    title = parts[1] if len(parts) > 1 else "economic news"
                    return f"{title}: {start.isoformat()} to {end.isoformat()}"
        except OSError:
            return None

    return None
