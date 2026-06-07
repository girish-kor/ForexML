"""Data structures and schemas."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class TICK:
    """Tick data from market."""
    instrument: str
    bid: float
    ask: float
    spread: float
    volume: int
    timestamp_us: int


@dataclass
class OHLCV_BAR:
    """OHLCV bar data."""
    instrument: str
    timeframe: int
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    spread_avg: float
    data_source: str


@dataclass
class MARKET_STRUCTURE_SHIFT:
    """Market structure shift (BOS/CHOCH)."""
    instrument: str
    timeframe: int
    structure_type: str
    break_price: float
    break_bar_id: int
    direction: str
    confirmation_bars: int
    displaced_high: float
    displaced_low: float
    confirmed: bool


@dataclass
class ORDER_BLOCK:
    """Order block structure."""
    instrument: str
    timeframe: int
    zone_high: float
    zone_low: float
    strength_score: float
    created_bar_id: int
    is_mitigated: bool = False
    mitigation_pct: float = 0.0


@dataclass
class FAIR_VALUE_GAP:
    """Fair value gap."""
    instrument: str
    timeframe: int
    gap_high: float
    gap_low: float
    gap_size_pips: float
    created_bar_id: int
    is_filled: bool = False
    fill_pct: float = 0.0


@dataclass
class LIQUIDITY_POOL:
    """Liquidity pool level."""
    instrument: str
    timeframe: int
    level_price: float
    touch_count: int
    created_bar_id: int
    is_swept: bool = False
    swept_at: Optional[int] = None


@dataclass
class FEATURE_SNAPSHOT:
    """Feature vector snapshot for ML."""
    instrument: str
    bar_id: int
    feature_set_version: str
    feature_vector: np.ndarray


@dataclass
class MODEL_PREDICTION:
    """ML model prediction."""
    bar_id: int
    instrument: str
    prob_long: float
    prob_short: float
    prob_neutral: float
    predicted_direction: str
    confidence_score: float
    calibration_score: float
    bar_timestamp: int


@dataclass
class TRADING_SIGNAL:
    """Trading signal."""
    signal_id: str
    instrument: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_ratio: float
    signal_strength: float
    confluence_score: float
    confluence_tags: List[str]
    status: str = 'PENDING'
    created_at: int = None
    filter_flags: List[str] = field(default_factory=list)


@dataclass
class POSITION_SIZING:
    """Position sizing calculation."""
    signal_id: str
    account_balance: float
    risk_amount: float
    lot_size: float
    margin_required: float
    leverage_used: float


@dataclass
class ORDER:
    """Order object."""
    order_type: str
    direction: str
    volume: float
    stop_loss: float
    take_profit: float
    status: str = 'PENDING'
    broker_order_id: Optional[int] = None
    executed_price: Optional[float] = None
    slippage_pips: Optional[float] = None
    filled_at: Optional[int] = None
    rejection_reason: Optional[str] = None


@dataclass
class TRADE:
    """Trade record."""
    trade_id: str
    signal_id: str
    instrument: str
    direction: str
    open_price: float
    volume: float
    stop_loss: float
    take_profit: float
    open_time: int
    close_price: Optional[float] = None
    close_time: Optional[int] = None
    close_reason: Optional[str] = None
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    pnl_pips: Optional[float] = None
    mae_pips: Optional[float] = None
    mfe_pips: Optional[float] = None


@dataclass
class PERFORMANCE_SNAPSHOT:
    """Performance metrics snapshot."""
    period: str
    model_version: str
    win_rate: float
    profit_factor: float
    expected_value: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration: int
    avg_win_pips: float
    avg_loss_pips: float
    avg_rr_realized: float


@dataclass
class MODEL_DRIFT_METRICS:
    """Model drift detection metrics."""
    model_version: str
    instrument: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    live_accuracy: float
    val_accuracy: float
    accuracy_degradation_pct: float
    live_log_loss: float
    drift_alert_triggered: bool = False
    alert_severity: Optional[str] = None


@dataclass
class CROSS_VALIDATION_FOLD:
    """Cross-validation fold indices."""
    fold_number: int
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]


@dataclass
class MODEL_VERSION:
    """Model version metadata."""
    version_id: str
    artifact_path: str
    serialization_fmt: str
    eval_metrics: dict
    status: str = 'STAGING'
    deployed_at: Optional[int] = None
    retired_at: Optional[int] = None


@dataclass
class PIPELINE_RUN:
    """Pipeline run metadata."""
    run_id: str
    pipeline_name: str
    triggered_by: str
    git_commit_hash: str
    status: str = 'RUNNING'
    started_at: int = None
    finished_at: Optional[int] = None
    duration_sec: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
