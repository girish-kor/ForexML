"""Trading signal generation and filtering."""

import time
import logging
from typing import List, Tuple

from schemas import (
    MODEL_PREDICTION,
    ORDER_BLOCK,
    FAIR_VALUE_GAP,
    MARKET_STRUCTURE_SHIFT,
    LIQUIDITY_POOL,
    TRADING_SIGNAL
)

logger = logging.getLogger(__name__)


class SignalGeneration:
    """Generate trading signals from predictions."""

    def __init__(self, confidence_threshold: float = 0.6, confluence_min: float = 5.0):
        self.confidence_threshold = confidence_threshold
        self.confluence_min = confluence_min
        self.signal_id_counter = 0

    def check_confidence_threshold(self, pred: MODEL_PREDICTION) -> bool:
        """Check if prediction meets confidence threshold."""
        return (
            pred.confidence_score >= self.confidence_threshold 
            and pred.predicted_direction != 'NEUTRAL'
        )

    def score_confluence(
        self,
        pred: MODEL_PREDICTION,
        ob: ORDER_BLOCK,
        fvg: FAIR_VALUE_GAP,
        mss: MARKET_STRUCTURE_SHIFT,
        liq: LIQUIDITY_POOL
    ) -> Tuple[float, List[str]]:
        """Score signal confluence."""
        score = 0.0
        tags = []

        if ob and not ob.is_mitigated:
            score += 3
            tags.append('OB_UNMITIGATED')
        if fvg:
            score += 2
            tags.append('FVG_PRESENT')
        if liq and liq.is_swept:
            score += 2
            tags.append('LIQ_SWEPT')
        if mss and mss.confirmed:
            score += 3
            tags.append('MSS_CONFIRMED')

        return score, tags

    def build_trading_signal(
        self,
        pred: MODEL_PREDICTION,
        ob: ORDER_BLOCK,
        confluence_score: float,
        confluence_tags: List[str],
        current_price: float
    ) -> TRADING_SIGNAL:
        """Build trading signal."""
        self.signal_id_counter += 1
        signal_id = f"SIG_{int(time.time())}_{self.signal_id_counter}"

        entry_price = ob.zone_high if pred.predicted_direction == 'LONG' else ob.zone_low
        stop_loss = ob.zone_low if pred.predicted_direction == 'LONG' else ob.zone_high

        rr_ratio = 1.5
        tp1_offset = (entry_price - stop_loss) * rr_ratio * 0.5
        tp1 = (
            entry_price + tp1_offset
            if pred.predicted_direction == 'LONG'
            else entry_price - tp1_offset
        )

        tp2_offset = (entry_price - stop_loss) * rr_ratio
        tp2 = (
            entry_price + tp2_offset
            if pred.predicted_direction == 'LONG'
            else entry_price - tp2_offset
        )

        return TRADING_SIGNAL(
            signal_id=signal_id,
            instrument=pred.instrument,
            direction=pred.predicted_direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward_ratio=rr_ratio,
            signal_strength=pred.confidence_score,
            confluence_score=confluence_score,
            confluence_tags=confluence_tags,
            status='PENDING',
            created_at=int(time.time()),
            filter_flags=[]
        )

    def apply_filter_flags(
        self,
        signal: TRADING_SIGNAL,
        news_suppressed: bool = False,
        max_positions: int = 5,
        open_positions: int = 0,
        drawdown_exceeded: bool = False
    ) -> TRADING_SIGNAL:
        """Apply filter flags to signal."""
        flags = []
        if news_suppressed:
            flags.append('NEWS_SUPPRESSED')
        if open_positions >= max_positions:
            flags.append('MAX_OPEN_POSITIONS')
        if drawdown_exceeded:
            flags.append('DRAWDOWN_LIMIT_HIT')

        signal.filter_flags = flags
        if flags:
            signal.status = 'FILTERED'

        return signal
