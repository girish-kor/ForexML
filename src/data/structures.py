"""Technical analysis and SMC structure detection."""

import logging
from typing import Optional, List

try:
    import numpy as np
except ImportError:
    np = None

from schemas import (
    OHLCV_BAR,
    MARKET_STRUCTURE_SHIFT,
    ORDER_BLOCK,
    FAIR_VALUE_GAP,
    LIQUIDITY_POOL
)

logger = logging.getLogger(__name__)


class SMCStructureDetection:
    """Smart Money Concepts structure detection."""

    def __init__(self):
        self.mss_records = {}
        self.ob_records = {}
        self.fvg_records = {}
        self.liq_records = {}

    def detect_market_structure_shift(
        self, bars: List[OHLCV_BAR], instrument: str, timeframe: int
    ) -> Optional[MARKET_STRUCTURE_SHIFT]:
        """Detect market structure shift (BOS/CHOCH)."""
        if len(bars) < 3 or np is None:
            return None

        mss = MARKET_STRUCTURE_SHIFT(
            instrument=instrument,
            timeframe=timeframe,
            structure_type='BOS',
            break_price=bars[-1].high,
            break_bar_id=len(bars) - 1,
            direction='UP',
            confirmation_bars=0,
            displaced_high=bars[-1].high,
            displaced_low=bars[-1].low,
            confirmed=False
        )
        key = (instrument, timeframe)
        if key not in self.mss_records:
            self.mss_records[key] = []
        self.mss_records[key].append(mss)
        return mss

    def detect_order_block(
        self, bars: List[OHLCV_BAR], instrument: str, timeframe: int
    ) -> Optional[ORDER_BLOCK]:
        """Detect order block."""
        if len(bars) < 2 or np is None:
            return None

        ob = ORDER_BLOCK(
            instrument=instrument,
            timeframe=timeframe,
            zone_high=max(bars[-2].high, bars[-1].high),
            zone_low=min(bars[-2].low, bars[-1].low),
            strength_score=0.75,
            created_bar_id=len(bars) - 1
        )
        key = (instrument, timeframe)
        if key not in self.ob_records:
            self.ob_records[key] = []
        self.ob_records[key].append(ob)
        return ob

    def detect_fair_value_gap(
        self, bars: List[OHLCV_BAR], instrument: str, timeframe: int
    ) -> Optional[FAIR_VALUE_GAP]:
        """Detect fair value gap."""
        if len(bars) < 3 or np is None:
            return None

        gap_high = min(bars[-3].high, bars[-2].high)
        gap_low = max(bars[-3].low, bars[-2].low)
        if gap_high > gap_low:
            fvg = FAIR_VALUE_GAP(
                instrument=instrument,
                timeframe=timeframe,
                gap_high=gap_high,
                gap_low=gap_low,
                gap_size_pips=(gap_high - gap_low) * 10000,
                created_bar_id=len(bars) - 1
            )
            key = (instrument, timeframe)
            if key not in self.fvg_records:
                self.fvg_records[key] = []
            self.fvg_records[key].append(fvg)
            return fvg
        return None

    def detect_liquidity_pool(
        self, bars: List[OHLCV_BAR], instrument: str, timeframe: int
    ) -> Optional[LIQUIDITY_POOL]:
        """Detect liquidity pool."""
        if len(bars) < 5 or np is None:
            return None

        recent_highs = [b.high for b in bars[-5:]]
        if len(set([round(h, 5) for h in recent_highs])) <= 2:
            pool_price = max(recent_highs)
            liq = LIQUIDITY_POOL(
                instrument=instrument,
                timeframe=timeframe,
                level_price=pool_price,
                touch_count=sum(1 for h in recent_highs if abs(h - pool_price) < 0.0001),
                created_bar_id=len(bars) - 1
            )
            key = (instrument, timeframe)
            if key not in self.liq_records:
                self.liq_records[key] = []
            self.liq_records[key].append(liq)
            return liq
        return None
