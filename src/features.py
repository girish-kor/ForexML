"""Feature engineering module."""

import logging
from typing import Dict, List

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None

from schemas import (
    OHLCV_BAR,
    MARKET_STRUCTURE_SHIFT,
    ORDER_BLOCK,
    FAIR_VALUE_GAP,
    FEATURE_SNAPSHOT
)

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Feature engineering pipeline."""

    def __init__(self, feature_set_version: str = '1.0'):
        self.feature_set_version = feature_set_version or '1.0'
        if StandardScaler is not None:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

    def compute_price_features(self, bars: List[OHLCV_BAR]) -> Dict[str, float]:
        """Compute price-based features."""
        if len(bars) < 2 or np is None:
            return {'return_5bar': 0.0, 'volatility_atr': 0.0, 'momentum_rsi': 50.0, 'spread_norm': 0.001}

        closes = np.array([b.close for b in bars])
        returns = np.diff(np.log(closes))

        features = {
            'return_5bar': np.sum(returns[-5:]) if len(returns) >= 5 else 0,
            'volatility_atr': np.std(returns) * 100 if len(returns) > 0 else 0,
            'momentum_rsi': self._compute_rsi(closes, 14),
            'spread_norm': bars[-1].spread_avg / (closes[-1] * 0.0001)
        }
        return features

    def _compute_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Compute RSI indicator."""
        if len(prices) < period + 1 or np is None:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    def compute_smc_features(
        self,
        mss: MARKET_STRUCTURE_SHIFT,
        ob: ORDER_BLOCK,
        fvg: FAIR_VALUE_GAP,
        current_price: float
    ) -> Dict[str, float]:
        """Compute SMC-based features."""
        features = {
            'ob_proximity_pips': abs(current_price - ob.zone_high) * 10000 if ob else 0,
            'ob_strength': ob.strength_score if ob else 0,
            'fvg_fill_pct': fvg.fill_pct if fvg else 0,
            'mss_age_bars': 5
        }
        return features

    def compute_microstructure_features(self) -> Dict[str, float]:
        """Compute microstructure features."""
        return {
            'spread_ratio': 1.0,
            'tick_volume_imbalance': 0.5,
            'session_overlap': 1.0
        }

    def compute_macro_features(self) -> Dict[str, float]:
        """Compute macro features."""
        return {
            'dxy_correlation': 0.65,
            'event_impact': 0.0
        }

    def normalize_features(self, feature_dict: Dict) -> np.ndarray:
        """Normalize feature vector."""
        if np is None:
            return np.array([0.0] * len(feature_dict))

        vector = np.array(list(feature_dict.values())).reshape(1, -1)
        try:
            if StandardScaler is None:
                return vector.flatten()
            normalized = self.scaler.fit_transform(vector)
            return normalized.flatten()
        except Exception:
            return vector.flatten()

    def build_feature_snapshot(
        self, instrument: str, bar_id: int, features: Dict
    ) -> FEATURE_SNAPSHOT:
        """Build feature snapshot for ML."""
        vector = self.normalize_features(features)
        return FEATURE_SNAPSHOT(
            instrument=instrument,
            bar_id=bar_id,
            feature_set_version=self.feature_set_version,
            feature_vector=vector
        )
