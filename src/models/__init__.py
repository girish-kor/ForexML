"""ML inference module."""

import os
import logging
from typing import Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from schemas import FEATURE_SNAPSHOT, MODEL_PREDICTION

logger = logging.getLogger(__name__)


class MLInference:
    """ML model inference engine."""

    def __init__(self, model_artifact_path: str):
        self.model_path = model_artifact_path
        self.lgbm_model = None
        self.tabnet_model = None
        self.calibrator = None
        self.load_models()

    def load_models(self):
        """Load trained models."""
        try:
            if joblib is None or not os.path.exists(self.model_path):
                logger.warning("Model artifact not available, using dummy inference")
                self.lgbm_model = self._create_dummy_model()
                return

            with open(self.model_path, 'rb') as f:
                models = joblib.load(f)
                self.lgbm_model = models.get('lgbm')
                self.tabnet_model = models.get('tabnet')
                self.calibrator = models.get('calibrator')
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading models: {e} - using dummy")
            self.lgbm_model = self._create_dummy_model()

    def _create_dummy_model(self):
        """Create dummy model for fallback."""
        if np is None:
            return lambda x: [0.33, 0.34, 0.33]
        return lambda x: np.array([[0.33, 0.34, 0.33]])

    def run_inference(self, feature_vector: np.ndarray) -> Tuple[float, float, float]:
        """Run inference on feature vector."""
        if np is None or self.lgbm_model is None:
            return 0.33, 0.34, 0.33

        try:
            pred = self.lgbm_model.predict_proba(feature_vector.reshape(1, -1))
            return float(pred[0, 0]), float(pred[0, 1]), float(pred[0, 2])
        except Exception:
            return 0.33, 0.34, 0.33

    def ensemble_blend(
        self, lgbm_probs: Tuple, tabnet_probs: Tuple
    ) -> Tuple[float, float, float]:
        """Blend ensemble predictions."""
        w_lgbm, w_tabnet = 0.6, 0.4
        blended = (
            w_lgbm * lgbm_probs[0] + w_tabnet * tabnet_probs[0],
            w_lgbm * lgbm_probs[1] + w_tabnet * tabnet_probs[1],
            w_lgbm * lgbm_probs[2] + w_tabnet * tabnet_probs[2]
        )
        return blended

    def calibrate_probabilities(self, probs: Tuple) -> float:
        """Calibrate prediction probabilities."""
        max_prob = max(probs)
        return min(max_prob, 1.0)

    def store_prediction(
        self, bar_id: int, instrument: str, probs: Tuple, bar_timestamp: int
    ) -> MODEL_PREDICTION:
        """Store prediction result."""
        if np is None:
            pred_class = 0
        else:
            pred_class = np.argmax(probs)

        direction_map = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}
        calibration_score = self.calibrate_probabilities(probs)

        return MODEL_PREDICTION(
            bar_id=bar_id,
            instrument=instrument,
            prob_long=float(probs[0]),
            prob_short=float(probs[1]),
            prob_neutral=float(probs[2]),
            predicted_direction=direction_map[pred_class],
            confidence_score=float(max(probs)),
            calibration_score=calibration_score,
            bar_timestamp=bar_timestamp
        )
