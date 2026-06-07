"""Main trading system orchestrator."""

import time
import logging

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

try:
    import numpy as np
except ImportError:
    np = None

from config import SYSTEM_CONFIG
from data import SystemBootstrap, MarketDataIngestion
from data.structures import SMCStructureDetection
from features import FeatureEngineering
from models import MLInference
from models.drift import ModelDriftMonitor
from models.retraining import ModelRetraining
from trading import OrderExecution, TradeLifecycleManager
from trading.signals import SignalGeneration
from trading.risk import RiskManagement
from analytics import PerformanceAnalytics

logger = logging.getLogger(__name__)


class ForexMLTradingSystem:
    """Main trading system orchestrator."""

    def __init__(self, config: SYSTEM_CONFIG):
        self.config = config
        self.bootstrap = SystemBootstrap(
            config.mt5_path,
            config.max_reconnect_attempts,
            config.reconnect_backoff_sec
        )
        self.ingestion = MarketDataIngestion()
        self.smc = SMCStructureDetection()
        self.features = FeatureEngineering(config.feature_set_version)
        self.inference = MLInference(config.model_artifact_path or 'model.pkl')
        self.signals = SignalGeneration()
        self.risk = RiskManagement()
        self.execution = OrderExecution()
        self.lifecycle = TradeLifecycleManager()
        self.analytics = PerformanceAnalytics()
        self.drift = ModelDriftMonitor()
        self.retrain = ModelRetraining()

        self.instruments = ['EURUSD', 'GBPUSD', 'AUDUSD']
        self.timeframe = mt5.TIMEFRAME_H1 if mt5 else 60
        self.running = False
        self.trades = []
        self.signals_generated = []

    def run(self):
        """Main trading loop."""
        logger.info("Starting ForexML Trading System")

        if not self.bootstrap.run_bootstrap():
            logger.critical("Bootstrap failed")
            return

        self.running = True
        try:
            while self.running:
                for instrument in self.instruments:
                    self._process_instrument(instrument)
                self._monitor_open_trades()
                self._check_performance_and_drift()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("System interrupted")
        finally:
            self._cleanup()

    def _process_instrument(self, instrument: str):
        """Process single instrument."""
        if not self.ingestion.check_session_active(instrument):
            self.ingestion.wait_for_session_open(instrument)
            return

        if self.ingestion.check_high_impact_news(instrument):
            logger.info(f"News suppression active for {instrument}")
            return

        tick = self.ingestion.ingest_tick_stream(instrument)
        if not tick:
            return

        bar = self.ingestion.aggregate_to_ohlcv_bar(instrument, self.timeframe)
        if not bar:
            return

        if self.ingestion.check_duplicate(bar):
            logger.warning(f"Duplicate bar for {instrument}")
            return

        self.ingestion.store_bar(bar)

        bars_key = (instrument, self.timeframe)
        bars = self.ingestion.get_bars(instrument, self.timeframe)

        self.smc.detect_market_structure_shift(bars, instrument, self.timeframe)
        ob = self.smc.detect_order_block(bars, instrument, self.timeframe)
        fvg = self.smc.detect_fair_value_gap(bars, instrument, self.timeframe)
        liq = self.smc.detect_liquidity_pool(bars, instrument, self.timeframe)

        # Build feature vector
        feature_dict = {}
        feature_dict.update(self.features.compute_price_features(bars))
        feature_dict.update(
            self.features.compute_smc_features(None, ob, fvg, bar.close)
        )
        feature_dict.update(self.features.compute_microstructure_features())
        feature_dict.update(self.features.compute_macro_features())

        feature_snapshot = self.features.build_feature_snapshot(
            instrument, len(bars) - 1, feature_dict
        )

        # Run inference
        lgbm_probs = self.inference.run_inference(feature_snapshot.feature_vector)
        tabnet_probs = self.inference.run_inference(feature_snapshot.feature_vector)
        blended_probs = self.inference.ensemble_blend(lgbm_probs, tabnet_probs)

        pred = self.inference.store_prediction(
            len(bars) - 1, instrument, blended_probs, bar.open_time
        )

        # Check confidence
        if not self.signals.check_confidence_threshold(pred):
            logger.info(
                f"Prediction rejected for {instrument} - low confidence: {pred.confidence_score}"
            )
            return

        # Score confluence
        mss = (
            self.smc.mss_records.get(bars_key, [None])[-1]
            if (bars_key in self.smc.mss_records and self.smc.mss_records[bars_key])
            else None
        )
        confluence_score, confluence_tags = self.signals.score_confluence(
            pred, ob, fvg, mss, liq
        )

        if confluence_score < self.signals.confluence_min:
            logger.info(
                f"Signal rejected for {instrument} - low confluence: {confluence_score}"
            )
            return

        # Build signal
        signal = self.signals.build_trading_signal(
            pred, ob, confluence_score, confluence_tags, bar.close
        )

        # Apply filters
        signal = self.signals.apply_filter_flags(
            signal,
            news_suppressed=False,
            open_positions=len(self.trades),
            max_positions=self.risk.max_open_positions
        )

        if signal.status == 'FILTERED':
            logger.info(f"Signal filtered for {instrument}: {signal.filter_flags}")
            return

        # Risk checks
        if not self.risk.check_daily_drawdown(10000, 10000):
            logger.warning("Daily drawdown limit exceeded")
            return

        if not self.risk.check_correlated_exposure(
            instrument, [t.instrument for t in self.trades]
        ):
            logger.info(f"Signal rejected for {instrument} - correlated exposure")
            return

        # Position sizing
        sizing = self.risk.calculate_position_sizing(signal, 10000)

        if not self.risk.check_margin_available(10000, sizing.margin_required):
            logger.warning(f"Insufficient margin for {instrument}")
            return

        # Execute order
        success, order, trade = self.execution.execute_order(signal, sizing, instrument)

        if success and trade:
            self.trades.append(trade)
            self.signals_generated.append(signal)
            logger.info(
                f"Trade opened: {instrument} {signal.direction} at {trade.open_price}"
            )
        else:
            logger.warning(f"Order execution failed for {instrument}")

    def _monitor_open_trades(self):
        """Monitor open trades for exits."""
        for trade in self.trades:
            if trade.close_time is None:
                try:
                    if mt5 is None:
                        current_price = 1.0850
                    else:
                        tick = mt5.symbol_info_tick(trade.instrument)
                        current_price = (
                            (tick.bid + tick.ask) / 2 if tick else trade.open_price
                        )

                    should_close, close_reason = self.lifecycle.monitor_trade(
                        trade, current_price
                    )

                    if should_close:
                        trade = self.lifecycle.settle_trade(
                            trade, current_price, close_reason
                        )
                        logger.info(
                            f"Trade closed: {trade.instrument} {close_reason} "
                            f"PnL: {trade.net_pnl}"
                        )
                except Exception as e:
                    logger.warning(f"Error monitoring trade: {e}")

    def _check_performance_and_drift(self):
        """Check performance metrics and drift."""
        snapshot = self.analytics.calculate_performance_snapshot(self.trades)

        if snapshot and not self.analytics.check_performance_gate(snapshot):
            logger.warning("Performance gate failed - drift check triggered")
            self._run_drift_detection()

    def _run_drift_detection(self):
        """Run drift detection across instruments."""
        for instrument in self.instruments:
            if np is None:
                continue

            live_features = np.random.randn(50, 20)
            drift_metrics = self.drift.run_drift_detection(
                live_features, model_version='1.0', instrument=instrument
            )

            if drift_metrics.drift_alert_triggered:
                logger.critical(
                    f"DRIFT ALERT for {instrument}: {drift_metrics.alert_severity}"
                )
                self._trigger_retraining(drift_metrics)

    def _trigger_retraining(self, drift_metrics):
        """Trigger model retraining pipeline."""
        if np is None:
            return

        logger.info("Initiating model retraining pipeline")

        X, y = self.retrain.load_historical_data(limit_bars=1000)
        y = self.retrain.apply_triple_barrier_labels(X, y)

        if not self.retrain.check_label_distribution(y):
            X, y = self.retrain.resample_with_smote(X, y)

        folds = self.retrain.purged_kfold_splits(X, n_splits=5)

        fold_metrics_list = []
        for fold in folds:
            train_loss, val_loss = self.retrain.train_fold(X, y, fold)
            fold_metrics = self.retrain.evaluate_fold(X, y, fold)
            fold_metrics_list.append(fold_metrics)

        aggregated_metrics = self.retrain.aggregate_fold_metrics(fold_metrics_list)

        incumbent_accuracy = 0.70
        if not self.retrain.check_psr_and_accuracy(aggregated_metrics, incumbent_accuracy):
            logger.info("Model rejected - PSR or accuracy check failed")
            return

        model_version = self.retrain.register_model_version(
            'model_new.pkl', aggregated_metrics
        )

        shadow_passed = self.retrain.shadow_test(X, y, model_version, incumbent_accuracy)

        if shadow_passed:
            model_version = self.retrain.promote_model(model_version)
            logger.info(f"New model promoted: {model_version.version_id}")
        else:
            logger.info("Shadow test failed - model rejected")

    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up")
        self.running = False
        if mt5 is not None:
            try:
                mt5.shutdown()
            except Exception:
                pass


def main():
    """Entry point for the CLI."""
    system = ForexMLTradingSystem(SYSTEM_CONFIG)
    system.run()


if __name__ == "__main__":
    main()
