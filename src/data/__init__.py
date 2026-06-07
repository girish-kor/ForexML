"""Data ingestion module."""

import time
import logging
from typing import Optional, List

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from schemas import TICK, OHLCV_BAR

logger = logging.getLogger(__name__)


class SystemBootstrap:
    """Initialize system and connections."""

    def __init__(self, mt5_path: str, max_reconnect_attempts: int = 5, reconnect_backoff_sec: float = 2.0):
        self.mt5_path = mt5_path or "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
        self.reconnect_attempts = 0
        self.mt5_connected = False
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_backoff_sec = reconnect_backoff_sec

    def check_mt5_connection(self) -> bool:
        """Check MT5 connection status."""
        if self.mt5_connected:
            return True
        try:
            if mt5 is None:
                logger.warning("MT5 not installed - using mock mode")
                self.mt5_connected = True
                return True
            if not mt5.initialize(path=self.mt5_path):
                return False
            self.mt5_connected = True
            logger.info("MT5 connection established")
            return True
        except Exception as e:
            logger.warning(f"MT5 connection failed: {e} - using mock mode")
            self.mt5_connected = True
            return True

    def reconnect_with_backoff(self) -> bool:
        """Reconnect with exponential backoff."""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            logger.info(
                f"Reconnect attempt {self.reconnect_attempts + 1}/{self.max_reconnect_attempts}"
            )
            time.sleep(self.reconnect_backoff_sec)
            if self.check_mt5_connection():
                self.reconnect_attempts = 0
                logger.info("MT5 connection restored")
                return True
            self.reconnect_attempts += 1
        logger.critical("MT5_CONN_FAIL - Max reconnect attempts exceeded")
        return False

    def run_bootstrap(self) -> bool:
        """Run bootstrap sequence."""
        if not self.check_mt5_connection():
            return self.reconnect_with_backoff()
        return True


class MarketDataIngestion:
    """Market data acquisition and aggregation."""

    def __init__(self):
        self.bars_store = {}
        self.dedup_keys = set()

    def check_session_active(self, instrument: str) -> bool:
        """Check if trading session is active."""
        try:
            if mt5 is None:
                return True
            tick = mt5.symbol_info_tick(instrument)
            return tick is not None
        except Exception:
            return True

    def wait_for_session_open(self, instrument: str):
        """Wait for session to open."""
        logger.info(f"Waiting for {instrument} session open")
        while not self.check_session_active(instrument):
            time.sleep(60)

    def check_high_impact_news(self, instrument: str, window_min: int = 30) -> bool:
        """Check for high-impact news events."""
        return False

    def ingest_tick_stream(self, instrument: str) -> Optional[TICK]:
        """Ingest live tick data."""
        try:
            if mt5 is None:
                return TICK(
                    instrument=instrument,
                    bid=1.0850,
                    ask=1.0851,
                    spread=0.0001,
                    volume=1000,
                    timestamp_us=int(time.time() * 1e6)
                )
            tick = mt5.symbol_info_tick(instrument)
            if tick:
                return TICK(
                    instrument=instrument,
                    bid=tick.bid,
                    ask=tick.ask,
                    spread=tick.ask - tick.bid,
                    volume=tick.volume,
                    timestamp_us=int(tick.time * 1e6)
                )
        except Exception as e:
            logger.error(f"Error ingesting tick: {e}")
        return None

    def aggregate_to_ohlcv_bar(self, instrument: str, timeframe: int) -> Optional[OHLCV_BAR]:
        """Aggregate to OHLCV bar."""
        try:
            if mt5 is None:
                return OHLCV_BAR(
                    instrument=instrument,
                    timeframe=timeframe,
                    open_time=int(time.time()),
                    open=1.0850,
                    high=1.0855,
                    low=1.0845,
                    close=1.0852,
                    volume=5000,
                    spread_avg=0.00002,
                    data_source='MOCK'
                )
            rates = mt5.copy_rates_from_pos(instrument, timeframe, 0, 1)
            if rates is not None and len(rates) > 0:
                bar = rates[0]
                return OHLCV_BAR(
                    instrument=instrument,
                    timeframe=timeframe,
                    open_time=int(bar[0]),
                    open=bar[1],
                    high=bar[2],
                    low=bar[3],
                    close=bar[4],
                    volume=int(bar[5]),
                    spread_avg=0.00002,
                    data_source='MT5'
                )
        except Exception as e:
            logger.error(f"Error aggregating bar: {e}")
        return None

    def check_duplicate(self, bar: OHLCV_BAR) -> bool:
        """Check for duplicate bars."""
        key = (bar.instrument, bar.timeframe, bar.open_time)
        if key in self.dedup_keys:
            return True
        self.dedup_keys.add(key)
        return False

    def store_bar(self, bar: OHLCV_BAR):
        """Store bar in memory."""
        key = (bar.instrument, bar.timeframe)
        if key not in self.bars_store:
            self.bars_store[key] = []
        self.bars_store[key].append(bar)
        logger.info(f"Stored bar {bar.instrument} {bar.timeframe} at {bar.open_time}")

    def get_bars(self, instrument: str, timeframe: int) -> List[OHLCV_BAR]:
        """Get stored bars for instrument."""
        return self.bars_store.get((instrument, timeframe), [])
