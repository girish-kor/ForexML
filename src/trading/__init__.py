"""Order execution module."""

import time
import logging
from typing import Tuple, Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from schemas import TRADING_SIGNAL, POSITION_SIZING, ORDER, TRADE

logger = logging.getLogger(__name__)


class OrderExecution:
    """Order building and execution."""

    def __init__(self):
        self.order_counter = 0
        self.retry_limit = 3

    def build_order(self, signal: TRADING_SIGNAL, sizing: POSITION_SIZING) -> ORDER:
        """Build order from signal and sizing."""
        self.order_counter += 1
        return ORDER(
            order_type='LIMIT',
            direction=signal.direction,
            volume=sizing.lot_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit_1,
            status='PENDING'
        )

    def send_to_mt5(
        self, order: ORDER, instrument: str
    ) -> Tuple[bool, Optional[int], Optional[dict]]:
        """Send order to MT5."""
        try:
            if mt5 is None:
                return False, None, {'error': 'MT5 not available'}

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": instrument,
                "volume": order.volume,
                "type": (
                    mt5.ORDER_TYPE_BUY
                    if order.direction == 'LONG' else mt5.ORDER_TYPE_SELL
                ),
                "price": order.stop_loss,
                "sl": order.stop_loss,
                "tp": order.take_profit,
                "deviation": 20,
                "magic": 12345,
                "comment": "ForexML",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return True, result.order, {'executed_price': result.price, 'slippage_pips': 0}
            else:
                return False, None, {'error': result.comment}
        except Exception as e:
            logger.error(f"MT5 send error: {e}")
            return False, None, {'error': str(e)}

    def execute_order(
        self, signal: TRADING_SIGNAL, sizing: POSITION_SIZING, instrument: str
    ) -> Tuple[bool, Optional[ORDER], Optional[TRADE]]:
        """Execute order with retries."""
        order = self.build_order(signal, sizing)

        for attempt in range(self.retry_limit):
            success, broker_order_id, response = self.send_to_mt5(order, instrument)
            if success:
                order.broker_order_id = broker_order_id
                order.status = 'FILLED'
                order.executed_price = response.get('executed_price', signal.entry_price)
                order.slippage_pips = response.get('slippage_pips', 0)
                order.filled_at = int(time.time())

                trade = TRADE(
                    trade_id=f"TRD_{int(time.time())}_{self.order_counter}",
                    signal_id=signal.signal_id,
                    instrument=instrument,
                    direction=signal.direction,
                    open_price=order.executed_price,
                    volume=order.volume,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit_1,
                    open_time=int(time.time())
                )

                return True, order, trade

            time.sleep(1)

        order.status = 'REJECTED'
        order.rejection_reason = 'TIMEOUT'
        return False, order, None


class TradeLifecycleManager:
    """Trade monitoring and settlement."""

    def __init__(self):
        self.open_trades = {}
        self.closed_trades = {}

    def monitor_trade(self, trade: TRADE, current_price: float) -> Tuple[bool, Optional[str]]:
        """Monitor trade for exit conditions."""
        if trade.direction == 'LONG':
            if current_price <= trade.stop_loss:
                return True, 'STOP_LOSS_HIT'
            if current_price >= trade.take_profit:
                return True, 'TAKE_PROFIT_HIT'
        else:
            if current_price >= trade.stop_loss:
                return True, 'STOP_LOSS_HIT'
            if current_price <= trade.take_profit:
                return True, 'TAKE_PROFIT_HIT'

        return False, None

    def settle_trade(self, trade: TRADE, close_price: float, close_reason: str) -> TRADE:
        """Settle and close trade."""
        trade.close_price = close_price
        trade.close_time = int(time.time())
        trade.close_reason = close_reason

        if trade.direction == 'LONG':
            trade.gross_pnl = (close_price - trade.open_price) * trade.volume
            trade.pnl_pips = (close_price - trade.open_price) * 10000
        else:
            trade.gross_pnl = (trade.open_price - close_price) * trade.volume
            trade.pnl_pips = (trade.open_price - close_price) * 10000

        commission = trade.gross_pnl * 0.001
        swap = 0
        trade.net_pnl = trade.gross_pnl - commission - swap

        mae = close_price - trade.open_price
        mfe = close_price - trade.open_price
        trade.mae_pips = min(mae, 0) * 10000 if trade.direction == 'LONG' else min(-mae, 0) * 10000
        trade.mfe_pips = max(mfe, 0) * 10000 if trade.direction == 'LONG' else max(-mfe, 0) * 10000

        return trade
