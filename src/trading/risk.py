"""Risk management module."""

import logging
from typing import List

from schemas import TRADING_SIGNAL, POSITION_SIZING

logger = logging.getLogger(__name__)


class RiskManagement:
    """Risk management and position sizing."""

    def __init__(
        self,
        max_risk_per_trade: float = 0.02,
        max_daily_drawdown: float = 0.05,
        max_open_positions: int = 5,
        kelly_fraction: float = 0.25
    ):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.max_open_positions = max_open_positions
        self.kelly_fraction = kelly_fraction
        self.daily_drawdown = 0.0
        self.open_positions_count = 0

    def check_daily_drawdown(self, current_equity: float, starting_equity: float) -> bool:
        """Check if daily drawdown limit is exceeded."""
        self.daily_drawdown = (starting_equity - current_equity) / starting_equity
        return self.daily_drawdown < self.max_daily_drawdown

    def check_correlated_exposure(self, instrument: str, current_positions: List[str]) -> bool:
        """Check for correlated exposure."""
        corr_pairs = {'EURUSD': 'GBPUSD', 'AUDUSD': 'NZDUSD'}
        if instrument in corr_pairs:
            return corr_pairs[instrument] not in current_positions
        return True

    def calculate_position_sizing(
        self, signal: TRADING_SIGNAL, account_balance: float, pip_value: float = 10.0
    ) -> POSITION_SIZING:
        """Calculate position size."""
        risk_amount = account_balance * self.max_risk_per_trade
        stop_loss_pips = abs(signal.entry_price - signal.stop_loss) * 10000

        if stop_loss_pips > 0:
            lot_size = risk_amount / (stop_loss_pips * pip_value)
        else:
            lot_size = 0.1

        lot_size = min(max(lot_size * self.kelly_fraction, 0.01), 10.0)
        margin_required = lot_size * signal.entry_price * 100
        leverage_used = (
            (lot_size * signal.entry_price) / account_balance
            if account_balance > 0 else 0
        )

        return POSITION_SIZING(
            signal_id=signal.signal_id,
            account_balance=account_balance,
            risk_amount=risk_amount,
            lot_size=lot_size,
            margin_required=margin_required,
            leverage_used=min(leverage_used, 100.0)
        )

    def check_margin_available(
        self,
        account_balance: float,
        margin_required: float,
        free_margin: float = None
    ) -> bool:
        """Check if margin is available."""
        if free_margin is None:
            free_margin = account_balance * 0.5
        return margin_required <= free_margin
