import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ForexMLTrading')


@dataclass
class SYSTEM_CONFIG:
    """System-wide configuration."""
    mt5_path: str = None
    account: int = None
    password: str = None
    server: str = None
    broker_credentials: Dict = field(default_factory=dict)
    risk_profile: Dict = field(default_factory=dict)
    model_artifact_path: str = None
    feature_set_version: str = '1.0'
    socket_host: str = 'localhost'
    socket_port: int = 5555
    max_reconnect_attempts: int = 5
    reconnect_backoff_sec: float = 2.0

    def __post_init__(self):
        if not self.broker_credentials:
            self.broker_credentials = {}
        if not self.risk_profile:
            self.risk_profile = {}


def load_system_config() -> SYSTEM_CONFIG:
    """Load system config from environment variables."""
    return SYSTEM_CONFIG(
        mt5_path=os.environ.get('MT5_PATH', 'C:\\Program Files\\MetaTrader 5\\terminal64.exe'),
        account=int(os.environ.get('MT5_ACCOUNT', '0')),
        password=os.environ.get('MT5_PASSWORD', ''),
        server=os.environ.get('MT5_SERVER', ''),
        broker_credentials={
            'account': int(os.environ.get('MT5_ACCOUNT', '0')),
            'password': os.environ.get('MT5_PASSWORD', ''),
            'server': os.environ.get('MT5_SERVER', '')
        },
        risk_profile={
            'max_risk_per_trade': 0.02,
            'max_daily_drawdown': 0.05,
            'max_open_positions': 5,
            'kelly_fraction': 0.25
        },
        model_artifact_path=os.environ.get('MODEL_PATH', 'model.pkl'),
        feature_set_version='1.0'
    )
