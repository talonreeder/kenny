"""
KENNY Configuration — SSM-backed secrets, environment-aware settings.
All secrets come from AWS SSM Parameter Store. NEVER hardcode secrets.
"""

import os
import logging
from functools import lru_cache
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class Settings:
    """Application settings with SSM Parameter Store integration."""

    def __init__(self):
        self.env = os.getenv("KENNY_ENV", "production")
        self._ssm_client = None
        self._cache = {}

        # Application settings
        self.host = os.getenv("KENNY_HOST", "0.0.0.0")
        self.port = int(os.getenv("KENNY_PORT", "8000"))
        self.debug = self.env == "development"

        # Paths
        self.models_dir = os.getenv("KENNY_MODELS_DIR", os.path.expanduser("~/kenny/backend/models"))
        self.s3_models_bucket = "yelena-models"
        self.s3_models_prefix = "models_v2/"
        self.s3_data_bucket = "yelena-data-lake"

        # Trading settings
        self.symbols = [
            "SPY", "QQQ", "TSLA", "NVDA", "META",
            "AAPL", "GOOGL", "MSFT", "AMZN", "AMD", "NFLX"
        ]
        self.timeframes = ["1min", "5min", "15min", "1hr"]
        self.primary_timeframe = "15min"

        # Risk settings
        self.max_position_pct = 0.10
        self.daily_loss_limit = 500.0
        self.max_concurrent_positions = 3
        self.min_rr_ratio = 1.2
        self.atr_sl_multiplier = 1.5
        self.atr_tp_multiplier = 2.0

        # ML settings
        self.ensemble_threshold = 0.55
        self.min_model_agreement = 2

        # TV Confluence settings
        self.tv_signal_ttl_seconds = 600
        self.tv_signal_half_life_seconds = 300
        self.tv_max_boost = 0.15
        self.tv_max_penalty = -0.20

        # Feature counts per timeframe
        self.feature_counts = {
            "1min": 163,
            "5min": 156,
            "15min": 149,
            "1hr": 139
        }

        # WebSocket API (set after API Gateway creation)
        self.ws_api_endpoint = os.getenv("KENNY_WS_API_ENDPOINT", "")
        self.ws_connections_table = "kenny-ws-connections"

    @property
    def ssm_client(self):
        if self._ssm_client is None:
            self._ssm_client = boto3.client("ssm", region_name="us-east-1")
        return self._ssm_client

    def _get_ssm_param(self, name: str, decrypt: bool = True) -> Optional[str]:
        if name in self._cache:
            return self._cache[name]
        try:
            response = self.ssm_client.get_parameter(Name=name, WithDecryption=decrypt)
            value = response["Parameter"]["Value"]
            self._cache[name] = value
            return value
        except ClientError as e:
            logger.error(f"Failed to get SSM parameter {name}: {e}")
            return None

    @property
    def database_url(self) -> str:
        return self._get_ssm_param("/yelena/database-url") or ""

    @property
    def polygon_api_key(self) -> str:
        return self._get_ssm_param("/yelena/polygon/api-key") or ""

    @property
    def schwab_api_key(self) -> str:
        return self._get_ssm_param("/yelena/schwab/api-key") or ""

    @property
    def schwab_api_secret(self) -> str:
        return self._get_ssm_param("/yelena/schwab/api-secret") or ""

    @property
    def schwab_callback_url(self) -> str:
        return self._get_ssm_param("/yelena/schwab/callback-url", decrypt=False) or ""

    @property
    def webhook_passphrase(self) -> str:
        return self._get_ssm_param("/yelena/webhook-passphrase") or ""

    def parse_database_url(self) -> dict:
        url = self.database_url
        if not url:
            return {}
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return {
            "user": parsed.username,
            "password": parsed.password,
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip("/")
        }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
