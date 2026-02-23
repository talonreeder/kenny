import boto3
from functools import lru_cache

def get_ssm_parameter(name: str, decrypt: bool = True) -> str:
    """Fetch a parameter from AWS SSM Parameter Store."""
    ssm = boto3.client("ssm", region_name="us-east-1")
    response = ssm.get_parameter(Name=name, WithDecryption=decrypt)
    return response["Parameter"]["Value"]

class Settings:
    APP_NAME: str = "YELENA v2"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = get_ssm_parameter("/yelena/database-url")
    
    # Polygon.io
    POLYGON_API_KEY: str = get_ssm_parameter("/yelena/polygon/api-key")
    
    # Schwab
    SCHWAB_API_KEY: str = get_ssm_parameter("/yelena/schwab/api-key")
    SCHWAB_API_SECRET: str = get_ssm_parameter("/yelena/schwab/api-secret")
    SCHWAB_CALLBACK_URL: str = get_ssm_parameter("/yelena/schwab/callback-url", decrypt=False)
    
    # Webhook
    WEBHOOK_PASSPHRASE: str = get_ssm_parameter("/yelena/webhook-passphrase")
    
    # AWS
    S3_DATA_LAKE: str = "yelena-data-lake"
    S3_MODELS: str = "yelena-models"
    S3_BACKUPS: str = "yelena-backups"
    AWS_REGION: str = "us-east-1"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
