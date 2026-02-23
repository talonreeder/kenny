"""
KENNY WebSocket Manager — Broadcasts verdicts to dashboard clients via API Gateway.
"""

import json
import logging
from typing import Set, Optional

import boto3
from botocore.exceptions import ClientError

from app.config import get_settings

logger = logging.getLogger(__name__)


class WebSocketManager:
    def __init__(self):
        self._settings = get_settings()
        self._dynamodb = None
        self._apigw_management = None

    @property
    def dynamodb(self):
        if self._dynamodb is None:
            self._dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        return self._dynamodb

    @property
    def connections_table(self):
        return self.dynamodb.Table(self._settings.ws_connections_table)

    def _get_management_client(self):
        endpoint = self._settings.ws_api_endpoint
        if not endpoint:
            return None
        return boto3.client(
            "apigatewaymanagementapi",
            endpoint_url=endpoint,
            region_name="us-east-1"
        )

    async def get_connection_ids(self) -> list:
        try:
            response = self.connections_table.scan(ProjectionExpression="connectionId")
            return [item["connectionId"] for item in response.get("Items", [])]
        except Exception as e:
            logger.error(f"Failed to get connection IDs: {e}")
            return []

    async def broadcast(self, message: dict):
        payload = json.dumps(message).encode("utf-8")
        client = self._get_management_client()
        if client:
            connection_ids = await self.get_connection_ids()
            stale = []
            for conn_id in connection_ids:
                try:
                    client.post_to_connection(ConnectionId=conn_id, Data=payload)
                except ClientError as e:
                    if e.response["Error"]["Code"] == "GoneException":
                        stale.append(conn_id)
                    else:
                        logger.error(f"Failed to post to {conn_id}: {e}")
            for conn_id in stale:
                try:
                    self.connections_table.delete_item(Key={"connectionId": conn_id})
                except Exception:
                    pass
            if connection_ids:
                logger.info(f"Broadcast to {len(connection_ids)} clients ({len(stale)} stale removed)")
        else:
            logger.debug("No WS API endpoint configured — broadcast skipped")

    async def broadcast_verdict(self, verdict: dict):
        await self.broadcast({"type": "verdict", "data": verdict})

    async def broadcast_health(self, status: dict):
        await self.broadcast({"type": "health", "data": status})


ws_manager = WebSocketManager()