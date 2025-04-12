"""Manages active WebSocket connections for broadcasting agent task updates."""

import logging
from typing import Dict, Set, Any
from fastapi import WebSocket # Ensure WebSocket type is available
import asyncio

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        # Store active connections: {user_id: {WebSocket}} - Use a set for uniqueness
        self.active_connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        logger.info(f"ConnectionManager: WebSocket connected for user {user_id}. Total: {len(self.active_connections[user_id])}")

    def disconnect(self, websocket: WebSocket, user_id: int):
        if user_id in self.active_connections:
            # Use discard instead of remove to avoid KeyError if already disconnected
            self.active_connections[user_id].discard(websocket) 
            logger.info(f"ConnectionManager: WebSocket disconnected for user {user_id}. Remaining: {len(self.active_connections[user_id])}")
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                logger.info(f"ConnectionManager: Removed user {user_id} from active connections.")
        else:
            logger.warning(f"ConnectionManager: Disconnect attempt for user {user_id} failed (user not found).")

    async def send_personal_message(self, message: str, user_id: int, websocket: WebSocket):
        """Sends a message to a specific websocket connection."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            # Log error but don't disconnect here, let the main loop handle disconnects
            logger.error(f"ConnectionManager: Failed to send personal message to user {user_id}: {e}")

    async def broadcast_to_user(self, message: str, user_id: int):
        """Sends a message to all active websocket connections for a specific user."""
        if user_id in self.active_connections:
            # Create a copy of the set to iterate over, in case of disconnections during broadcast
            connections_to_send = list(self.active_connections[user_id])
            if not connections_to_send:
                 logger.debug(f"ConnectionManager: No active connections found for user {user_id} to broadcast to (list was empty).")
                 return
                 
            logger.info(f"ConnectionManager: Broadcasting message to {len(connections_to_send)} connection(s) for user {user_id}.")
            tasks = [conn.send_text(message) for conn in connections_to_send]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Log any errors that occurred during broadcast
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Get corresponding websocket safely
                    ws = connections_to_send[i] if i < len(connections_to_send) else None
                    client_info = ws.client if ws else "unknown client"
                    logger.error(f"ConnectionManager: Failed broadcast to {client_info} for user {user_id}: {result}")
        else:
            logger.debug(f"ConnectionManager: No active connections dict entry found for user {user_id} to broadcast.")

# Instantiate the manager (singleton instance)
connection_manager = ConnectionManager() 