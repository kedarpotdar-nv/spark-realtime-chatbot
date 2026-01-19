"""Shared HTTP session manager for all API clients."""

import asyncio
from typing import Optional

import aiohttp


class HTTPSessionManager:
    """Manages shared aiohttp sessions for all HTTP clients.

    Reusing sessions avoids TCP connection overhead (~50-100ms per request).
    Sessions are created lazily and cleaned up on shutdown.
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create the shared aiohttp session."""
        if self._session is None or self._session.closed:
            async with self._lock:
                # Double-check after acquiring lock
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=100,
                        limit_per_host=30,
                        keepalive_timeout=30,
                        enable_cleanup_closed=True,
                    )
                    timeout = aiohttp.ClientTimeout(total=120, connect=10)
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                    )
                    print("[HTTP] Created shared aiohttp session with connection pooling")
        return self._session

    async def close(self):
        """Close the shared session."""
        if self._session and not self._session.closed:
            await self._session.close()
            print("[HTTP] Closed shared aiohttp session")
            self._session = None


# Global session manager instance
http_manager: Optional[HTTPSessionManager] = None


def get_http_manager() -> HTTPSessionManager:
    """Get the global HTTP session manager."""
    global http_manager
    if http_manager is None:
        http_manager = HTTPSessionManager()
    return http_manager


def set_http_manager(manager: HTTPSessionManager):
    """Set the global HTTP session manager."""
    global http_manager
    http_manager = manager
