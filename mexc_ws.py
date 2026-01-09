"""
MEXC WebSocket Client
Real-time price feeds for maximum speed
"""
import asyncio
import json
import logging
from typing import Callable, Optional
import aiohttp

logger = logging.getLogger(__name__)

MEXC_WS_URL = "wss://contract.mexc.com/edge"


class MEXCWebSocket:
    """
    WebSocket client for real-time MEXC futures prices.
    Much faster than REST API polling.
    """
    
    def __init__(self):
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._prices: dict[str, float] = {}  # {symbol: last_price}
        self._callbacks: list[Callable] = []
        self._reconnect_delay = 1
        self._subscribed_symbols: set[str] = set()
        self._listen_task: Optional[asyncio.Task] = None
    
    @property
    def prices(self) -> dict[str, float]:
        """Get current price snapshot"""
        return self._prices.copy()
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get price for specific symbol"""
        return self._prices.get(symbol)
    
    async def connect(self):
        """Connect to WebSocket"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        
        try:
            self._ws = await self._session.ws_connect(
                MEXC_WS_URL,
                heartbeat=30,
                receive_timeout=60
            )
            self._running = True
            logger.info("✅ MEXC WebSocket connected")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def subscribe_tickers(self, symbols: list[str] = None):
        """
        Subscribe to ticker updates.
        If symbols is None, subscribes to ALL tickers.
        """
        if not self._ws:
            return
        
        # Subscribe to all tickers at once (more efficient)
        sub_msg = {
            "method": "sub.tickers",
            "param": {}
        }
        
        await self._ws.send_json(sub_msg)
        logger.info("📡 Subscribed to all MEXC tickers")

    async def listen(self):
        """Main listening loop - Robust & Single-threaded per instance"""
        # Self-check to prevent duplicates
        if self._listen_task and asyncio.current_task() != self._listen_task:
             logger.warning("Duplicate listener detected, stopping extraneous task")
             return

        while self._running:
            try:
                # 1. Ensure connected
                if not self._ws or self._ws.closed:
                    success = await self.connect()
                    if not success:
                        await asyncio.sleep(self._reconnect_delay)
                        self._reconnect_delay = min(self._reconnect_delay * 2, 30)
                        continue
                    
                    # 2. Resubscribe after connection
                    await self.subscribe_tickers()
                
                # 3. Read loop
                try:
                    async for msg in self._ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                channel = data.get("channel", "")
                                
                                if channel == "push.tickers":
                                    for ticker in data.get("data", []):
                                        sym = ticker.get("symbol", "")
                                        if sym.endswith("_USDT"):
                                            price = float(ticker.get("lastPrice", 0))
                                            if price > 0:
                                                self._prices[sym[:-5]] = price
                                                
                                elif channel == "push.ticker":
                                    pass

                            except json.JSONDecodeError:
                                continue
                                
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error frame: {msg.data}")
                            await asyncio.sleep(1)
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.warning("WebSocket closed by server")
                            break
                except RuntimeError as e:
                    if "Concurrent call to receive" in str(e):
                         logger.error("Concurrent receive error detected. Resetting connection.")
                         await self.close()
                         continue
                    raise e
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket listener loop error: {e}")
                await asyncio.sleep(1)
                
            # Reconnection delay if loop breaks
            if self._running:
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    async def start(self):
        """Start WebSocket - Idempotent with Task Tracking"""
        if self._listen_task and not self._listen_task.done():
            logger.warning("WebSocket listener already running")
            return

        self._running = True
        self._listen_task = asyncio.create_task(self.listen())
        logger.info("WebSocket listener started")
    
    async def close(self):
        """Close WebSocket connection"""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("WebSocket closed")


# Global instance
_ws_client: Optional[MEXCWebSocket] = None


def get_ws_client() -> MEXCWebSocket:
    """Get singleton WebSocket client"""
    global _ws_client
    if _ws_client is None:
        _ws_client = MEXCWebSocket()
    return _ws_client
