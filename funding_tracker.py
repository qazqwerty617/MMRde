"""
Funding Rate Tracker
Tracks MEXC funding rates to calculate position costs.
Critical for accurate profit calculation in Lead-Lag strategy.
"""
import asyncio
import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FundingInfo:
    """Funding rate information for a symbol"""
    symbol: str
    funding_rate: float  # Current funding rate (e.g., 0.0001 = 0.01%)
    next_funding_time: int  # Unix timestamp of next funding
    predicted_rate: float  # Predicted next funding rate
    last_updated: float


class FundingTracker:
    """
    Tracks funding rates for all MEXC futures.
    Used to adjust net profit calculations.
    """
    
    MEXC_CONTRACT_BASE = "https://contract.mexc.com"
    
    def __init__(self):
        self._funding_cache: Dict[str, FundingInfo] = {}
        self._cache_ttl = 300  # 5 minutes cache (funding updates every 8h)
        self._session = None
    
    async def _get_session(self):
        import aiohttp
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def fetch_funding_rate(self, symbol: str) -> Optional[FundingInfo]:
        """
        Fetch current funding rate for a symbol.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            FundingInfo with current rate, or None on error
        """
        # Check cache first
        cached = self._funding_cache.get(symbol)
        if cached and (time.time() - cached.last_updated) < self._cache_ttl:
            return cached
        
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.MEXC_CONTRACT_BASE}/api/v1/contract/funding_rate/{symbol}_USDT"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success") and data.get("data"):
                        rate_data = data["data"]
                        
                        info = FundingInfo(
                            symbol=symbol,
                            funding_rate=float(rate_data.get("fundingRate", 0)),
                            next_funding_time=int(rate_data.get("nextSettleTime", 0)),
                            predicted_rate=float(rate_data.get("expectedFundingRate", 0)),
                            last_updated=time.time()
                        )
                        
                        self._funding_cache[symbol] = info
                        return info
                        
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
        
        return None
    
    async def fetch_all_funding_rates(self) -> Dict[str, FundingInfo]:
        """
        Fetch funding rates for all active futures.
        More efficient than individual calls.
        """
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.MEXC_CONTRACT_BASE}/api/v1/contract/funding_rate"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success") and data.get("data"):
                        now = time.time()
                        for item in data["data"]:
                            symbol_full = item.get("symbol", "")
                            if symbol_full.endswith("_USDT"):
                                symbol = symbol_full[:-5]
                                
                                info = FundingInfo(
                                    symbol=symbol,
                                    funding_rate=float(item.get("fundingRate", 0)),
                                    next_funding_time=int(item.get("nextSettleTime", 0)),
                                    predicted_rate=float(item.get("expectedFundingRate", 0)),
                                    last_updated=now
                                )
                                self._funding_cache[symbol] = info
                        
                        logger.info(f"📊 Loaded {len(self._funding_cache)} funding rates")
                        return self._funding_cache
                        
        except Exception as e:
            logger.error(f"Error fetching all funding rates: {e}")
        
        return {}
    
    def get_cached_rate(self, symbol: str) -> float:
        """
        Get cached funding rate for symbol.
        Returns 0 if not cached.
        """
        info = self._funding_cache.get(symbol)
        return info.funding_rate if info else 0
    
    def calculate_funding_cost(
        self, 
        symbol: str, 
        direction: str,
        hold_hours: float = 4.0
    ) -> float:
        """
        Calculate expected funding cost for a position.
        
        Args:
            symbol: Token symbol
            direction: "LONG" or "SHORT"
            hold_hours: Expected holding time in hours
            
        Returns:
            Expected funding cost as percentage (negative = cost, positive = gain)
        """
        info = self._funding_cache.get(symbol)
        if not info:
            return 0
        
        # Funding is paid every 8 hours
        funding_periods = hold_hours / 8.0
        
        # LONG pays funding when rate is positive
        # SHORT receives funding when rate is positive
        rate = info.funding_rate * 100  # Convert to percentage
        
        if direction == "LONG":
            # LONG pays when funding is positive
            cost = -rate * funding_periods
        else:
            # SHORT receives when funding is positive
            cost = rate * funding_periods
        
        return cost
    
    def get_funding_adjustment(self, symbol: str, direction: str) -> float:
        """
        Get funding rate adjustment for profit calculation.
        Uses predicted rate for accuracy.
        
        Returns adjustment as percentage to subtract from gross profit.
        """
        info = self._funding_cache.get(symbol)
        if not info:
            return 0
        
        # Use predicted rate if available, otherwise current
        rate = info.predicted_rate if info.predicted_rate != 0 else info.funding_rate
        rate_pct = rate * 100
        
        # Average hold time for Lead-Lag: ~2-4 hours
        # Funding period: 8 hours
        # Expected funding events: 0.25 to 0.5
        avg_funding_events = 0.3
        
        if direction == "LONG":
            # LONG pays positive funding, receives negative
            return rate_pct * avg_funding_events
        else:
            # SHORT receives positive funding, pays negative
            return -rate_pct * avg_funding_events


# Singleton instance
_funding_tracker: Optional[FundingTracker] = None


def get_funding_tracker() -> FundingTracker:
    """Get singleton funding tracker instance"""
    global _funding_tracker
    if _funding_tracker is None:
        _funding_tracker = FundingTracker()
    return _funding_tracker
