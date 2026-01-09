"""
Entry Validator
Validates that it's the right time to enter a position.
Checks that MEXC hasn't started moving yet (optimal entry).
"""
import logging
import time
from typing import Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PriceHistory:
    """Price history for a symbol"""
    prices: deque  # (timestamp, price) tuples
    max_age: int = 120  # Keep 2 minutes of history


class EntryValidator:
    """
    Validates entry points by checking MEXC price movement.
    If MEXC has already started moving toward DEX, entry is too late.
    """
    
    def __init__(self, ws_client=None):
        self._ws = ws_client
        self._price_history: Dict[str, PriceHistory] = {}
        self._last_prices: Dict[str, float] = {}
    
    def set_ws_client(self, ws_client):
        """Set WebSocket client for price updates"""
        self._ws = ws_client
    
    def record_price(self, symbol: str, price: float):
        """Record a price update"""
        now = time.time()
        
        if symbol not in self._price_history:
            self._price_history[symbol] = PriceHistory(
                prices=deque(maxlen=500)
            )
        
        history = self._price_history[symbol]
        history.prices.append((now, price))
        
        # Clean old entries
        cutoff = now - history.max_age
        while history.prices and history.prices[0][0] < cutoff:
            history.prices.popleft()
        
        self._last_prices[symbol] = price
    
    def get_recent_movement(self, symbol: str, seconds: int = 60) -> Optional[float]:
        """
        Get price change over the last N seconds.
        
        Returns:
            Percentage change, or None if not enough data
        """
        # Try WebSocket first for real-time data
        current_price = self._last_prices.get(symbol)
        if self._ws:
            ws_price = self._ws.get_price(symbol)
            if ws_price:
                current_price = ws_price
                self.record_price(symbol, ws_price)
        
        if not current_price:
            return None
        
        history = self._price_history.get(symbol)
        if not history or not history.prices:
            return None
        
        # Find price from N seconds ago
        now = time.time()
        target_time = now - seconds
        
        old_price = None
        for ts, price in history.prices:
            if ts <= target_time:
                old_price = price
            else:
                break
        
        if old_price is None or old_price <= 0:
            return None
        
        return ((current_price - old_price) / old_price) * 100
    
    def validate_entry(
        self, 
        symbol: str, 
        direction: str,
        spread_percent: float,
        max_movement: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Validate that entry timing is optimal.
        
        Args:
            symbol: Token symbol
            direction: "LONG" or "SHORT"
            spread_percent: Current spread percentage
            max_movement: Max MEXC movement allowed (%)
            
        Returns:
            (valid: bool, reason: str)
        """
        # Check 30-second movement
        movement_30s = self.get_recent_movement(symbol, 30)
        if movement_30s is not None:
            if direction == "LONG" and movement_30s > max_movement:
                return False, f"MEXC already rising (+{movement_30s:.1f}% in 30s)"
            if direction == "SHORT" and movement_30s < -max_movement:
                return False, f"MEXC already falling ({movement_30s:.1f}% in 30s)"
        
        # Check 60-second movement
        movement_60s = self.get_recent_movement(symbol, 60)
        if movement_60s is not None:
            if direction == "LONG" and movement_60s > max_movement * 1.5:
                return False, f"MEXC moving up fast (+{movement_60s:.1f}% in 1m)"
            if direction == "SHORT" and movement_60s < -max_movement * 1.5:
                return False, f"MEXC moving down fast ({movement_60s:.1f}% in 1m)"
        
        # Check if movement closes significant portion of spread
        if movement_30s is not None and abs(movement_30s) > spread_percent * 0.3:
            return False, f"Spread already closing ({movement_30s:.1f}% of {spread_percent:.1f}%)"
        
        return True, "Entry timing optimal"
    
    def get_entry_quality(
        self, 
        symbol: str, 
        direction: str,
        spread_percent: float
    ) -> float:
        """
        Get entry quality score (0-10).
        Higher = better entry point.
        """
        movement_30s = self.get_recent_movement(symbol, 30)
        movement_60s = self.get_recent_movement(symbol, 60)
        
        score = 10.0  # Start with perfect score
        
        if movement_30s is None and movement_60s is None:
            return 7.0  # No data = assume moderate
        
        # Penalize if MEXC already moving in signal direction
        if movement_30s is not None:
            if direction == "LONG" and movement_30s > 0:
                score -= movement_30s * 5  # -5 points per 1% movement
            elif direction == "SHORT" and movement_30s < 0:
                score -= abs(movement_30s) * 5
        
        if movement_60s is not None:
            if direction == "LONG" and movement_60s > 0:
                score -= movement_60s * 2
            elif direction == "SHORT" and movement_60s < 0:
                score -= abs(movement_60s) * 2
        
        # Bonus if MEXC moving OPPOSITE to signal direction (perfect entry!)
        if movement_30s is not None:
            if direction == "LONG" and movement_30s < 0:
                score += 2  # MEXC dipping = perfect LONG entry
            elif direction == "SHORT" and movement_30s > 0:
                score += 2  # MEXC pumping = perfect SHORT entry
        
        return max(0, min(10, score))
    
    def get_optimal_entry_delay(self, symbol: str, direction: str) -> int:
        """
        Recommend delay before entry (in seconds).
        If MEXC moving fast, wait for pullback.
        
        Returns:
            Recommended delay in seconds (0 = enter now)
        """
        movement = self.get_recent_movement(symbol, 30)
        
        if movement is None:
            return 0
        
        # If MEXC moving in our direction, wait
        if direction == "LONG" and movement > 0.3:
            return min(60, int(movement * 20))  # 20s per 1% movement
        if direction == "SHORT" and movement < -0.3:
            return min(60, int(abs(movement) * 20))
        
        return 0


# Singleton
_entry_validator: Optional[EntryValidator] = None


def get_entry_validator() -> EntryValidator:
    """Get singleton entry validator"""
    global _entry_validator
    if _entry_validator is None:
        _entry_validator = EntryValidator()
    return _entry_validator
