"""
Momentum Tracker
Tracks DEX price momentum to confirm trend direction.
Only signals when DEX momentum aligns with spread direction.
"""
import logging
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PricePoint:
    """Single price data point"""
    price: float
    timestamp: float


@dataclass
class MomentumData:
    """Momentum analysis result"""
    symbol: str
    current_price: float
    price_1m: Optional[float]  # Price 1 minute ago
    price_5m: Optional[float]  # Price 5 minutes ago
    change_1m: float  # % change in last 1 min
    change_5m: float  # % change in last 5 min
    trend: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-10 scale
    last_updated: float


class MomentumTracker:
    """
    Tracks DEX price changes over time to detect momentum.
    Used to confirm signals: only signal when momentum matches direction.
    """
    
    def __init__(self, max_history_sec: int = 600):
        # {symbol: deque of PricePoints}
        self._price_history: Dict[str, deque] = {}
        self._max_history = max_history_sec  # Keep 10 minutes of history
        self._momentum_cache: Dict[str, MomentumData] = {}
    
    def record_price(self, symbol: str, price: float):
        """
        Record a new DEX price for momentum tracking.
        Should be called every time we get DEX price.
        """
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=1000)
        
        now = time.time()
        self._price_history[symbol].append(PricePoint(price, now))
        
        # Clean old entries
        cutoff = now - self._max_history
        while (self._price_history[symbol] and 
               self._price_history[symbol][0].timestamp < cutoff):
            self._price_history[symbol].popleft()
    
    def _get_price_at_age(self, symbol: str, age_seconds: int) -> Optional[float]:
        """Get price approximately X seconds ago"""
        history = self._price_history.get(symbol)
        if not history:
            return None
        
        now = time.time()
        target_time = now - age_seconds
        
        # Find closest price point
        closest = None
        closest_diff = float('inf')
        
        for point in history:
            diff = abs(point.timestamp - target_time)
            if diff < closest_diff:
                closest_diff = diff
                closest = point
        
        # Only return if within 30% tolerance
        if closest and closest_diff < age_seconds * 0.3:
            return closest.price
        
        return None
    
    def analyze_momentum(self, symbol: str, current_price: float) -> MomentumData:
        """
        Analyze price momentum for a symbol.
        
        Args:
            symbol: Token symbol
            current_price: Current DEX price
            
        Returns:
            MomentumData with trend and strength
        """
        # Record current price
        self.record_price(symbol, current_price)
        
        # Get historical prices
        price_1m = self._get_price_at_age(symbol, 60)
        price_5m = self._get_price_at_age(symbol, 300)
        
        # Calculate changes
        change_1m = 0
        change_5m = 0
        
        if price_1m and price_1m > 0:
            change_1m = ((current_price - price_1m) / price_1m) * 100
        
        if price_5m and price_5m > 0:
            change_5m = ((current_price - price_5m) / price_5m) * 100
        
        # Determine trend
        trend, strength = self._calculate_trend(change_1m, change_5m)
        
        momentum = MomentumData(
            symbol=symbol,
            current_price=current_price,
            price_1m=price_1m,
            price_5m=price_5m,
            change_1m=change_1m,
            change_5m=change_5m,
            trend=trend,
            strength=strength,
            last_updated=time.time()
        )
        
        self._momentum_cache[symbol] = momentum
        return momentum
    
    def _calculate_trend(self, change_1m: float, change_5m: float) -> Tuple[str, float]:
        """
        Calculate trend direction and strength.
        
        Returns:
            (trend: "bullish"/"bearish"/"neutral", strength: 0-10)
        """
        # Weight short-term more heavily
        weighted_change = change_1m * 0.6 + change_5m * 0.4
        
        # Trend thresholds
        if weighted_change > 0.5:
            trend = "bullish"
        elif weighted_change < -0.5:
            trend = "bearish"
        else:
            trend = "neutral"
        
        # Strength calculation (0-10 scale)
        # Based on magnitude of change
        strength = min(10, abs(weighted_change) * 2)
        
        # Bonus for consistent direction
        if change_1m * change_5m > 0:  # Same direction
            strength = min(10, strength * 1.3)
        elif change_1m * change_5m < 0:  # Opposite direction
            strength *= 0.7
        
        return trend, round(strength, 1)
    
    def get_cached_momentum(self, symbol: str) -> Optional[MomentumData]:
        """Get cached momentum data"""
        return self._momentum_cache.get(symbol)
    
    def confirms_direction(
        self, 
        symbol: str, 
        direction: str,
        min_strength: float = 2.0
    ) -> Tuple[bool, str]:
        """
        Check if momentum confirms the signal direction.
        
        Args:
            symbol: Token symbol
            direction: "LONG" or "SHORT"
            min_strength: Minimum momentum strength required
            
        Returns:
            (confirms: bool, reason: str)
        """
        momentum = self._momentum_cache.get(symbol)
        
        if not momentum:
            return True, "No momentum data"
        
        # Check if momentum aligns with direction
        if direction == "LONG":
            if momentum.trend == "bearish" and momentum.strength >= min_strength:
                return False, f"Bearish momentum ({momentum.change_1m:+.1f}% 1m) contradicts LONG"
            if momentum.trend == "bullish" and momentum.strength >= min_strength:
                return True, f"Strong bullish momentum confirms LONG (+{momentum.strength:.1f})"
                
        elif direction == "SHORT":
            if momentum.trend == "bullish" and momentum.strength >= min_strength:
                return False, f"Bullish momentum ({momentum.change_1m:+.1f}% 1m) contradicts SHORT"
            if momentum.trend == "bearish" and momentum.strength >= min_strength:
                return True, f"Strong bearish momentum confirms SHORT (+{momentum.strength:.1f})"
        
        # Neutral or weak momentum - don't block
        return True, f"Neutral/weak momentum (strength: {momentum.strength:.1f})"
    
    def get_momentum_bonus(self, symbol: str, direction: str) -> float:
        """
        Get a bonus multiplier for signal quality based on momentum alignment.
        
        Returns:
            Multiplier from 0.5 (bad) to 1.5 (excellent)
        """
        momentum = self._momentum_cache.get(symbol)
        
        if not momentum:
            return 1.0  # Neutral
        
        aligned = (
            (direction == "LONG" and momentum.trend == "bullish") or
            (direction == "SHORT" and momentum.trend == "bearish")
        )
        
        opposite = (
            (direction == "LONG" and momentum.trend == "bearish") or
            (direction == "SHORT" and momentum.trend == "bullish")
        )
        
        if aligned:
            return 1.0 + (momentum.strength * 0.05)  # Up to 1.5x
        elif opposite:
            return 1.0 - (momentum.strength * 0.05)  # Down to 0.5x
        else:
            return 1.0  # Neutral


# Singleton
_momentum_tracker: Optional[MomentumTracker] = None


def get_momentum_tracker() -> MomentumTracker:
    """Get singleton momentum tracker"""
    global _momentum_tracker
    if _momentum_tracker is None:
        _momentum_tracker = MomentumTracker()
    return _momentum_tracker
