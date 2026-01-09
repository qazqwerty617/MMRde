"""
Convergence Analyzer
Tracks historical spread convergence speed for each token.
Key for understanding which tokens are good for Lead-Lag.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceStats:
    """Statistics about token convergence behavior"""
    symbol: str
    total_signals: int = 0
    converged_signals: int = 0
    avg_convergence_time_sec: float = 0  # Average time for spread to close
    fastest_convergence_sec: float = float('inf')
    slowest_convergence_sec: float = 0
    convergence_rate: float = 0  # % of signals that converged
    avg_profit_on_converge: float = 0  # Average PnL when converged
    last_updated: float = 0


class ConvergenceAnalyzer:
    """
    Analyzes and tracks how quickly spreads converge for each token.
    Used to prioritize fast-converging tokens and filter slow/dead spreads.
    """
    
    def __init__(self, db=None):
        self._stats_cache: Dict[str, ConvergenceStats] = {}
        self._db = db  # Optional database connection
        self._blacklist: set = set()  # Tokens that rarely converge
        self._whitelist: set = set()  # Tokens that converge well
    
    async def load_from_database(self):
        """Load historical convergence stats from database"""
        if not self._db:
            return
        
        try:
            # Query closed signals with convergence time
            query = """
                SELECT 
                    token,
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome = 'win' OR outcome = 'draw' THEN 1 ELSE 0 END) as converged,
                    AVG(CASE WHEN closed_at IS NOT NULL 
                        THEN CAST((julianday(closed_at) - julianday(created_at)) * 86400 AS INTEGER)
                        ELSE NULL END) as avg_time,
                    MIN(CASE WHEN closed_at IS NOT NULL 
                        THEN CAST((julianday(closed_at) - julianday(created_at)) * 86400 AS INTEGER)
                        ELSE NULL END) as min_time,
                    MAX(CASE WHEN closed_at IS NOT NULL 
                        THEN CAST((julianday(closed_at) - julianday(created_at)) * 86400 AS INTEGER)
                        ELSE NULL END) as max_time,
                    AVG(CASE WHEN outcome = 'win' THEN price_change_percent ELSE 0 END) as avg_profit
                FROM signals
                WHERE created_at > datetime('now', '-7 days')
                GROUP BY token
                HAVING total >= 3
            """
            
            async with self._db.execute(query) as cursor:
                rows = await cursor.fetchall()
                
                for row in rows:
                    token = row[0]
                    total = row[1]
                    converged = row[2] or 0
                    avg_time = row[3] or 3600
                    min_time = row[4] or 0
                    max_time = row[5] or 0
                    avg_profit = row[6] or 0
                    
                    stats = ConvergenceStats(
                        symbol=token,
                        total_signals=total,
                        converged_signals=converged,
                        avg_convergence_time_sec=avg_time,
                        fastest_convergence_sec=min_time,
                        slowest_convergence_sec=max_time,
                        convergence_rate=converged / total if total > 0 else 0,
                        avg_profit_on_converge=avg_profit,
                        last_updated=time.time()
                    )
                    self._stats_cache[token] = stats
                    
                    # Update whitelist/blacklist
                    if stats.convergence_rate >= 0.6:
                        self._whitelist.add(token)
                    elif stats.convergence_rate < 0.3 and total >= 5:
                        self._blacklist.add(token)
                
                logger.info(f"📊 Loaded convergence stats for {len(self._stats_cache)} tokens")
                logger.info(f"   Whitelist: {len(self._whitelist)}, Blacklist: {len(self._blacklist)}")
                
        except Exception as e:
            logger.error(f"Error loading convergence stats: {e}")
    
    def record_convergence(
        self, 
        symbol: str, 
        converged: bool,
        time_seconds: int,
        profit_percent: float
    ):
        """
        Record a new convergence event.
        
        Args:
            symbol: Token symbol
            converged: Whether spread closed within threshold
            time_seconds: Time to convergence (or max tracking time if not converged)
            profit_percent: PnL percentage
        """
        stats = self._stats_cache.get(symbol)
        
        if stats is None:
            stats = ConvergenceStats(symbol=symbol)
            self._stats_cache[symbol] = stats
        
        # Update stats
        stats.total_signals += 1
        if converged:
            stats.converged_signals += 1
            
            # Update timing stats
            if time_seconds < stats.fastest_convergence_sec:
                stats.fastest_convergence_sec = time_seconds
            if time_seconds > stats.slowest_convergence_sec:
                stats.slowest_convergence_sec = time_seconds
            
            # Rolling average of convergence time (exponential moving average)
            alpha = 0.3
            if stats.avg_convergence_time_sec == 0:
                stats.avg_convergence_time_sec = time_seconds
            else:
                stats.avg_convergence_time_sec = (
                    alpha * time_seconds + 
                    (1 - alpha) * stats.avg_convergence_time_sec
                )
            
            # Rolling average of profit
            if stats.avg_profit_on_converge == 0:
                stats.avg_profit_on_converge = profit_percent
            else:
                stats.avg_profit_on_converge = (
                    alpha * profit_percent + 
                    (1 - alpha) * stats.avg_profit_on_converge
                )
        
        stats.convergence_rate = (
            stats.converged_signals / stats.total_signals 
            if stats.total_signals > 0 else 0
        )
        stats.last_updated = time.time()
        
        # Update whitelist/blacklist
        if stats.total_signals >= 5:
            if stats.convergence_rate >= 0.6:
                self._whitelist.add(symbol)
                self._blacklist.discard(symbol)
            elif stats.convergence_rate < 0.3:
                self._blacklist.add(symbol)
                self._whitelist.discard(symbol)
    
    def get_stats(self, symbol: str) -> Optional[ConvergenceStats]:
        """Get convergence stats for a symbol"""
        return self._stats_cache.get(symbol)
    
    def is_blacklisted(self, symbol: str) -> bool:
        """Check if token is blacklisted due to poor convergence"""
        return symbol in self._blacklist
    
    def is_whitelisted(self, symbol: str) -> bool:
        """Check if token is whitelisted for good convergence"""
        return symbol in self._whitelist
    
    def get_priority_score(self, symbol: str) -> float:
        """
        Calculate priority score for a token based on convergence stats.
        Higher = better for trading.
        
        Returns:
            Score from 0-10
        """
        stats = self._stats_cache.get(symbol)
        
        if stats is None:
            return 5.0  # Unknown token = neutral score
        
        if stats.total_signals < 3:
            return 5.0  # Not enough data
        
        # Components (all normalized to 0-10):
        
        # 1. Convergence rate (40% weight)
        rate_score = stats.convergence_rate * 10
        
        # 2. Speed score (30% weight) - faster is better
        # Ideal: <300s, Bad: >3600s
        if stats.avg_convergence_time_sec <= 0:
            speed_score = 5
        elif stats.avg_convergence_time_sec <= 300:
            speed_score = 10
        elif stats.avg_convergence_time_sec >= 3600:
            speed_score = 0
        else:
            # Linear interpolation between 300s and 3600s
            speed_score = 10 * (1 - (stats.avg_convergence_time_sec - 300) / 3300)
        
        # 3. Profit score (30% weight)
        # Ideal: >5%, Bad: <0%
        profit_score = min(10, max(0, stats.avg_profit_on_converge * 2))
        
        total_score = (
            rate_score * 0.4 +
            speed_score * 0.3 +
            profit_score * 0.3
        )
        
        return round(total_score, 2)
    
    def should_signal(self, symbol: str, min_score: float = 4.0) -> Tuple[bool, str]:
        """
        Determine if a signal should be sent based on convergence history.
        
        Args:
            symbol: Token symbol
            min_score: Minimum priority score required
            
        Returns:
            (should_signal, reason)
        """
        if symbol in self._blacklist:
            return False, "Token blacklisted (poor convergence history)"
        
        stats = self._stats_cache.get(symbol)
        
        if stats is None:
            return True, "New token, no history"
        
        if stats.total_signals < 3:
            return True, "Insufficient data"
        
        score = self.get_priority_score(symbol)
        
        if score < min_score:
            return False, f"Low priority score: {score:.1f} < {min_score}"
        
        if stats.convergence_rate < 0.3:
            return False, f"Low convergence rate: {stats.convergence_rate:.0%}"
        
        return True, f"Score: {score:.1f}, Rate: {stats.convergence_rate:.0%}"
    
    def get_top_tokens(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top N tokens by priority score"""
        scored = [
            (symbol, self.get_priority_score(symbol))
            for symbol in self._stats_cache.keys()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]


# Singleton
_convergence_analyzer: Optional[ConvergenceAnalyzer] = None


def get_convergence_analyzer() -> ConvergenceAnalyzer:
    """Get singleton convergence analyzer"""
    global _convergence_analyzer
    if _convergence_analyzer is None:
        _convergence_analyzer = ConvergenceAnalyzer()
    return _convergence_analyzer
