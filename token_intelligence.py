"""
Token Intelligence System
ML-like scoring for signal quality based on historical performance.
Learns which tokens are profitable for Lead-Lag trading.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TokenStats:
    """Complete statistics for a token"""
    symbol: str
    
    # Signal history
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    
    # Performance
    total_profit: float = 0  # Sum of all PnL
    avg_profit: float = 0
    best_trade: float = 0
    worst_trade: float = 0
    
    # Timing
    avg_convergence_time: float = 0  # seconds
    fastest_convergence: float = float('inf')
    
    # Direction performance
    long_wins: int = 0
    long_total: int = 0
    short_wins: int = 0
    short_total: int = 0
    
    # Recent performance (last 10 trades)
    recent_outcomes: List[str] = field(default_factory=list)
    
    # Calculated scores
    win_rate: float = 0
    quality_score: float = 5.0  # 0-10 scale
    
    last_updated: float = 0


class TokenIntelligence:
    """
    Intelligent signal scoring based on token history.
    Helps filter out consistently losing tokens.
    """
    
    def __init__(self):
        self._stats: Dict[str, TokenStats] = {}
        self._min_trades_for_scoring = 3
    
    async def load_from_database(self, db):
        """Load historical stats from database"""
        try:
            # Join with signal_outcomes to get outcome data
            query = """
                SELECT 
                    s.token,
                    s.direction,
                    so.outcome,
                    so.price_change_percent,
                    CAST((julianday(s.closed_at) - julianday(s.created_at)) * 86400 AS INTEGER) as duration
                FROM signals s
                JOIN signal_outcomes so ON so.signal_id = s.id
                WHERE s.closed_at IS NOT NULL
                AND s.created_at > datetime('now', '-14 days')
                ORDER BY s.created_at DESC
            """
            
            async with db.execute(query) as cursor:
                rows = await cursor.fetchall()
            
            for row in rows:
                token, direction, outcome, pnl, duration = row
                self.record_outcome(token, direction, outcome, pnl or 0, duration or 0)
            
            # Calculate scores
            for stats in self._stats.values():
                self._calculate_score(stats)
            
            logger.info(f"🧠 Loaded intelligence for {len(self._stats)} tokens")
            
        except Exception as e:
            logger.error(f"Error loading token intelligence: {e}")
    
    def record_outcome(
        self, 
        symbol: str, 
        direction: str,
        outcome: str,  # "win", "lose", "draw"
        profit_percent: float,
        convergence_time: int
    ):
        """Record a trade outcome for learning"""
        if symbol not in self._stats:
            self._stats[symbol] = TokenStats(symbol=symbol)
        
        stats = self._stats[symbol]
        stats.total_signals += 1
        
        # Update win/loss counters
        if outcome == "win":
            stats.wins += 1
            if direction == "LONG":
                stats.long_wins += 1
            else:
                stats.short_wins += 1
        elif outcome == "lose":
            stats.losses += 1
        else:
            stats.draws += 1
        
        # Direction tracking
        if direction == "LONG":
            stats.long_total += 1
        else:
            stats.short_total += 1
        
        # Profit tracking
        stats.total_profit += profit_percent
        if profit_percent > stats.best_trade:
            stats.best_trade = profit_percent
        if profit_percent < stats.worst_trade:
            stats.worst_trade = profit_percent
        
        # Convergence timing
        if convergence_time > 0 and outcome != "lose":
            if convergence_time < stats.fastest_convergence:
                stats.fastest_convergence = convergence_time
            
            # Rolling average
            if stats.avg_convergence_time == 0:
                stats.avg_convergence_time = convergence_time
            else:
                alpha = 0.3
                stats.avg_convergence_time = (
                    alpha * convergence_time + 
                    (1 - alpha) * stats.avg_convergence_time
                )
        
        # Recent outcomes (keep last 10)
        stats.recent_outcomes.append(outcome)
        if len(stats.recent_outcomes) > 10:
            stats.recent_outcomes.pop(0)
        
        # Recalculate averages and score
        stats.avg_profit = stats.total_profit / stats.total_signals
        stats.win_rate = stats.wins / stats.total_signals if stats.total_signals > 0 else 0
        stats.last_updated = time.time()
        
        self._calculate_score(stats)
    
    def _calculate_score(self, stats: TokenStats):
        """Calculate quality score for a token"""
        if stats.total_signals < self._min_trades_for_scoring:
            stats.quality_score = 5.0  # Neutral for unknown
            return
        
        # Components:
        
        # 1. Win rate (35% weight)
        # 50% = 5/10, 80% = 8/10, 100% = 10/10
        win_score = stats.win_rate * 10
        
        # 2. Average profit (25% weight)
        # 0% = 5/10, +5% = 7.5/10, -5% = 2.5/10
        profit_score = 5 + (stats.avg_profit * 0.5)
        profit_score = max(0, min(10, profit_score))
        
        # 3. Speed score (20% weight)
        # <5min = 10/10, >1h = 0/10
        if stats.avg_convergence_time <= 300:
            speed_score = 10
        elif stats.avg_convergence_time >= 3600:
            speed_score = 0
        else:
            speed_score = 10 * (1 - (stats.avg_convergence_time - 300) / 3300)
        
        # 4. Consistency (10% weight) - recent performance
        recent_wins = stats.recent_outcomes.count("win")
        recent_total = len(stats.recent_outcomes)
        if recent_total > 0:
            recent_rate = recent_wins / recent_total
            consistency_score = recent_rate * 10
        else:
            consistency_score = 5
        
        # 5. Sample size confidence (10% weight)
        # More trades = more confidence
        if stats.total_signals >= 20:
            sample_score = 10
        else:
            sample_score = (stats.total_signals / 20) * 10
        
        # Weighted average
        stats.quality_score = (
            win_score * 0.35 +
            profit_score * 0.25 +
            speed_score * 0.20 +
            consistency_score * 0.10 +
            sample_score * 0.10
        )
        
        stats.quality_score = round(stats.quality_score, 2)
    
    def get_stats(self, symbol: str) -> Optional[TokenStats]:
        """Get stats for a token"""
        return self._stats.get(symbol)
    
    def get_score(self, symbol: str) -> float:
        """Get quality score for a token (0-10)"""
        stats = self._stats.get(symbol)
        return stats.quality_score if stats else 5.0
    
    def should_signal(
        self, 
        symbol: str, 
        direction: str,
        min_score: float = 4.0,
        min_win_rate: float = 0.35
    ) -> Tuple[bool, str]:
        """
        Determine if signal should be sent based on history.
        
        Returns:
            (should_signal, reason)
        """
        stats = self._stats.get(symbol)
        
        if not stats:
            return True, "New token, no history"
        
        if stats.total_signals < self._min_trades_for_scoring:
            return True, f"Insufficient data ({stats.total_signals} trades)"
        
        # Check quality score
        if stats.quality_score < min_score:
            return False, f"Low quality score: {stats.quality_score:.1f} < {min_score}"
        
        # Check win rate
        if stats.win_rate < min_win_rate:
            return False, f"Low win rate: {stats.win_rate:.0%} < {min_win_rate:.0%}"
        
        # Check direction-specific performance
        if direction == "LONG" and stats.long_total >= 3:
            long_rate = stats.long_wins / stats.long_total
            if long_rate < 0.3:
                return False, f"Poor LONG performance: {long_rate:.0%}"
        
        if direction == "SHORT" and stats.short_total >= 3:
            short_rate = stats.short_wins / stats.short_total
            if short_rate < 0.3:
                return False, f"Poor SHORT performance: {short_rate:.0%}"
        
        # Check recent streak
        if len(stats.recent_outcomes) >= 5:
            recent_losses = stats.recent_outcomes[-5:].count("lose")
            if recent_losses >= 4:
                return False, "Recent losing streak (4/5 losses)"
        
        return True, f"Score: {stats.quality_score:.1f}, Win: {stats.win_rate:.0%}"
    
    def get_signal_modifier(self, symbol: str, direction: str) -> float:
        """
        Get profit modifier based on token history.
        Used to adjust expected profit.
        
        Returns:
            Multiplier from 0.5 to 1.5
        """
        stats = self._stats.get(symbol)
        
        if not stats or stats.total_signals < 3:
            return 1.0
        
        # Base modifier from score
        score_modifier = 0.5 + (stats.quality_score / 10) * 1.0  # 0.5 to 1.5
        
        # Direction-specific adjustment
        if direction == "LONG" and stats.long_total >= 3:
            dir_rate = stats.long_wins / stats.long_total
            dir_modifier = 0.7 + (dir_rate * 0.6)  # 0.7 to 1.3
        elif direction == "SHORT" and stats.short_total >= 3:
            dir_rate = stats.short_wins / stats.short_total
            dir_modifier = 0.7 + (dir_rate * 0.6)
        else:
            dir_modifier = 1.0
        
        return (score_modifier + dir_modifier) / 2
    
    def get_recommended_tokens(self, min_score: float = 6.0, limit: int = 20) -> List[Tuple[str, float]]:
        """Get list of recommended tokens with scores"""
        recommended = [
            (symbol, stats.quality_score)
            for symbol, stats in self._stats.items()
            if stats.quality_score >= min_score and stats.total_signals >= 5
        ]
        recommended.sort(key=lambda x: x[1], reverse=True)
        return recommended[:limit]
    
    def get_avoid_tokens(self) -> List[str]:
        """Get list of tokens to avoid"""
        return [
            symbol for symbol, stats in self._stats.items()
            if stats.quality_score < 3.0 and stats.total_signals >= 5
        ]


# Singleton
_token_intelligence: Optional[TokenIntelligence] = None


def get_token_intelligence() -> TokenIntelligence:
    """Get singleton token intelligence"""
    global _token_intelligence
    if _token_intelligence is None:
        _token_intelligence = TokenIntelligence()
    return _token_intelligence
