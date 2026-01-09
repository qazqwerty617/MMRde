"""
Spread Tracker v2.0
Enhanced tracking with:
- WebSocket real-time MEXC prices
- Learning feedback to intelligence modules
- Faster closure detection
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config import SPREAD_CLOSURE_THRESHOLD
from mexc_client import MEXCClient
from mexc_ws import get_ws_client
from dexscreener_client import DexScreenerClient
from database import get_active_signals, close_signal

# Intelligence feedback
from convergence_analyzer import get_convergence_analyzer
from token_intelligence import get_token_intelligence

logger = logging.getLogger(__name__)


@dataclass
class ClosedSignal:
    """Represents a closed arbitrage signal"""
    signal_id: int
    token: str
    direction: str
    chain: str
    initial_spread: float
    final_spread: float
    price_change_percent: float
    outcome: str  # win, draw, lose
    align_seconds: int  # Time until spread closed


class SpreadTracker:
    """
    Enhanced spread tracker with WebSocket support and learning feedback.
    """
    
    def __init__(self, mexc: MEXCClient, dexscreener: DexScreenerClient):
        self.mexc = mexc
        self.dexscreener = dexscreener
        self.ws = get_ws_client()
        
        # Intelligence modules for feedback
        self.convergence_analyzer = get_convergence_analyzer()
        self.token_intelligence = get_token_intelligence()
    
    async def check_closures(self) -> list[ClosedSignal]:
        """
        Check all active signals for spread closure.
        Uses WebSocket for faster MEXC price updates.
        Feeds results back to intelligence modules.
        """
        closed = []
        
        # Get active signals from database
        active_signals = await get_active_signals()
        if not active_signals:
            return []
        
        # Try WebSocket prices first (faster!), fallback to REST
        ws_prices = self.ws.prices
        if not ws_prices:
            tickers_list = await self.mexc.get_futures_tickers()
            ws_prices = {t[0]: t[1] for t in tickers_list}
        
        for signal in active_signals:
            try:
                token = signal["token"]
                
                # Get current MEXC price (from WS or REST)
                current_mexc_price = ws_prices.get(token)
                if not current_mexc_price:
                    continue
                
                # Get current DEX price
                dex_pair = await self.dexscreener.get_best_dex_price(token)
                if not dex_pair:
                    continue
                
                current_dex_price = dex_pair["price_usd"]
                
                # Calculate current spread
                current_spread = abs((current_dex_price - current_mexc_price) / current_mexc_price * 100)
                
                # Check if spread has closed
                if current_spread < SPREAD_CLOSURE_THRESHOLD:
                    # Calculate price change from signal
                    original_mexc_price = signal["mexc_price"]
                    price_change = ((current_mexc_price - original_mexc_price) / original_mexc_price) * 100
                    
                    # Adjust for direction
                    if signal["direction"] == "SHORT":
                        price_change = -price_change
                    
                    # Close the signal in DB
                    outcome = await close_signal(
                        signal_id=signal["id"],
                        final_spread=current_spread,
                        price_change_percent=price_change
                    )
                    
                    # Calculate alignment time
                    created_at = signal.get("created_at", "")
                    align_seconds = 0
                    if created_at:
                        try:
                            created_time = datetime.fromisoformat(created_at)
                            align_seconds = int((datetime.utcnow() - created_time).total_seconds())
                        except:
                            pass
                    
                    # ===== INTELLIGENCE FEEDBACK =====
                    # Record to convergence analyzer
                    converged = outcome in ["win", "draw"]
                    self.convergence_analyzer.record_convergence(
                        symbol=token,
                        converged=converged,
                        time_seconds=align_seconds,
                        profit_percent=price_change
                    )
                    
                    # Record to token intelligence
                    self.token_intelligence.record_outcome(
                        symbol=token,
                        direction=signal["direction"],
                        outcome=outcome,
                        profit_percent=price_change,
                        convergence_time=align_seconds
                    )
                    
                    closed_signal = ClosedSignal(
                        signal_id=signal["id"],
                        token=token,
                        direction=signal["direction"],
                        chain=signal.get("chain", "unknown"),
                        initial_spread=signal["spread_percent"],
                        final_spread=current_spread,
                        price_change_percent=price_change,
                        outcome=outcome,
                        align_seconds=align_seconds
                    )
                    closed.append(closed_signal)
                    
                    # Log with intelligence info
                    token_score = self.token_intelligence.get_score(token)
                    logger.info(
                        f"Spread closed: {token} | {outcome.upper()} | "
                        f"PnL: {price_change:+.1f}% | Time: {align_seconds}s | "
                        f"Token Score: {token_score:.1f}"
                    )
                    
            except Exception as e:
                logger.error(f"Error checking signal {signal.get('id')}: {e}")
                continue
        
        return closed


def format_closure_message(closed: ClosedSignal) -> str:
    """Format spread closure notification with learning stats"""
    # Format align time
    if closed.align_seconds >= 3600:
        align_str = f"{closed.align_seconds // 3600}h {(closed.align_seconds % 3600) // 60}m"
    elif closed.align_seconds >= 60:
        align_str = f"{closed.align_seconds // 60}m {closed.align_seconds % 60}s"
    else:
        align_str = f"{closed.align_seconds}s"
    
    # Outcome styling
    if closed.outcome == "win":
        emoji = "✅"
        pnl_emoji = "🟢"
        outcome_text = "WIN"
    elif closed.outcome == "lose":
        emoji = "❌"
        pnl_emoji = "🔴"
        outcome_text = "LOSS"
    else:
        emoji = "➖"
        pnl_emoji = "🟠"
        outcome_text = "DRAW"
    
    # Get token stats for context
    token_intel = get_token_intelligence()
    stats = token_intel.get_stats(closed.token)
    
    if stats and stats.total_signals >= 3:
        stats_line = f"📊 Token: {stats.win_rate:.0%} Win ({stats.total_signals} trades)\n"
    else:
        stats_line = ""
    
    return (
        f"{emoji} <b>{outcome_text}</b> #{closed.token} #{closed.chain.upper()}\n"
        f"⏱️ Aligned in {align_str}\n"
        f"{pnl_emoji} PnL: {closed.price_change_percent:+.1f}%\n"
        f"{stats_line}"
        f"📉 Spread: {closed.initial_spread:.1f}% → {closed.final_spread:.1f}%"
    )
