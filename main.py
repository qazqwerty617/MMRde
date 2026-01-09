"""
MMRdex Bot v4.0 - ULTRA INTELLIGENT Edition
Advanced Lead-Lag arbitrage system with:
- WebSocket real-time MEXC prices
- Parallel DEX scanning
- Token validation & fake detection
- Funding rate calculation
- Convergence analysis & learning
- Momentum confirmation
- Entry point optimization
- ML-like token intelligence
"""
import asyncio
import logging
import sys
import platform

from config import (
    SCAN_INTERVAL_SEC, 
    SPREAD_CHECK_INTERVAL_SEC,
    SIGNAL_COOLDOWN_SEC,
    LOG_LEVEL
)

# Fix for Windows Event Loop Issue (WinError 10022)
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from database import init_db, get_token_stats, get_db
from mexc_client import MEXCClient
from mexc_ws import get_ws_client
from dexscreener_client import DexScreenerClient
from turbo_scanner import TurboScanner, format_turbo_signal
from spread_tracker import SpreadTracker, format_closure_message
from bot import TelegramBot

# Intelligence modules
from funding_tracker import get_funding_tracker
from convergence_analyzer import get_convergence_analyzer
from token_intelligence import get_token_intelligence

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class MMRdexBot:
    """Main bot orchestrator - ULTRA INTELLIGENT edition"""
    
    def __init__(self):
        self.mexc = MEXCClient()
        self.dexscreener = DexScreenerClient()
        self.scanner = TurboScanner(self.mexc, self.dexscreener)
        self.tracker = SpreadTracker(self.mexc, self.dexscreener)
        self.telegram = TelegramBot()
        self.ws = get_ws_client()
        self._running = False
        
        # Intelligence references
        self.funding_tracker = get_funding_tracker()
        self.convergence_analyzer = get_convergence_analyzer()
        self.token_intelligence = get_token_intelligence()
    
    async def _initialize_intelligence(self):
        """Initialize all intelligence modules from database and APIs"""
        logger.info("🧠 Loading intelligence modules...")
        
        try:
            # Load funding rates from MEXC API
            await self.funding_tracker.fetch_all_funding_rates()
            logger.info("✅ Funding rates loaded")
        except Exception as e:
            logger.warning(f"⚠️ Funding rates unavailable: {e}")
        
        try:
            # Load historical convergence data
            db = await get_db()
            await self.convergence_analyzer.load_from_database()
            logger.info("✅ Convergence analyzer loaded")
        except Exception as e:
            logger.warning(f"⚠️ Convergence data unavailable: {e}")
        
        try:
            # Load token intelligence from history
            db = await get_db()
            await self.token_intelligence.load_from_database(db)
            logger.info("✅ Token intelligence loaded")
        except Exception as e:
            logger.warning(f"⚠️ Token intelligence unavailable: {e}")
        
        # Log intelligence summary
        recommended = self.token_intelligence.get_recommended_tokens(min_score=5.0, limit=10)
        avoid = self.token_intelligence.get_avoid_tokens()
        if recommended:
            logger.info(f"🏆 Top tokens: {', '.join([t[0] for t in recommended[:5]])}")
        if avoid:
            logger.info(f"⛔ Avoid tokens: {', '.join(avoid[:5])}")
    
    async def start(self):
        """Start all bot components"""
        logger.info("=" * 50)
        logger.info("🚀 MMRdex ULTRA INTELLIGENT v4.0 Starting...")
        logger.info("=" * 50)
        
        # Initialize database
        await init_db()
        logger.info("✅ Database initialized")
        
        # Start WebSocket for real-time prices
        await self.ws.start()
        logger.info("✅ WebSocket connected")
        
        # Wait for prices to load
        await asyncio.sleep(3)
        logger.info(f"✅ Loaded {len(self.ws.prices)} MEXC prices")
        
        # Initialize intelligence modules
        await self._initialize_intelligence()
        
        self._running = True
        
        # Start all tasks
        await asyncio.gather(
            self._run_scanner(),
            self._run_tracker(),
            self._run_telegram(),
            self._run_funding_refresh(),  # NEW: Periodic funding rate refresh
        )
    
    async def stop(self):
        """Stop all components"""
        self._running = False
        await self.ws.close()
        await self.mexc.close()
        await self.dexscreener.close()
        await self.funding_tracker.close()
        await self.telegram.stop()
        logger.info("Bot stopped")
    
    async def _run_scanner(self):
        """Main scanner loop - ULTRA INTELLIGENT"""
        from chart_generator import generate_spread_chart
        
        logger.info(f"⚡ ULTRA Scanner started (interval: {SCAN_INTERVAL_SEC}s)")
        
        scan_count = 0
        signal_count = 0
        
        while self._running:
            try:
                scan_count += 1
                signals = await self.scanner.scan()
                
                # Send notifications for new signals (sorted by quality)
                for signal in signals:
                    signal_count += 1
                    
                    # Get token statistics
                    token_stats = await get_token_stats(signal.token)
                    message = format_turbo_signal(signal, token_stats)
                    
                    # Generate chart
                    chart_image = None
                    try:
                        klines = await self.mexc.get_kline_data(signal.token, "Min15", limit=48)
                        
                        if klines:
                            chart_image = generate_spread_chart(
                                signal.token, 
                                klines, 
                                signal.dex_price,
                                signal.direction
                            )
                    except Exception as e:
                        logger.error(f"Chart error: {e}")
                    
                    await self.telegram.send_signal(message, chart_image)
                    logger.info(
                        f"📤 Signal: {signal.direction} {signal.token} | "
                        f"Net: +{signal.net_profit:.1f}% | "
                        f"Quality: {signal.quality_score:.1f}/10"
                    )
                    
                    await asyncio.sleep(SIGNAL_COOLDOWN_SEC)
                
                # Periodic stats with intelligence info
                if scan_count % 100 == 0:
                    top_tokens = self.token_intelligence.get_recommended_tokens(n=3)
                    top_str = ", ".join([f"{t[0]}({t[1]:.1f})" for t in top_tokens])
                    logger.info(
                        f"📊 Stats: {scan_count} scans, {signal_count} signals | "
                        f"Top: {top_str}"
                    )
                
            except Exception as e:
                logger.error(f"Scanner error: {e}")
            
            await asyncio.sleep(SCAN_INTERVAL_SEC)
    
    async def _run_tracker(self):
        """Spread tracker loop with learning feedback"""
        logger.info(f"📊 Tracker started (interval: {SPREAD_CHECK_INTERVAL_SEC}s)")
        
        while self._running:
            try:
                closed = await self.tracker.check_closures()
                
                for closure in closed:
                    message = format_closure_message(closure)
                    await self.telegram.send_closure(message)
                    logger.info(
                        f"✅ Closed: {closure.token} | {closure.outcome} | "
                        f"PnL: {closure.price_change_percent:+.1f}%"
                    )
                
            except Exception as e:
                logger.error(f"Tracker error: {e}")
            
            await asyncio.sleep(SPREAD_CHECK_INTERVAL_SEC)
    
    async def _run_telegram(self):
        """Run Telegram bot"""
        try:
            await self.telegram.start()
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    async def _run_funding_refresh(self):
        """Periodically refresh funding rates"""
        refresh_interval = 300  # Every 5 minutes
        
        while self._running:
            await asyncio.sleep(refresh_interval)
            try:
                await self.funding_tracker.fetch_all_funding_rates()
                logger.debug("Funding rates refreshed")
            except Exception as e:
                logger.warning(f"Funding refresh error: {e}")


async def main():
    bot = MMRdexBot()
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await bot.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bye!")
        sys.exit(0)
