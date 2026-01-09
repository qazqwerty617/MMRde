"""
Turbo Scanner v4.0 - ULTRA INTELLIGENT EDITION
Advanced Lead-Lag arbitrage scanner with:
- Funding rate cost calculation
- Convergence speed analysis
- DEX momentum tracking
- Token intelligence scoring
- Optimal entry validation
- Real-time WebSocket prices
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, List
from collections import defaultdict

from config import (
    MIN_SPREAD_PERCENT, MAX_SPREAD_PERCENT, MIN_LIQUIDITY_USD, 
    MIN_VOLUME_24H_USD, MIN_FDV_USD, DEXSCREENER_BATCH_SIZE, 
    TOKEN_BLACKLIST, TOTAL_FEES_PERCENT, MAJOR_TOKENS, MIN_TXNS_24H
)
from mexc_client import MEXCClient
from mexc_ws import get_ws_client, MEXCWebSocket
from dexscreener_client import DexScreenerClient, get_chain_display_name
from database import save_signal, check_signal_exists, save_price_history
from pair_manager import PairManager
from token_validator import get_validator

# NEW: Intelligence modules
from funding_tracker import get_funding_tracker, FundingTracker
from convergence_analyzer import get_convergence_analyzer, ConvergenceAnalyzer
from momentum_tracker import get_momentum_tracker, MomentumTracker
from token_intelligence import get_token_intelligence, TokenIntelligence
from entry_validator import get_entry_validator, EntryValidator

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageSignal:
    token: str
    direction: Literal["LONG", "SHORT"]
    spread_percent: float
    net_profit: float
    mexc_price: float
    dex_price: float
    dex_url: str
    chain: str
    liquidity_usd: float = 0
    volume_24h: float = 0
    order_book_depth: float = 0
    detected_at: float = field(default_factory=time.time)
    # NEW: Intelligence data
    quality_score: float = 5.0
    funding_cost: float = 0
    momentum_strength: float = 0
    entry_quality: float = 5.0
    convergence_time_est: float = 0  # Estimated time to converge (seconds)


class TurboScanner:
    """
    Ultra-fast scanner using WebSocket + parallel requests.
    v4.0: Enhanced with intelligence modules for better signal quality.
    """
    
    def __init__(self, mexc_client: MEXCClient, dexscreener_client: DexScreenerClient):
        self.mexc = mexc_client
        self.dexscreener = dexscreener_client
        self.pair_manager = PairManager(dexscreener_client)
        self.validator = get_validator()
        self.ws = get_ws_client()
        
        # NEW: Intelligence modules
        self.funding_tracker = get_funding_tracker()
        self.convergence_analyzer = get_convergence_analyzer()
        self.momentum_tracker = get_momentum_tracker()
        self.token_intelligence = get_token_intelligence()
        self.entry_validator = get_entry_validator()
        
        # Connect entry validator to WebSocket
        self.entry_validator.set_ws_client(self.ws)
        
        # Caching for speed
        self._dex_cache: Dict[str, dict] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 5
        
        # Signal cooldowns (prevent spam)
        self._signal_cooldowns: Dict[str, float] = {}
        self._cooldown_sec = 300  # 5 min cooldown per token+direction
        
        self._discovery_task = None
        self._ws_started = False
        self._funding_loaded = False
        
        # Quality thresholds
        self.min_quality_score = 4.0  # Minimum token intelligence score
        self.min_entry_quality = 3.0  # Minimum entry timing quality
    
    async def initialize_intelligence(self):
        """Load intelligence data from database and APIs"""
        try:
            # Load funding rates
            await self.funding_tracker.fetch_all_funding_rates()
            self._funding_loaded = True
            logger.info("✅ Funding rates loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load funding rates: {e}")
        
        # Note: Convergence and Token Intelligence are loaded from DB in main.py
    
    async def start_ws(self):
        """Initialize WebSocket connection"""
        if not self._ws_started:
            await self.ws.start()
            self._ws_started = True
            await asyncio.sleep(2)
            logger.info(f"📊 WebSocket loaded {len(self.ws.prices)} prices")
            
            # Initialize intelligence after WS is connected
            if not self._funding_loaded:
                asyncio.create_task(self.initialize_intelligence())
    
    async def start_discovery(self, tickers: dict[str, float]):
        """Start background pair discovery"""
        if self._discovery_task is None or self._discovery_task.done():
            self._discovery_task = asyncio.create_task(
                self.pair_manager.discover_pairs(tickers)
            )
    
    def _is_on_cooldown(self, token: str, direction: str) -> bool:
        """Check if we recently sent this signal"""
        key = f"{token}_{direction}"
        last_time = self._signal_cooldowns.get(key, 0)
        return (time.time() - last_time) < self._cooldown_sec
    
    def _set_cooldown(self, token: str, direction: str):
        """Set cooldown for this signal"""
        key = f"{token}_{direction}"
        self._signal_cooldowns[key] = time.time()
    
    async def scan(self) -> List[ArbitrageSignal]:
        """
        TURBO SCAN v4.0 - Intelligent scanning
        Uses WebSocket prices + parallel DEX requests + intelligence filters
        """
        signals = []
        start_time = time.time()
        
        # Ensure WebSocket is running
        await self.start_ws()
        
        # Get real-time prices from WebSocket (instant!)
        mexc_prices = self.ws.prices
        
        if not mexc_prices:
            # Fallback to REST if WS not ready
            tickers = await self.mexc.get_futures_tickers()
            mexc_prices = {t[0]: t[1] for t in tickers}
        
        if not mexc_prices:
            return []
        
        # Record prices for entry validation
        for symbol, price in mexc_prices.items():
            self.entry_validator.record_price(symbol, price)
        
        # Background discovery
        await self.start_discovery(mexc_prices)
        
        # Get batch candidates
        batches = self.pair_manager.get_batch_candidates()
        if not batches:
            return []
        
        # PARALLEL processing - scan all chains at once
        tasks = []
        for chain, addresses in batches.items():
            for i in range(0, len(addresses), DEXSCREENER_BATCH_SIZE):
                chunk = addresses[i:i + DEXSCREENER_BATCH_SIZE]
                tasks.append(self._scan_batch(chain, chunk, mexc_prices))
        
        # Execute all batches in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    signals.extend(result)
        
        # Sort by quality score (best first)
        signals.sort(key=lambda s: s.quality_score, reverse=True)
        
        scan_time = (time.time() - start_time) * 1000
        if signals:
            logger.info(f"⚡ Scan completed in {scan_time:.0f}ms - Found {len(signals)} signals")
        
        return signals
    
    async def _scan_batch(
        self, 
        chain: str, 
        addresses: List[str],
        mexc_prices: Dict[str, float]
    ) -> List[ArbitrageSignal]:
        """Scan a single batch of addresses"""
        signals = []
        
        try:
            pairs = await self.dexscreener.get_pairs_by_addresses(chain, addresses)
            
            for pair in pairs:
                signal = await self._process_pair(pair, chain, mexc_prices)
                if signal:
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"Batch scan error ({chain}): {e}")
        
        return signals
    
    async def _process_pair(
        self,
        pair: dict,
        chain: str,
        mexc_prices: Dict[str, float]
    ) -> Optional[ArbitrageSignal]:
        """Process single pair with FULL intelligence pipeline"""
        symbol = pair.get("symbol", "")
        dex_price = pair.get("price_usd", 0)
        
        # ===== STAGE 1: Quick filters (fast rejection) =====
        if not symbol or symbol in TOKEN_BLACKLIST:
            return None
        if symbol not in mexc_prices:
            return None
        
        mexc_price = mexc_prices[symbol]
        if mexc_price <= 0 or dex_price <= 0:
            return None
        
        # Calculate spread
        spread = ((dex_price - mexc_price) / mexc_price) * 100
        abs_spread = abs(spread)
        direction = "LONG" if spread > 0 else "SHORT"
        
        # Quick spread filter
        if abs_spread < MIN_SPREAD_PERCENT or abs_spread > MAX_SPREAD_PERCENT:
            return None
        
        # Cooldown check (prevent spam)
        if self._is_on_cooldown(symbol, direction):
            return None
        
        # ===== STAGE 2: Track momentum =====
        momentum = self.momentum_tracker.analyze_momentum(symbol, dex_price)
        
        # ===== STAGE 3: Token Intelligence check =====
        should_signal, intel_reason = self.token_intelligence.should_signal(
            symbol, direction, 
            min_score=self.min_quality_score,
            min_win_rate=0.35
        )
        if not should_signal:
            logger.debug(f"[INTEL] Skip {symbol}: {intel_reason}")
            return None
        
        # ===== STAGE 4: Convergence check =====
        should_signal, conv_reason = self.convergence_analyzer.should_signal(
            symbol, min_score=self.min_quality_score
        )
        if not should_signal:
            logger.debug(f"[CONVERGENCE] Skip {symbol}: {conv_reason}")
            return None
        
        # ===== STAGE 5: Momentum confirmation =====
        momentum_ok, momentum_reason = self.momentum_tracker.confirms_direction(
            symbol, direction, min_strength=2.0
        )
        if not momentum_ok:
            logger.debug(f"[MOMENTUM] Skip {symbol}: {momentum_reason}")
            return None
        
        # ===== STAGE 6: Entry validation =====
        entry_ok, entry_reason = self.entry_validator.validate_entry(
            symbol, direction, abs_spread, max_movement=0.5
        )
        if not entry_ok:
            logger.debug(f"[ENTRY] Skip {symbol}: {entry_reason}")
            return None
        
        entry_quality = self.entry_validator.get_entry_quality(symbol, direction, abs_spread)
        if entry_quality < self.min_entry_quality:
            return None
        
        # Save price history for charts
        now = time.time()
        last_save = self._cache_time.get(f"history_{symbol}", 0)
        if now - last_save > 60:
            await save_price_history(
                token=symbol, chain=chain,
                cex_price=mexc_price, dex_price=dex_price,
                spread_percent=spread
            )
            self._cache_time[f"history_{symbol}"] = now
        
        # ===== STAGE 7: Standard validation =====
        is_valid, reason = self.validator.validate_token(
            symbol=symbol, chain=chain,
            dex_price=dex_price, mexc_price=mexc_price,
            spread_percent=abs_spread
        )
        if not is_valid:
            return None
        
        # Liquidity & Volume filters
        liquidity = pair.get("liquidity_usd", 0)
        volume_24h = pair.get("volume_24h", 0)
        
        if liquidity < MIN_LIQUIDITY_USD:
            return None
        if volume_24h < MIN_VOLUME_24H_USD:
            return None
        
        # Volume ratio filter
        if liquidity > 0 and (volume_24h / liquidity) < 0.05:
            return None
        
        # ===== STAGE 8: Calculate NET profit with funding =====
        gross_profit = self.validator.calculate_net_profit(abs_spread)
        
        # Get funding rate adjustment
        funding_cost = self.funding_tracker.get_funding_adjustment(symbol, direction)
        
        # Adjusted net profit
        net_profit = gross_profit - funding_cost
        
        if net_profit < 3.0:  # Minimum 3% after ALL costs
            return None
        
        # FDV check
        fdv = float(pair.get("fdv", 0) or 0)
        if fdv > 0 and fdv < MIN_FDV_USD:
            return None

        # Min Transactions Check
        txns = pair.get("txns", {}).get("h24", {})
        total_txns = txns.get("buys", 0) + txns.get("sells", 0)
        if total_txns < MIN_TXNS_24H:
            return None

        # Name/Symbol Spam Filter
        name_lower = pair.get("baseToken", {}).get("name", "").lower()
        if "test" in name_lower or "harry" in name_lower or "potter" in name_lower:
            return None
        
        # Wrapped Token Filter
        if "wrapped" in name_lower and symbol not in MAJOR_TOKENS:
             if "sol" not in name_lower and "eth" not in name_lower:
                 return None
        
        # Database duplicate check
        if await check_signal_exists(symbol, direction):
            return None
        
        # Order book depth check (with timeout)
        order_book_depth = 0
        try:
            ob = await asyncio.wait_for(
                self.mexc.get_order_book_depth(symbol, 5000),
                timeout=1.0
            )
            if ob:
                order_book_depth = ob.get("depth_usd", 0)
                if order_book_depth < 10_000:
                    return None
        except asyncio.TimeoutError:
            pass
        
        # ===== STAGE 9: Calculate final quality score =====
        token_score = self.token_intelligence.get_score(symbol)
        convergence_score = self.convergence_analyzer.get_priority_score(symbol)
        momentum_bonus = self.momentum_tracker.get_momentum_bonus(symbol, direction)
        
        quality_score = (
            token_score * 0.3 +
            convergence_score * 0.3 +
            entry_quality * 0.2 +
            (net_profit * 0.5) * 0.2  # Profit contributes to score
        ) * momentum_bonus
        
        quality_score = min(10, max(0, quality_score))
        
        # Get estimated convergence time
        conv_stats = self.convergence_analyzer.get_stats(symbol)
        convergence_time_est = conv_stats.avg_convergence_time_sec if conv_stats else 0
        
        # Create signal with intelligence data
        signal = ArbitrageSignal(
            token=symbol,
            direction=direction,
            spread_percent=abs_spread,
            net_profit=net_profit,
            mexc_price=mexc_price,
            dex_price=dex_price,
            dex_url=pair.get("url", ""),
            chain=chain,
            liquidity_usd=liquidity,
            volume_24h=volume_24h,
            order_book_depth=order_book_depth,
            quality_score=round(quality_score, 1),
            funding_cost=funding_cost,
            momentum_strength=momentum.strength if momentum else 0,
            entry_quality=entry_quality,
            convergence_time_est=convergence_time_est
        )
        
        # Save to DB and set cooldown
        await self._save_signal(signal, pair)
        self._set_cooldown(symbol, direction)
        
        logger.info(
            f"🚀 SIGNAL: {direction} ${symbol} | "
            f"Net: +{net_profit:.1f}% | "
            f"Quality: {quality_score:.1f}/10 | "
            f"Entry: {entry_quality:.1f}"
        )
        
        return signal
    
    async def _save_signal(self, signal: ArbitrageSignal, pair: dict):
        """Save signal to database"""
        dep_status = self.mexc.get_cached_deposit_status(signal.token)
        
        await save_signal(
            token=signal.token,
            chain=signal.chain,
            direction=signal.direction,
            spread_percent=signal.spread_percent,
            dex_price=signal.dex_price,
            mexc_price=signal.mexc_price,
            dex_source="DexScreener",
            liquidity_usd=signal.liquidity_usd,
            volume_24h_usd=signal.volume_24h,
            deposit_enabled=dep_status.get("deposit_enabled", False),
            withdraw_enabled=dep_status.get("withdraw_enabled", False)
        )
        
        await save_price_history(
            token=signal.token,
            chain=signal.chain,
            cex_price=signal.mexc_price,
            dex_price=signal.dex_price,
            spread_percent=signal.spread_percent
        )


def format_turbo_signal(signal: ArbitrageSignal, token_stats: dict = None) -> str:
    """Format signal - BALANCED MODE (Best of both worlds)"""
    chain = get_chain_display_name(signal.chain)
    
    if signal.direction == "LONG":
        header = f"🟢 <b>LONG #{signal.token}</b> | +{signal.net_profit:.1f}%"
        desc = "MEXC Futures Opportunity"
    else:
        header = f"🔴 <b>SHORT #{signal.token}</b> | +{signal.net_profit:.1f}%"
        desc = "MEXC Futures Opportunity"
    
    # Quality text with indicators
    if signal.quality_score >= 8:
        quality_str = "🔥 Excellent"
    elif signal.quality_score >= 6:
        quality_str = "✅ Good"
    elif signal.quality_score >= 4:
        quality_str = "⚠️ Moderate"
    else:
        quality_str = "❓ Low"
    
    # Time estimation
    if signal.convergence_time_est > 0:
        if signal.convergence_time_est >= 3600:
            time_str = f"{signal.convergence_time_est / 3600:.1f}h"
        elif signal.convergence_time_est >= 60:
            time_str = f"{signal.convergence_time_est / 60:.0f}m"
        else:
            time_str = f"{signal.convergence_time_est:.0f}s"
        time_info = f" • ⏱️ ~{time_str}"
    else:
        time_info = ""
    
    # Funding info compact
    funding_info = ""
    if abs(signal.funding_cost) > 0.05:
        if signal.funding_cost > 0:
            funding_info = f" • 💸 -{signal.funding_cost:.2f}%"
        else:
            funding_info = f" • 💰 +{abs(signal.funding_cost):.2f}%"
    
    # Compact liquidity
    liq_str = f"${signal.liquidity_usd/1000:.0f}k" if signal.liquidity_usd > 1000 else f"${signal.liquidity_usd:.0f}"
    
    return (
        f"{header}\n"
        f"{desc}\n\n"
        f"<b>Gap: {signal.spread_percent:.1f}%</b>\n"
        f"MEXC: ${signal.mexc_price}\n"
        f"DEX: ${signal.dex_price}\n\n"
        f"Score: <b>{signal.quality_score}/10</b> ({quality_str})\n"
        f"💧 Liq: {liq_str}{time_info}{funding_info}\n"
        f"{chain} • <a href='{signal.dex_url}'>DexScreener</a> • <a href='https://futures.mexc.com/exchange/{signal.token}_USDT'>MEXC</a>"
    )


# Alias for backward compatibility
ArbitrageScanner = TurboScanner
format_signal_message = format_turbo_signal
