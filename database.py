"""
MMRdex Database Module
SQLite database for storing signals history and statistics
"""
import aiosqlite
from datetime import datetime
from typing import Optional, Literal
from config import DATABASE_PATH, WIN_THRESHOLD, LOSE_THRESHOLD


# Global database connection for reuse
_db_connection = None


async def get_db():
    """Get reusable database connection for intelligence modules"""
    global _db_connection
    if _db_connection is None:
        _db_connection = await aiosqlite.connect(DATABASE_PATH)
        _db_connection.row_factory = aiosqlite.Row
    return _db_connection


async def init_db():
    """Initialize database tables"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Signals table - stores all arbitrage signals
        await db.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT NOT NULL,
                chain TEXT NOT NULL,
                direction TEXT NOT NULL,
                spread_percent REAL NOT NULL,
                dex_price REAL NOT NULL,
                mexc_price REAL NOT NULL,
                dex_source TEXT,
                liquidity_usd REAL,
                volume_24h_usd REAL,
                deposit_enabled INTEGER,
                withdraw_enabled INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        """)
        
        # Signal outcomes table - stores results after spread closure
        await db.execute("""
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                initial_spread REAL NOT NULL,
                final_spread REAL NOT NULL,
                price_change_percent REAL NOT NULL,
                closed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES signals(id)
            )
        """)
        
        # Price history table - stores CEX/DEX prices for chart generation
        await db.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT NOT NULL,
                chain TEXT NOT NULL,
                cex_price REAL NOT NULL,
                dex_price REAL NOT NULL,
                spread_percent REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for fast queries
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_history_token 
            ON price_history(token, timestamp DESC)
        """)
        
        # Index for active signals lookup
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_active
            ON signals(is_active, token, direction)
        """)
        
        # Index for signal outcomes by token
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_outcomes_signal
            ON signal_outcomes(signal_id)
        """)
        
        # Index for signals by token and created_at
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_token_time
            ON signals(token, created_at DESC)
        """)
        
        await db.commit()


async def clear_all_signals():
    """Clear all signals from database (fresh start)"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("DELETE FROM signals")
        await db.execute("DELETE FROM signal_outcomes")
        await db.commit()


async def save_signal(
    token: str,
    chain: str,
    direction: str,
    spread_percent: float,
    dex_price: float,
    mexc_price: float,
    dex_source: str,
    liquidity_usd: float,
    volume_24h_usd: float,
    deposit_enabled: bool,
    withdraw_enabled: bool
) -> int:
    """Save a new signal to database, returns signal ID"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute("""
            INSERT INTO signals (
                token, chain, direction, spread_percent, dex_price, mexc_price,
                dex_source, liquidity_usd, volume_24h_usd, deposit_enabled, withdraw_enabled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            token, chain, direction, spread_percent, dex_price, mexc_price,
            dex_source, liquidity_usd, volume_24h_usd,
            1 if deposit_enabled else 0, 1 if withdraw_enabled else 0
        ))
        await db.commit()
        return cursor.lastrowid


async def get_active_signals() -> list[dict]:
    """Get all active (unclosed) signals"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT * FROM signals WHERE is_active = 1
        """)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def close_signal(
    signal_id: int,
    final_spread: float,
    price_change_percent: float
):
    """Close a signal and record the outcome"""
    # Determine outcome based on price change
    if price_change_percent > WIN_THRESHOLD:
        outcome = "win"
    elif price_change_percent < LOSE_THRESHOLD:
        outcome = "lose"
    else:
        outcome = "draw"
    
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Get initial spread
        cursor = await db.execute(
            "SELECT spread_percent FROM signals WHERE id = ?", (signal_id,)
        )
        row = await cursor.fetchone()
        initial_spread = row[0] if row else 0
        
        # Update signal as closed
        await db.execute("""
            UPDATE signals SET is_active = 0, closed_at = ? WHERE id = ?
        """, (datetime.now().isoformat(), signal_id))
        
        # Record outcome
        await db.execute("""
            INSERT INTO signal_outcomes (
                signal_id, outcome, initial_spread, final_spread, price_change_percent
            ) VALUES (?, ?, ?, ?, ?)
        """, (signal_id, outcome, initial_spread, final_spread, price_change_percent))
        
        await db.commit()
    
    return outcome


async def get_statistics() -> dict:
    """Get overall statistics for signals"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Average spread
        cursor = await db.execute(
            "SELECT AVG(spread_percent) FROM signals"
        )
        avg_spread = (await cursor.fetchone())[0] or 0
        
        # Average price change for closed signals
        cursor = await db.execute(
            "SELECT AVG(price_change_percent) FROM signal_outcomes"
        )
        avg_change = (await cursor.fetchone())[0] or 0
        
        # Win/Draw/Lose counts
        cursor = await db.execute("""
            SELECT 
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'draw' THEN 1 ELSE 0 END) as draws,
                SUM(CASE WHEN outcome = 'lose' THEN 1 ELSE 0 END) as loses
            FROM signal_outcomes
        """)
        row = await cursor.fetchone()
        wins = row[0] or 0
        draws = row[1] or 0
        loses = row[2] or 0
        
        # Total signals
        cursor = await db.execute("SELECT COUNT(*) FROM signals")
        total = (await cursor.fetchone())[0] or 0
        
        return {
            "total_signals": total,
            "avg_spread": round(avg_spread, 2),
            "avg_change": round(avg_change, 2),
            "wins": wins,
            "draws": draws,
            "loses": loses
        }


async def check_signal_exists(token: str, direction: str) -> bool:
    """Check if an active signal already exists for this token/direction"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute("""
            SELECT id FROM signals 
            WHERE token = ? AND direction = ? AND is_active = 1
        """, (token, direction))
        row = await cursor.fetchone()
        return row is not None


async def get_token_stats(token: str) -> dict:
    """Get statistics for a specific token"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Win/Draw/Lose counts for this token
        cursor = await db.execute("""
            SELECT 
                SUM(CASE WHEN so.outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN so.outcome = 'draw' THEN 1 ELSE 0 END) as draws,
                SUM(CASE WHEN so.outcome = 'lose' THEN 1 ELSE 0 END) as loses,
                AVG(so.price_change_percent) as avg_pnl,
                AVG(so.initial_spread) as avg_spread,
                MAX(so.initial_spread) as max_spread
            FROM signal_outcomes so
            JOIN signals s ON s.id = so.signal_id
            WHERE s.token = ?
        """, (token,))
        row = await cursor.fetchone()
        
        wins = row[0] or 0
        draws = row[1] or 0
        loses = row[2] or 0
        avg_pnl = row[3] or 0
        avg_spread = row[4] or 0
        max_spread = row[5] or 0
        
        total = wins + draws + loses
        winrate = (wins / total * 100) if total > 0 else 0
        
        return {
            "wins": wins,
            "draws": draws,
            "loses": loses,
            "total": total,
            "winrate": round(winrate, 1),
            "avg_pnl": round(avg_pnl, 2),
            "avg_spread": round(avg_spread, 2),
            "max_spread": round(max_spread, 2)
        }


async def save_price_history(
    token: str,
    chain: str,
    cex_price: float,
    dex_price: float,
    spread_percent: float,
    custom_timestamp: Optional[float] = None
):
    """Save a price point to history for chart generation"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        if custom_timestamp:
            # Convert float timestamp to datetime string
            ts_str = datetime.fromtimestamp(custom_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            await db.execute("""
                INSERT INTO price_history (token, chain, cex_price, dex_price, spread_percent, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (token, chain, cex_price, dex_price, spread_percent, ts_str))
        else:
            await db.execute("""
                INSERT INTO price_history (token, chain, cex_price, dex_price, spread_percent)
                VALUES (?, ?, ?, ?, ?)
            """, (token, chain, cex_price, dex_price, spread_percent))
        await db.commit()


async def get_price_history(token: str, hours: float = 1.5, interval_minutes: int = 5) -> list[dict]:
    """
    Get price history for a token aggregated by time intervals.
    
    Args:
        token: Token symbol
        hours: How many hours of history to fetch (default 1.5)
        interval_minutes: Aggregate data by this interval in minutes (default 5)
    
    Returns list of aggregated price points for charts
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        # Get data grouped by 15-minute intervals over last 1.5 hours
        cursor = await db.execute("""
            SELECT 
                AVG(cex_price) as cex_price,
                AVG(dex_price) as dex_price,
                AVG(spread_percent) as spread_percent,
                strftime('%Y-%m-%d %H:', timestamp) || 
                    printf('%02d', (CAST(strftime('%M', timestamp) AS INTEGER) / ?) * ?) || ':00' as timestamp
            FROM price_history
            WHERE token = ? 
              AND timestamp > datetime('now', ?)
            GROUP BY strftime('%Y-%m-%d %H:', timestamp) || 
                     printf('%02d', (CAST(strftime('%M', timestamp) AS INTEGER) / ?) * ?)
            ORDER BY timestamp ASC
        """, (interval_minutes, interval_minutes, token, f"-{hours} hours", interval_minutes, interval_minutes))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def cleanup_old_price_history(hours: int = 24):
    """Remove price history older than X hours to keep DB small"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            DELETE FROM price_history 
            WHERE timestamp < datetime('now', ?)
        """, (f"-{hours} hours",))
        await db.commit()
