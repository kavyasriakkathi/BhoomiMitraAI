"""
BhoomiMitra AI — Market Price Repository

Database access layer for the market_prices table.
Follows the same pattern as src/shops/repository.py.
"""
from typing import Optional, List
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from src.core.models import MarketPrice
from src.market.schemas import MarketPriceCreate
from src.core.logging import logger


class MarketPriceRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert_prices(self, prices: List[MarketPriceCreate]) -> int:
        """
        Insert new price records.
        Skips records where (commodity, market_name, price_date) already exists
        to avoid duplicates from repeated API calls.
        Returns the count of newly inserted records.
        """
        inserted = 0
        for price_data in prices:
            # Check for existing record with same commodity + market + date
            exists_result = await self.db.execute(
                select(MarketPrice.id).where(
                    and_(
                        MarketPrice.commodity.ilike(price_data.commodity),
                        MarketPrice.market_name.ilike(price_data.market_name),
                        func.date(MarketPrice.price_date) == price_data.price_date.date(),
                    )
                )
            )
            if exists_result.scalar_one_or_none():
                continue  # Already stored — skip

            record = MarketPrice(
                commodity=price_data.commodity,
                commodity_telugu=price_data.commodity_telugu,
                market_name=price_data.market_name,
                district=price_data.district,
                state=price_data.state,
                min_price=price_data.min_price,
                max_price=price_data.max_price,
                modal_price=price_data.modal_price,
                unit=price_data.unit,
                price_date=price_data.price_date,
                source=price_data.source,
            )
            self.db.add(record)
            inserted += 1

        if inserted:
            await self.db.commit()
            logger.info(f"[MARKET REPO] Inserted {inserted} new price records.")

        return inserted

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_prices_by_commodity(
        self,
        commodity: str,
        district: Optional[str] = None,
        state: Optional[str] = None,
        limit_days: int = 3,
    ) -> List[MarketPrice]:
        """
        Return the most recent market price records for a commodity.

        Priority:
          1. Filter by district if provided
          2. Fall back to state-level data if no district records found
          3. Return any national data if still nothing found
        """
        cutoff = datetime.utcnow() - timedelta(days=limit_days)

        base_filters = [
            MarketPrice.commodity.ilike(f"%{commodity}%"),
            MarketPrice.price_date >= cutoff,
        ]

        # Try district-level first
        if district:
            district_results = await self._query_prices(
                base_filters + [MarketPrice.district.ilike(f"%{district}%")]
            )
            if district_results:
                logger.info(
                    f"[MARKET REPO] Found {len(district_results)} district-level records "
                    f"for '{commodity}' in '{district}'"
                )
                return district_results

        # Fall back to state-level
        if state:
            state_results = await self._query_prices(
                base_filters + [MarketPrice.state.ilike(f"%{state}%")]
            )
            if state_results:
                logger.info(
                    f"[MARKET REPO] Found {len(state_results)} state-level records "
                    f"for '{commodity}' in '{state}' (no district match)"
                )
                return state_results

        # Final fallback: any record for this commodity within the date range
        any_results = await self._query_prices(base_filters)
        logger.info(
            f"[MARKET REPO] Found {len(any_results)} national records "
            f"for '{commodity}' (no district/state match)"
        )
        return any_results

    async def _query_prices(self, filters: list) -> List[MarketPrice]:
        """Execute a price query with the given filters, sorted newest-first."""
        result = await self.db.execute(
            select(MarketPrice)
            .where(and_(*filters))
            .order_by(MarketPrice.price_date.desc())
            .limit(10)
        )
        return list(result.scalars().all())

    async def get_latest_price_date(self, commodity: str) -> Optional[datetime]:
        """Returns the most recent price_date stored for a commodity."""
        result = await self.db.execute(
            select(func.max(MarketPrice.price_date)).where(
                MarketPrice.commodity.ilike(f"%{commodity}%")
            )
        )
        return result.scalar_one_or_none()

    async def list_commodities(self) -> List[str]:
        """Return all distinct commodity names stored in the DB."""
        result = await self.db.execute(
            select(MarketPrice.commodity).distinct().order_by(MarketPrice.commodity)
        )
        return [row[0] for row in result.all()]
