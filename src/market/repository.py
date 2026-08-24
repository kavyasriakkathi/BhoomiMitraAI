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


DEFAULT_MARKET_PRICES = [
    # Cotton (Telangana Mandis)
    {
        "commodity": "Cotton",
        "commodity_telugu": "పత్తి",
        "market_name": "Warangal Mandi",
        "district": "Warangal",
        "state": "Telangana",
        "min_price": 7100.0,
        "max_price": 7650.0,
        "modal_price": 7450.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    {
        "commodity": "Cotton",
        "commodity_telugu": "పత్తి",
        "market_name": "Adilabad Mandi",
        "district": "Adilabad",
        "state": "Telangana",
        "min_price": 7000.0,
        "max_price": 7550.0,
        "modal_price": 7350.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    {
        "commodity": "Cotton",
        "commodity_telugu": "పత్తి",
        "market_name": "Khammam Mandi",
        "district": "Khammam",
        "state": "Telangana",
        "min_price": 7050.0,
        "max_price": 7600.0,
        "modal_price": 7400.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    # Paddy (Telangana Mandis)
    {
        "commodity": "Paddy",
        "commodity_telugu": "వరి",
        "market_name": "Suryapet Mandi",
        "district": "Suryapet",
        "state": "Telangana",
        "min_price": 2203.0,
        "max_price": 2380.0,
        "modal_price": 2320.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    {
        "commodity": "Paddy",
        "commodity_telugu": "వరి",
        "market_name": "Miryalaguda Mandi",
        "district": "Nalgonda",
        "state": "Telangana",
        "min_price": 2220.0,
        "max_price": 2400.0,
        "modal_price": 2350.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    # Chilli (Telangana Mandis)
    {
        "commodity": "Chilli",
        "commodity_telugu": "మిర్చి",
        "market_name": "Khammam Mandi",
        "district": "Khammam",
        "state": "Telangana",
        "min_price": 16000.0,
        "max_price": 21000.0,
        "modal_price": 18500.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    {
        "commodity": "Chilli",
        "commodity_telugu": "మిర్చి",
        "market_name": "Warangal Mandi",
        "district": "Warangal",
        "state": "Telangana",
        "min_price": 15500.0,
        "max_price": 20500.0,
        "modal_price": 17800.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    # Maize (Telangana Mandis)
    {
        "commodity": "Maize",
        "commodity_telugu": "మొక్కజొన్న",
        "market_name": "Nizamabad Mandi",
        "district": "Nizamabad",
        "state": "Telangana",
        "min_price": 2050.0,
        "max_price": 2350.0,
        "modal_price": 2250.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    {
        "commodity": "Maize",
        "commodity_telugu": "మొక్కజొన్న",
        "market_name": "Badepally Mandi",
        "district": "Mahbubnagar",
        "state": "Telangana",
        "min_price": 2000.0,
        "max_price": 2300.0,
        "modal_price": 2200.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    # Tomato (Telangana / AP Mandis)
    {
        "commodity": "Tomato",
        "commodity_telugu": "టమాటా",
        "market_name": "Bowenpally Mandi",
        "district": "Hyderabad",
        "state": "Telangana",
        "min_price": 1200.0,
        "max_price": 2400.0,
        "modal_price": 1800.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    {
        "commodity": "Tomato",
        "commodity_telugu": "టమాటా",
        "market_name": "Madanapalle Mandi",
        "district": "Chittoor",
        "state": "Andhra Pradesh",
        "min_price": 1300.0,
        "max_price": 2500.0,
        "modal_price": 1950.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    # Red Gram / Pigeonpea
    {
        "commodity": "Red Gram",
        "commodity_telugu": "కందులు",
        "market_name": "Tandur Mandi",
        "district": "Vikarabad",
        "state": "Telangana",
        "min_price": 9500.0,
        "max_price": 10500.0,
        "modal_price": 10100.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    # Turmeric
    {
        "commodity": "Turmeric",
        "commodity_telugu": "పసుపు",
        "market_name": "Nizamabad Mandi",
        "district": "Nizamabad",
        "state": "Telangana",
        "min_price": 12000.0,
        "max_price": 15000.0,
        "modal_price": 13500.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    # Groundnut
    {
        "commodity": "Groundnut",
        "commodity_telugu": "వేరుశనగ",
        "market_name": "Gadwal Mandi",
        "district": "Jogulamba Gadwal",
        "state": "Telangana",
        "min_price": 6200.0,
        "max_price": 7100.0,
        "modal_price": 6700.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    # Soybean
    {
        "commodity": "Soybean",
        "commodity_telugu": "సోయాబీన్",
        "market_name": "Adilabad Mandi",
        "district": "Adilabad",
        "state": "Telangana",
        "min_price": 4200.0,
        "max_price": 4800.0,
        "modal_price": 4550.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
    # Onion
    {
        "commodity": "Onion",
        "commodity_telugu": "ఉల్లిపాయ",
        "market_name": "Malakpet Mandi",
        "district": "Hyderabad",
        "state": "Telangana",
        "min_price": 1800.0,
        "max_price": 2800.0,
        "modal_price": 2300.0,
        "unit": "Quintal",
        "source": "manual_seed",
    },
]


class MarketPriceRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def seed_default_prices_if_empty(self) -> List[MarketPrice]:
        """Idempotently seed default market prices if table is empty or has no recent records."""
        try:
            count_res = await self.db.execute(select(func.count(MarketPrice.id)))
            count = count_res.scalar() or 0
            if count == 0:
                logger.info("[MARKET REPO] market_prices table empty — seeding default market price records.")
                records = []
                now = datetime.utcnow()
                for item in DEFAULT_MARKET_PRICES:
                    record = MarketPrice(
                        commodity=item["commodity"],
                        commodity_telugu=item["commodity_telugu"],
                        market_name=item["market_name"],
                        district=item["district"],
                        state=item["state"],
                        min_price=item["min_price"],
                        max_price=item["max_price"],
                        modal_price=item["modal_price"],
                        unit=item["unit"],
                        price_date=now,
                        source=item["source"],
                    )
                    self.db.add(record)
                    records.append(record)
                await self.db.commit()
                logger.info(f"[MARKET REPO] Seeded {len(records)} default market prices.")
                return records
        except Exception as e:
            logger.warning(f"[MARKET REPO] Failed to seed default prices: {e}")
        return []

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
          4. If nothing within limit_days, fall back to latest available records without date cutoff
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

        # Fall back to any record for this commodity within the date range
        any_results = await self._query_prices(base_filters)
        if any_results:
            logger.info(
                f"[MARKET REPO] Found {len(any_results)} national records "
                f"for '{commodity}' (within {limit_days} days cutoff)"
            )
            return any_results

        # Final resilient fallback: latest records for commodity without date cutoff
        fallback_filters = [MarketPrice.commodity.ilike(f"%{commodity}%")]
        if district:
            d_res = await self._query_prices(fallback_filters + [MarketPrice.district.ilike(f"%{district}%")])
            if d_res:
                return d_res
        if state:
            s_res = await self._query_prices(fallback_filters + [MarketPrice.state.ilike(f"%{state}%")])
            if s_res:
                return s_res
        return await self._query_prices(fallback_filters)

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
