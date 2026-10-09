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
    {
        "commodity": "Paddy",
        "commodity_telugu": "వరి",
        "market_name": "Jagtial Mandi",
        "district": "Jagtial",
        "state": "Telangana",
        "min_price": 2250.0,
        "max_price": 2420.0,
        "modal_price": 2360.0,
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
        strict_location: bool = False,
        exact_location: Optional[str] = None,
    ) -> List[MarketPrice]:
        """
        Return the most relevant market price records for a commodity using strict geographic hierarchy.

        Priority Hierarchy:
          1. Exact town/mandi records (if exact_location provided, e.g. 'Korutla Mandi').
          2. Exact district records with current date cutoff (price_date >= cutoff).
          3. Exact district records from latest available local DB date (no date cutoff).
          4. Same-state records with current date cutoff (price_date >= cutoff).
          5. Same-state records from latest available date (no date cutoff).
          6. National records with current date cutoff (price_date >= cutoff).
          7. Any national records from latest available date (no date cutoff).

        If strict_location is True, only returns records matching the requested district/location.
        Never falls back to state or national records.
        """
        cutoff = datetime.utcnow() - timedelta(days=limit_days)
        base_commodity_filter = [MarketPrice.commodity.ilike(f"%{commodity}%")]

        if district:
            from src.market.service import normalize_district_name, infer_state_from_district
            norm_dist = normalize_district_name(district)
            if norm_dist and district.lower() != norm_dist.lower() and not exact_location:
                exact_location = district
            district = norm_dist
            state = infer_state_from_district(district, state)

        # 1. Exact town / mandi records (e.g. Korutla Mandi)
        if exact_location and (not district or exact_location.lower() != district.lower()):
            exact_recent = await self._query_prices(
                base_commodity_filter + [
                    MarketPrice.price_date >= cutoff,
                    or_(
                        MarketPrice.district.ilike(f"%{exact_location}%"),
                        MarketPrice.market_name.ilike(f"%{exact_location}%"),
                    ),
                ]
            )
            if exact_recent:
                logger.info(
                    f"[MARKET REPO] Priority 1: Found {len(exact_recent)} recent exact-location records "
                    f"for '{commodity}' in '{exact_location}'"
                )
                return exact_recent

            exact_all = await self._query_prices(
                base_commodity_filter + [
                    or_(
                        MarketPrice.district.ilike(f"%{exact_location}%"),
                        MarketPrice.market_name.ilike(f"%{exact_location}%"),
                    ),
                ]
            )
            if exact_all:
                logger.info(
                    f"[MARKET REPO] Priority 2: Found {len(exact_all)} exact-location records (all-time) "
                    f"for '{commodity}' in '{exact_location}'"
                )
                return exact_all

        # 2. District records with current date cutoff
        if district:
            district_recent = await self._query_prices(
                base_commodity_filter + [
                    MarketPrice.price_date >= cutoff,
                    or_(
                        MarketPrice.district.ilike(f"%{district}%"),
                        MarketPrice.market_name.ilike(f"%{district}%"),
                    ),
                ]
            )
            if district_recent:
                logger.info(
                    f"[MARKET REPO] Priority 3: Found {len(district_recent)} recent district-level records "
                    f"for '{commodity}' in '{district}'"
                )
                return district_recent

            # Exact district records from latest available local DB date (all-time)
            district_all = await self._query_prices(
                base_commodity_filter + [
                    or_(
                        MarketPrice.district.ilike(f"%{district}%"),
                        MarketPrice.market_name.ilike(f"%{district}%"),
                    ),
                ]
            )
            if district_all:
                logger.info(
                    f"[MARKET REPO] Priority 4: Found {len(district_all)} district-level records (all-time) "
                    f"for '{commodity}' in '{district}'"
                )
                return district_all

            # When a specific/explicit location was requested by the farmer,
            # never substitute unrelated markets from other parts of the state or country.
            if strict_location:
                logger.info(
                    f"[MARKET REPO] Strict location requested for '{exact_location or district}' — "
                    "no matching records found, suppressing state/national fallback."
                )
                return []

        # 3. Same-state records with current date cutoff
        if state:
            state_recent = await self._query_prices(
                base_commodity_filter + [
                    MarketPrice.price_date >= cutoff,
                    MarketPrice.state.ilike(f"%{state}%"),
                ]
            )
            if state_recent:
                logger.info(
                    f"[MARKET REPO] Priority 3: Found {len(state_recent)} recent state-level records "
                    f"for '{commodity}' in '{state}'"
                )
                return state_recent

            # 4. Same-state records from latest available date (all-time)
            state_all = await self._query_prices(
                base_commodity_filter + [
                    MarketPrice.state.ilike(f"%{state}%"),
                ]
            )
            if state_all:
                logger.info(
                    f"[MARKET REPO] Priority 4: Found {len(state_all)} state-level records (all-time) "
                    f"for '{commodity}' in '{state}'"
                )
                return state_all

        # Strict geographic fencing:
        # If the user or profile has a known district or state (e.g. Telangana / Warangal / Jagtial):
        # NEVER silently fall back to cross-state or national data (e.g. Cumbum, AP or Kalediya, Gujarat).
        if district or state:
            logger.info(
                f"[MARKET REPO] Strict geographic fencing active for commodity='{commodity}', "
                f"district='{district}', state='{state}'. No local or state-level records found. "
                "Suppressed national cross-state fallback to prevent wrong location answers."
            )
            return []

        # 5. National records with current date cutoff (only when completely ungrounded)
        national_recent = await self._query_prices(
            base_commodity_filter + [MarketPrice.price_date >= cutoff]
        )
        if national_recent:
            logger.info(
                f"[MARKET REPO] Priority 5: Found {len(national_recent)} recent national records "
                f"for '{commodity}' (within {limit_days} days cutoff)"
            )
            return national_recent

        # 6. Final fallback: Any national records from latest available date (all-time)
        national_all = await self._query_prices(base_commodity_filter)
        if national_all:
            logger.info(
                f"[MARKET REPO] Priority 6: Found {len(national_all)} all-time national records "
                f"for '{commodity}'"
            )
        return national_all

    async def get_prices_by_location(
        self,
        location: str,
        state: Optional[str] = None,
        limit_days: int = 3,
        strict_location: bool = True,
    ) -> List[MarketPrice]:
        """
        Return available market price records for a location across all commodities.
        Priority Hierarchy:
          1. Exact district/market records with current date cutoff (price_date >= cutoff).
          2. Exact district/market records from latest available local DB date (all-time).
        Deduplicates by commodity, returning the most recent record for each distinct commodity.
        """
        if not location or not isinstance(location, str):
            return []

        cutoff = datetime.utcnow() - timedelta(days=limit_days)
        from src.market.service import normalize_district_name, infer_state_from_district
        norm_location = normalize_district_name(location)
        search_loc = norm_location if norm_location else location
        state = infer_state_from_district(search_loc, state)

        # If location is a town that normalizes to a district (e.g. Korutla -> Jagtial),
        # prioritize exact town first, then fall back to district
        if location and norm_location and location.lower() != norm_location.lower():
            exact_filter = or_(
                MarketPrice.district.ilike(f"%{location}%"),
                MarketPrice.market_name.ilike(f"%{location}%"),
            )
            records = await self._query_prices_all([MarketPrice.price_date >= cutoff, exact_filter])
            if not records:
                records = await self._query_prices_all([exact_filter])
            if not records:
                dist_filter = or_(
                    MarketPrice.district.ilike(f"%{norm_location}%"),
                    MarketPrice.market_name.ilike(f"%{norm_location}%"),
                )
                records = await self._query_prices_all([MarketPrice.price_date >= cutoff, dist_filter])
                if not records:
                    records = await self._query_prices_all([dist_filter])
        else:
            location_filter = or_(
                MarketPrice.district.ilike(f"%{search_loc}%"),
                MarketPrice.market_name.ilike(f"%{search_loc}%"),
            )

            # 1. Recent records with cutoff
            records = await self._query_prices_all([MarketPrice.price_date >= cutoff, location_filter])
            if not records:
                # 2. Latest records all-time
                records = await self._query_prices_all([location_filter])

        if not records and not strict_location and state:
            state_recent = await self._query_prices_all([MarketPrice.price_date >= cutoff, MarketPrice.state.ilike(f"%{state}%")])
            records = state_recent if state_recent else await self._query_prices_all([MarketPrice.state.ilike(f"%{state}%")])

        # Deduplicate to keep the latest record per commodity
        seen_commodities = set()
        deduped = []
        for r in records:
            comm_key = r.commodity.strip().lower()
            if comm_key not in seen_commodities:
                seen_commodities.add(comm_key)
                deduped.append(r)
        return deduped

    async def _query_prices_all(self, filters: list, limit: int = 30) -> List[MarketPrice]:
        """Execute a price query with the given filters, sorted newest-first."""
        result = await self.db.execute(
            select(MarketPrice)
            .where(and_(*filters))
            .order_by(MarketPrice.price_date.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

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
