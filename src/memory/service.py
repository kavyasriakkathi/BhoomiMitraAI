import json
import re
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.logging import logger
from src.core.models import FarmerProfile, Farm, Crop, CropHealth, OrderRequest, SchemeApplication, Conversation
from src.memory.models import FarmerMemory
from src.memory.repository import FarmerMemoryRepository
from src.memory.schemas import (
    FarmerMemoryResponse,
    FarmerMemoryUpdate,
    FarmerMemorySummaryResponse,
    VoiceSettingsResponse,
)
from src.memory.prompts import (
    MEMORY_EXTRACTION_SYSTEM_PROMPT,
    MEMORY_SUMMARIZATION_SYSTEM_PROMPT,
    build_memory_context_prompt,
)
from src.ai.gemini_client import generate_response


class FarmerMemoryService:
    def __init__(self, repository: FarmerMemoryRepository):
        self.repository = repository

    async def get_memory(self, farmer_id: UUID) -> FarmerMemory:
        """Fetch or initialize FarmerMemory record."""
        return await self.repository.get_or_create(farmer_id)

    async def get_memory_response(self, farmer_id: UUID) -> FarmerMemoryResponse:
        """Fetch FarmerMemory as Pydantic Response Schema."""
        memory = await self.get_memory(farmer_id)
        return FarmerMemoryResponse.model_validate(memory)

    async def update_memory(self, farmer_id: UUID, data: FarmerMemoryUpdate) -> FarmerMemoryResponse:
        """Manual / Expert update of Farmer Memory profile."""
        memory = await self.get_memory(farmer_id)
        update_dict = data.model_dump(exclude_unset=True)

        for key, value in update_dict.items():
            if value is not None:
                setattr(memory, key, value)

        memory.last_updated = datetime.utcnow()
        saved = await self.repository.save(memory)
        return FarmerMemoryResponse.model_validate(saved)

    async def extract_and_update_memory(
        self,
        farmer_id: UUID,
        user_message: str,
        ai_response: str = ""
    ) -> FarmerMemory:
        """
        Automatic memory extraction after a conversation step.
        Uses rule-based heuristic regex & LLM extraction to update long-term memory
        with strict confidence scores.
        """
        memory = await self.get_memory(farmer_id)
        confidence_map = memory.confidence_scores or {}
        if not isinstance(confidence_map, dict):
            confidence_map = {}

        updates_applied = False

        # 1. Rule-based heuristic extraction (High confidence 0.9)
        if user_message:
            msg_lower = user_message.lower()

            # Land size regex (e.g., "5 acres", "10 acre", "2.5 acres")
            size_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:acres?|acre|ekaram|ekara)", msg_lower)
            if size_match:
                try:
                    val = float(size_match.group(1))
                    if confidence_map.get("farm_size", 0.0) <= 0.9:
                        memory.farm_size = val
                        confidence_map["farm_size"] = 0.9
                        updates_applied = True
                except ValueError:
                    pass

            # Village regex (e.g., "my village is Karimnagar", "village: Warangal", "naa ooru Karimnagar")
            village_match = re.search(r"(?:my village is|village is|village:|naa ooru|naji ooru)\s+([a-zA-Z\s]+)", msg_lower)
            if village_match:
                val = village_match.group(1).strip().title()
                if len(val) > 2 and confidence_map.get("village", 0.0) <= 0.9:
                    memory.village = val
                    confidence_map["village"] = 0.9
                    updates_applied = True

            # Primary crops regex (e.g., "I grow cotton", "crop: paddy", "mirchi panta")
            crop_keywords = {
                "cotton": "Cotton",
                "paddy": "Paddy",
                "rice": "Paddy",
                "chilli": "Chilli",
                "chili": "Chilli",
                "mirchi": "Chilli",
                "maize": "Maize",
                "corn": "Maize",
                "groundnut": "Groundnut",
                "turmeric": "Turmeric",
                "sugarcane": "Sugarcane",
                "pulses": "Pulses",
            }
            current_crops = set(memory.primary_crops or [])
            for kw, std_crop in crop_keywords.items():
                if kw in msg_lower and std_crop not in current_crops:
                    current_crops.add(std_crop)
                    updates_applied = True

            if updates_applied:
                memory.primary_crops = list(current_crops)

            # Shop regex (e.g., "buy from Ramesh Fertilizers", "shop Ramesh")
            shop_match = re.search(r"(?:buy from|shop|store)\s+([a-zA-Z0-9\s]+(?:fertilizers|agri|seeds|traders|shop)?)", msg_lower)
            if shop_match:
                shop_name = shop_match.group(1).strip().title()
                current_shops = set(memory.favorite_shops or [])
                if shop_name not in current_shops and len(shop_name) > 3:
                    current_shops.add(shop_name)
                    memory.favorite_shops = list(current_shops)
                    updates_applied = True

        # 2. LLM-assisted Memory Extraction
        try:
            prompt_input = f"Farmer Message: {user_message}\nAI Response: {ai_response}"
            extracted_json = await generate_response(
                system_prompt=MEMORY_EXTRACTION_SYSTEM_PROMPT,
                user_message=prompt_input
            )

            if extracted_json:
                # Clean codeblock wrappers if present
                clean_json = extracted_json.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json[7:]
                if clean_json.startswith("```"):
                    clean_json = clean_json[3:]
                if clean_json.endswith("```"):
                    clean_json = clean_json[:-3]
                clean_json = clean_json.strip()

                parsed = json.loads(clean_json)
                updates = parsed.get("updates", {})
                llm_conf = parsed.get("confidence_scores", {})

                for field, val in updates.items():
                    if val is None or val == "":
                        continue

                    score = llm_conf.get(field, 0.7)

                    # Only merge if confidence is >= 0.7 and >= existing confidence
                    if score >= 0.7 and score >= confidence_map.get(field, 0.0):
                        if field in ["farm_size", "village", "district", "state", "soil_type", "water_source", "irrigation_method", "preferred_language", "preferred_voice", "voice_speed", "voice_gender"]:
                            setattr(memory, field, val)
                            confidence_map[field] = float(score)
                            updates_applied = True
                        elif field == "primary_crops" and isinstance(val, list):
                            existing = set(memory.primary_crops or [])
                            existing.update(val)
                            memory.primary_crops = list(existing)
                            confidence_map[field] = float(score)
                            updates_applied = True
                        elif field == "secondary_crops" and isinstance(val, list):
                            existing = set(memory.secondary_crops or [])
                            existing.update(val)
                            memory.secondary_crops = list(existing)
                            confidence_map[field] = float(score)
                            updates_applied = True
                        elif field == "favorite_shops" and isinstance(val, list):
                            existing = set(memory.favorite_shops or [])
                            existing.update(val)
                            memory.favorite_shops = list(existing)
                            confidence_map[field] = float(score)
                            updates_applied = True
                        elif field == "preferred_brands" and isinstance(val, list):
                            existing = set(memory.preferred_brands or [])
                            existing.update(val)
                            memory.preferred_brands = list(existing)
                            confidence_map[field] = float(score)
                            updates_applied = True
                        elif field == "disease_mentioned" and isinstance(val, str):
                            d_hist = memory.disease_history or []
                            d_hist.append({"disease": val, "timestamp": datetime.utcnow().isoformat()})
                            memory.disease_history = d_hist
                            updates_applied = True
                        elif field == "pesticide_mentioned" and isinstance(val, str):
                            p_hist = memory.pesticide_history or []
                            p_hist.append({"name": val, "timestamp": datetime.utcnow().isoformat()})
                            memory.pesticide_history = p_hist
                            updates_applied = True
                        elif field == "fertilizer_mentioned" and isinstance(val, str):
                            f_hist = memory.fertilizer_history or []
                            f_hist.append({"name": val, "timestamp": datetime.utcnow().isoformat()})
                            memory.fertilizer_history = f_hist
                            updates_applied = True

        except Exception as e:
            logger.warning(f"LLM memory extraction skipped for farmer {farmer_id}: {e}")

        memory.confidence_scores = confidence_map
        memory.last_updated = datetime.utcnow()
        return await self.repository.save(memory)

    async def refresh_farmer_memory(self, farmer_id: UUID) -> FarmerMemoryResponse:
        """
        Synchronizes memory from relational entities (`FarmerProfile`, `Farm`, `CropHealth`, `OrderRequest`, `SchemeApplication`).
        """
        memory = await self.get_memory(farmer_id)
        session = self.repository.session

        # 1. Sync from FarmerProfile
        res_prof = await session.execute(
            select(FarmerProfile).where(FarmerProfile.farmer_id == farmer_id)
        )
        profile = res_prof.scalar_one_or_none()
        if profile:
            if profile.district and not memory.district:
                memory.district = profile.district
            if profile.state and not memory.state:
                memory.state = profile.state
            if profile.land_size_acres and not memory.farm_size:
                memory.farm_size = profile.land_size_acres
            if profile.current_crop:
                crops = set(memory.primary_crops or [])
                crops.add(profile.current_crop)
                memory.primary_crops = list(crops)

        # 2. Sync from Farms & Crops
        res_farms = await session.execute(
            select(Farm).where(Farm.farmer_id == farmer_id)
        )
        farms = res_farms.scalars().all()
        for farm in farms:
            if farm.village and not memory.village:
                memory.village = farm.village
            if farm.district and not memory.district:
                memory.district = farm.district
            if farm.state and not memory.state:
                memory.state = farm.state
            if farm.soil_type and not memory.soil_type:
                memory.soil_type = farm.soil_type
            if farm.irrigation_type and not memory.irrigation_method:
                memory.irrigation_method = farm.irrigation_type
            if farm.land_size_acres and not memory.farm_size:
                memory.farm_size = farm.land_size_acres
            if farm.latitude and farm.longitude:
                memory.gps_coordinates = {"latitude": farm.latitude, "longitude": farm.longitude}

            # Fetch crops for farm
            res_crops = await session.execute(
                select(Crop).where(Crop.farm_id == farm.id)
            )
            farm_crops = res_crops.scalars().all()
            primary = set(memory.primary_crops or [])
            for c in farm_crops:
                if c.crop_name:
                    primary.add(c.crop_name)
            memory.primary_crops = list(primary)

        # 3. Sync from CropHealth Diagnoses
        res_health = await session.execute(
            select(CropHealth).where(CropHealth.farmer_id == farmer_id)
        )
        diagnoses = res_health.scalars().all()
        if diagnoses:
            disease_hist = memory.disease_history or []
            existing_diseases = {d.get("disease") for d in disease_hist if isinstance(d, dict)}
            for diag in diagnoses:
                if diag.disease_name and diag.disease_name not in existing_diseases:
                    disease_hist.append({
                        "disease": diag.disease_name,
                        "symptoms": diag.symptoms,
                        "treatment": diag.treatment_recommendation,
                        "timestamp": diag.created_at.isoformat() if diag.created_at else datetime.utcnow().isoformat()
                    })
                    existing_diseases.add(diag.disease_name)
            memory.disease_history = disease_hist

        # 4. Sync from Order Requests (Purchases & Favorite Shops)
        res_orders = await session.execute(
            select(OrderRequest).where(OrderRequest.farmer_id == farmer_id)
        )
        orders = res_orders.scalars().all()
        if orders:
            purchases = memory.purchase_history or []
            brands = set(memory.preferred_brands or [])
            for ord_req in orders:
                purchases.append({
                    "product": ord_req.product_name,
                    "brand": ord_req.brand,
                    "quantity": ord_req.quantity,
                    "status": ord_req.status,
                    "timestamp": ord_req.created_at.isoformat() if ord_req.created_at else datetime.utcnow().isoformat()
                })
                if ord_req.brand:
                    brands.add(ord_req.brand)
            memory.purchase_history = purchases
            memory.preferred_brands = list(brands)

        # 5. Sync from Scheme Applications
        res_schemes = await session.execute(
            select(SchemeApplication).where(SchemeApplication.farmer_id == farmer_id)
        )
        schemes = res_schemes.scalars().all()
        if schemes:
            schemes_used = set(memory.government_schemes_used or [])
            for app in schemes:
                schemes_used.add(str(app.scheme_id))
            memory.government_schemes_used = list(schemes_used)

        memory.last_updated = datetime.utcnow()
        saved = await self.repository.save(memory)
        return FarmerMemoryResponse.model_validate(saved)

    async def summarize_conversations(self, farmer_id: UUID) -> str:
        """
        Generates and persists an AI summary of past farmer conversations.
        """
        memory = await self.get_memory(farmer_id)
        session = self.repository.session

        result = await session.execute(
            select(Conversation)
            .where(Conversation.farmer_id == farmer_id)
            .order_by(Conversation.created_at.desc())
            .limit(20)
        )
        conversations = result.scalars().all()

        if not conversations:
            return "No previous conversations found."

        history_text = []
        for conv in reversed(conversations):
            if conv.user_message:
                history_text.append(f"Farmer: {conv.user_message}")
            if conv.ai_response:
                history_text.append(f"BhoomiMitra: {conv.ai_response}")

        transcript = "\n".join(history_text)
        summary = await generate_response(
            system_prompt=MEMORY_SUMMARIZATION_SYSTEM_PROMPT,
            user_message=transcript
        )

        if summary:
            memory.conversation_summary = summary.strip()
            memory.last_updated = datetime.utcnow()
            await self.repository.save(memory)
            return memory.conversation_summary

        return memory.conversation_summary or "Summary unavailable."

    async def format_memory_for_system_prompt(self, farmer_id: UUID) -> str:
        """Build formatted memory prompt string for injection into Gemini system prompt."""
        memory = await self.get_memory(farmer_id)
        return build_memory_context_prompt(memory)

    async def get_voice_settings(self, farmer_id: UUID) -> VoiceSettingsResponse:
        """Retrieve voice personalization parameters for STT/TTS."""
        memory = await self.get_memory(farmer_id)
        return VoiceSettingsResponse(
            farmer_id=farmer_id,
            preferred_language=memory.preferred_language or "te",
            preferred_voice=memory.preferred_voice or "Google-te-IN-Standard-A",
            voice_speed=memory.voice_speed or 1.0,
            voice_gender=memory.voice_gender or "FEMALE"
        )
