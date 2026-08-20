from typing import Tuple, List
from uuid import UUID
from fastapi import HTTPException, status
from src.core.models import FarmerProfile
from src.farmer_profiles.repository import FarmerProfileRepository
from src.farmer_profiles.schemas import FarmerProfileCreate, FarmerProfileUpdate
from src.farmers.repository import FarmerRepository

class FarmerProfileService:
    def __init__(self, repository: FarmerProfileRepository, farmer_repository: FarmerRepository):
        self.repository = repository
        self.farmer_repository = farmer_repository

    async def create_profile(self, profile_data: FarmerProfileCreate) -> FarmerProfile:
        # Validate farmer exists
        farmer = await self.farmer_repository.get_by_id(profile_data.farmer_id)
        if not farmer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Farmer not found."
            )

        # Validate farmer doesn't already have a profile
        existing_profile = await self.repository.get_by_farmer_id(profile_data.farmer_id)
        if existing_profile:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Farmer already has a profile."
            )
        
        profile = FarmerProfile(**profile_data.model_dump())
        return await self.repository.create(profile)

    async def get_profile(self, profile_id: UUID) -> FarmerProfile:
        profile = await self.repository.get_by_id(profile_id)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Farmer profile not found."
            )
        return profile

    async def get_profile_by_farmer(self, farmer_id: UUID) -> FarmerProfile:
        profile = await self.repository.get_by_farmer_id(farmer_id)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Farmer profile not found for the given farmer ID."
            )
        return profile

    async def get_profiles(self, page: int = 1, size: int = 10) -> Tuple[int, List[FarmerProfile]]:
        skip = (page - 1) * size
        return await self.repository.get_all(skip=skip, limit=size)

    async def update_profile(self, profile_id: UUID, profile_data: FarmerProfileUpdate) -> FarmerProfile:
        profile = await self.get_profile(profile_id)
        
        update_data = profile_data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(profile, key, value)
            
        await self.repository.session.flush()
        await self.repository.session.refresh(profile)
        return profile

    async def delete_profile(self, profile_id: UUID) -> None:
        profile = await self.get_profile(profile_id)
        await self.repository.delete(profile)
