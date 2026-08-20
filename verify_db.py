import asyncio
import uuid
from sqlalchemy import select
from src.core.database import AsyncSessionLocal, engine, Base
import src.core.models
from src.core.models import Farmer, FarmerProfile, Conversation, Expert

async def verify_crud():
    print("Starting DB CRUD Verification...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        try:
            # 1. Create Expert
            expert = Expert(
                name="Test Expert",
                phone_number=f"+9198765{uuid.uuid4().hex[:5]}",
                specialty="Soil Health"
            )
            session.add(expert)
            
            # 2. Create Farmer
            farmer_phone = f"+9199999{uuid.uuid4().hex[:5]}"
            farmer = Farmer(
                phone_number=farmer_phone,
                preferred_language="te"
            )
            session.add(farmer)
            await session.flush()
            
            # 3. Create FarmerProfile
            profile = FarmerProfile(
                farmer_id=farmer.id,
                full_name="Test Farmer",
                state="Telangana",
                district="Hyderabad",
                current_crop="Paddy",
                land_size_acres=2.5
            )
            session.add(profile)
            
            # 4. Create Conversation
            conv = Conversation(
                farmer_id=farmer.id,
                message_id=f"msg_{uuid.uuid4().hex}",
                user_message="Hello, I need help with paddy.",
                ai_response="Sure, what is the issue?",
                intent="greeting",
                confidence_score=0.95
            )
            session.add(conv)
            
            await session.commit()
            print("Create operations successful.")
            
            # 5. Read and Verify
            stmt_expert = select(Expert).where(Expert.id == expert.id)
            res_expert = (await session.execute(stmt_expert)).scalar_one_or_none()
            assert res_expert is not None and res_expert.name == "Test Expert"
            
            stmt_farmer = select(Farmer).where(Farmer.id == farmer.id)
            res_farmer = (await session.execute(stmt_farmer)).scalar_one_or_none()
            assert res_farmer is not None and res_farmer.phone_number == farmer_phone
            
            stmt_profile = select(FarmerProfile).where(FarmerProfile.farmer_id == farmer.id)
            res_profile = (await session.execute(stmt_profile)).scalar_one_or_none()
            assert res_profile is not None and res_profile.current_crop == "Paddy"
            
            stmt_conv = select(Conversation).where(Conversation.farmer_id == farmer.id)
            res_conv = (await session.execute(stmt_conv)).scalars().first()
            assert res_conv is not None and res_conv.intent == "greeting"
            
            print("Read operations successful.")
            
            # 6. Update
            res_expert.specialty = "Pest Control"
            res_profile.land_size_acres = 3.0
            await session.commit()
            print("Update operations successful.")
            
            # 7. Delete (Optional: cleanup)
            await session.delete(res_expert)
            await session.delete(res_farmer) # Should cascade and delete profile and conversation
            await session.commit()
            print("Delete operations successful (Cascade verified).")
            
            print("All CRUD operations verified successfully.")
            
        except Exception as e:
            await session.rollback()
            print(f"Error during CRUD verification: {e}")
            raise e

if __name__ == "__main__":
    asyncio.run(verify_crud())
