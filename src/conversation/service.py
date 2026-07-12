from typing import Tuple, List
from uuid import UUID
from fastapi import HTTPException, status
from src.core.models import Conversation
from src.conversation.repository import ConversationRepository
from src.conversation.schemas import ConversationCreate, ConversationUpdate
from src.farmers.repository import FarmerRepository


class ConversationService:
    def __init__(self, repository: ConversationRepository, farmer_repository: FarmerRepository):
        self.repository = repository
        self.farmer_repository = farmer_repository

    async def create_conversation(self, data: ConversationCreate) -> Conversation:
        # Validate farmer exists
        farmer = await self.farmer_repository.get_by_id(data.farmer_id)
        if not farmer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Farmer not found."
            )

        # Idempotency — reject duplicate message_id
        existing = await self.repository.get_by_message_id(data.message_id)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Conversation with this message_id already exists."
            )

        conversation = Conversation(**data.model_dump())
        return await self.repository.create(conversation)

    async def get_conversation(self, conversation_id: UUID) -> Conversation:
        conversation = await self.repository.get_by_id(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found."
            )
        return conversation

    async def get_conversations(self, page: int = 1, size: int = 10) -> Tuple[int, List[Conversation]]:
        skip = (page - 1) * size
        return await self.repository.get_all(skip=skip, limit=size)

    async def get_farmer_conversations(
        self, farmer_id: UUID, page: int = 1, size: int = 10
    ) -> Tuple[int, List[Conversation]]:
        # Validate farmer exists
        farmer = await self.farmer_repository.get_by_id(farmer_id)
        if not farmer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Farmer not found."
            )

        skip = (page - 1) * size
        return await self.repository.get_by_farmer_id(farmer_id, skip=skip, limit=size)

    async def update_conversation(self, conversation_id: UUID, data: ConversationUpdate) -> Conversation:
        conversation = await self.get_conversation(conversation_id)

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(conversation, key, value)

        await self.repository.session.flush()
        await self.repository.session.refresh(conversation)
        return conversation

    async def delete_conversation(self, conversation_id: UUID) -> None:
        conversation = await self.get_conversation(conversation_id)
        await self.repository.delete(conversation)
