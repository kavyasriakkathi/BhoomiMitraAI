from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.conversation.schemas import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    PaginatedConversationResponse,
)
from src.conversation.service import ConversationService
from src.conversation.dependencies import get_conversation_service
from src.auth.dependencies import require_admin
from src.core.models import UserAccount

router = APIRouter()


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
    description="Record a new farmer-AI interaction. The message_id must be unique (idempotency key from Meta).",
)
async def create_conversation(
    data: ConversationCreate,
    service: ConversationService = Depends(get_conversation_service),
    db: AsyncSession = Depends(get_db),
):
    conversation = await service.create_conversation(data)
    await db.commit()
    return conversation


@router.get(
    "",
    response_model=PaginatedConversationResponse,
    status_code=status.HTTP_200_OK,
    summary="List conversations",
    description="Retrieve a paginated list of all conversations, ordered by newest first.",
)
async def get_conversations(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    service: ConversationService = Depends(get_conversation_service),
):
    total, items = await service.get_conversations(page=page, size=size)
    return PaginatedConversationResponse(
        total=total,
        items=items,
        page=page,
        size=size,
    )


@router.get(
    "/farmer/{farmer_id}",
    response_model=PaginatedConversationResponse,
    status_code=status.HTTP_200_OK,
    summary="List conversations for a farmer",
    description="Retrieve a paginated conversation history for a specific farmer, ordered by newest first.",
)
async def get_farmer_conversations(
    farmer_id: UUID,
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    service: ConversationService = Depends(get_conversation_service),
):
    total, items = await service.get_farmer_conversations(farmer_id, page=page, size=size)
    return PaginatedConversationResponse(
        total=total,
        items=items,
        page=page,
        size=size,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a conversation",
    description="Retrieve a single conversation by its UUID.",
)
async def get_conversation(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service),
):
    return await service.get_conversation(conversation_id)


@router.put(
    "/{conversation_id}",
    response_model=ConversationResponse,
    status_code=status.HTTP_200_OK,
    summary="Update a conversation",
    description="Update conversation fields (AI response, intent, delivery status). Only provided fields will be updated.",
)
async def update_conversation(
    conversation_id: UUID,
    data: ConversationUpdate,
    service: ConversationService = Depends(get_conversation_service),
    db: AsyncSession = Depends(get_db),
):
    conversation = await service.update_conversation(conversation_id, data)
    await db.commit()
    return conversation


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation (Admin only)",
    description="Hard delete a conversation record by its UUID.",
)
async def delete_conversation(
    conversation_id: UUID,
    current_user: UserAccount = Depends(require_admin),
    service: ConversationService = Depends(get_conversation_service),
    db: AsyncSession = Depends(get_db),
):
    await service.delete_conversation(conversation_id)
    await db.commit()
