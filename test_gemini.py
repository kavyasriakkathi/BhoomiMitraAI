import asyncio
import os
from dotenv import load_dotenv
from src.ai.gemini_client import generate_response

async def main():
    load_dotenv(".env")
    system_prompt = "You are a helpful farming assistant."
    history = [
        {"role": "user", "parts": "Hello!"},
        {"role": "model", "parts": "Hi there, how can I help?"}
    ]
    user_msg = "What is the best fertilizer for wheat?"
    try:
        response = await generate_response(system_prompt, history, user_msg)
        print("Response:", response)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
