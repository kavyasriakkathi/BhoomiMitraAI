import os
import google.generativeai as genai
from dotenv import load_dotenv

def main():
    load_dotenv(".env")
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        print("No GOOGLE_GEMINI_API_KEY found.")
        return
    genai.configure(api_key=api_key)
    
    print("Available Gemini Models:")
    try:
        models = genai.list_models()
        for m in models:
            if "generateContent" in m.supported_generation_methods:
                print(f"Name: {m.name}")
                print(f"Supported methods: {m.supported_generation_methods}")
                print("---")
    except Exception as e:
        print(f"Error fetching models: {e}")

if __name__ == "__main__":
    main()
