import os
from dotenv import load_dotenv
from google import genai

# Load .env
load_dotenv()

# Read settings
api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("GEMINI_MODEL")

print("====================================")
print("       CampusGPT Gemini Test")
print("====================================")

print("API key found:", bool(api_key))
print("Model:", model_name)

if not api_key:
    print("\nERROR: GOOGLE_API_KEY not found in .env")
    exit()

try:
    # Create Gemini client
    client = genai.Client(api_key=api_key)

    # Send test request
    response = client.models.generate_content(
        model=model_name,
        contents="Say exactly: CampusGPT Gemini API is working."
    )

    print("\nGemini response:")
    print(response.text)

except Exception as e:
    print("\nGemini API Error:")
    print(type(e).__name__)
    print(e)

print("\n====================================")