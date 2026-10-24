import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # Load GEMINI_API_KEY from the .env file into the environment variables

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
question = input("Sawaal pucho: ") 

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=question,  
    config=types.GenerateContentConfig(
        system_instruction="You are Mosaic, a helpful assistant for a college student. Keep answers short.",
    ),
)

print(response.text)
print("---")
print("Input tokens: ", response.usage_metadata.prompt_token_count)
print("Output tokens:", response.usage_metadata.candidates_token_count)
print("Total tokens: ", response.usage_metadata.total_token_count)
