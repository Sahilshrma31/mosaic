"""Send one question to Gemini together with the tool schemas.

This script does NOT run any tool. It only shows what the LLM sends back when it has
tools available: instead of answering, it asks us to call a tool with specific arguments.
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import TOOL_DECLARATIONS

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Which tasks have I not started yet?",
    config=types.GenerateContentConfig(
        system_instruction="You are Mosaic, a student's assistant. Use the tools to find answers.",
        tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
    ),
)

if response.function_calls:
    for call in response.function_calls:
        print("LLM wants to call:", call.name, "with args:", call.args)
else:
    print("LLM answered directly:", response.text)
