import os

from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Mosaic")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = "You are Mosaic, a helpful assistant for a college student. Keep answers short."


class ChatRequest(BaseModel):
    """Request body for POST /chat, e.g. {"question": "When is my Razorpay interview?"}.

    FastAPI validates incoming JSON against this model and rejects a body without a
    `question` string with a 422 error, before any LLM call is made.
    """

    question: str


class ChatResponse(BaseModel):
    """Response body for POST /chat, e.g. {"answer": "Your interview is on 28 Oct..."}."""

    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send the user's question to Gemini with Mosaic's system prompt and return its reply.

    This is a single LLM call with no tools, so the answer comes only from the model's
    training knowledge, not from the user's Gmail or Notion data.
    """
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=request.question,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return ChatResponse(answer=response.text)
