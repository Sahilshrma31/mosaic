"""Hand-written agent loop: the LLM asks for tool calls, we run them, and repeat until it answers."""

import os
from datetime import date

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import TOOL_DECLARATIONS, TOOL_FUNCTIONS

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.5-flash"

# Upper bound on LLM round trips per question. Without it, a model that keeps asking for
# tools would loop forever (and burn through the free-tier rate limit).
MAX_STEPS = 6

SYSTEM_PROMPT = f"""You are Mosaic, an assistant for a college student.
Answer questions using the student's Gmail and Notion data, which you can read with the tools.
Today's date is {date.today().isoformat()}.
Rules:
- Always look the facts up with the tools before answering. Never invent dates, names or statuses.
- If two sources disagree, trust the most recent email and point out the mismatch.
- Ignore promotional mails unless they contain an action the student must take.
- Keep the final answer short and clear."""

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
)


def run_tool(name: str, args: dict) -> dict:
    """Run one tool the LLM asked for and wrap the outcome in a dict for the LLM to read.

    Never raises. An unknown tool name, bad arguments or an exception inside the tool all
    come back as {"error": "..."}, so the LLM sees what went wrong and can retry with a
    different call instead of the whole request crashing.
    """
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool '{name}'. Available tools: {list(TOOL_FUNCTIONS)}"}
    try:
        return {"result": func(**args)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_agent(question: str) -> str:
    """Answer one question by looping between the LLM and the tools.

    Each round sends the whole conversation so far (the question, every earlier tool call
    and every tool result) because the LLM is stateless and remembers nothing between
    requests. If the reply contains function calls, we run them, append the results and
    loop again. A reply with no function calls is the final answer.

    Gives up after MAX_STEPS rounds so a confused model can't loop forever.
    """
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]

    for step in range(1, MAX_STEPS + 1):
        response = client.models.generate_content(model=MODEL, contents=contents, config=CONFIG)

        if not response.function_calls:
            return response.text

        # Keep the model's own turn (its function calls) in the history, unchanged.
        contents.append(response.candidates[0].content)

        # Run every requested tool and send all the results back in one turn.
        result_parts = []
        for call in response.function_calls:
            result = run_tool(call.name, call.args or {})
            print(f"[step {step}] {call.name}({call.args}) -> {_summary(result)}")
            result_parts.append(
                types.Part(function_response=types.FunctionResponse(id=call.id, name=call.name, response=result))
            )
        contents.append(types.Content(role="user", parts=result_parts))

    return "Sorry, I couldn't finish answering within the step limit. Try a more specific question."


def _summary(result: dict) -> str:
    """One-line description of a tool result for the console trace."""
    if "error" in result:
        return f"ERROR {result['error']}"
    return f"{len(result['result'])} results"


if __name__ == "__main__":
    print(run_agent(input("Ask Mosaic: ")))
