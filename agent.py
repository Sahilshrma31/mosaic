"""Hand-written agent loop: the LLM asks for tool calls, we run them, and repeat until it answers."""

import os
from datetime import date

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from tools import TOOL_DECLARATIONS, TOOL_FUNCTIONS

load_dotenv()

# The SDK silently retries failed requests with growing waits, which can take minutes on an
# overloaded model. Fail fast instead (60s timeout, 2 attempts) so the fallback model can step in.
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=60_000, retry_options=types.HttpRetryOptions(attempts=2)),
)

MODEL = "gemini-3.5-flash"
# Used when MODEL is overloaded (503) or times out (504).
FALLBACK_MODEL = "gemini-3.6-flash"

# Upper bound on LLM round trips per question. Without it, a model that keeps asking for
# tools would loop forever (and burn through the free-tier rate limit).
MAX_STEPS = 6

SYSTEM_PROMPT = f"""You are Mosaic, an assistant for a college student.
Answer questions using the student's Gmail and Notion data, which you can read with the tools.
Today's date is {date.today().isoformat()}.
Rules:
- Always look the facts up with the tools before answering. Never invent dates, names or statuses.
- Batch independent lookups: request all the tool calls you need in one turn.
- If two sources disagree, trust the most recent email and point out the mismatch.
- Ignore promotional mails unless they contain an action the student must take.
- Keep the final answer short and clear."""

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
)

# Used for the last allowed step: tool calls are switched off (mode NONE), so the model must
# answer from the evidence it already gathered instead of the user getting no answer at all.
FINAL_STEP_CONFIG = CONFIG.model_copy(
    update={
        "system_instruction": SYSTEM_PROMPT
        + "\n- You cannot call tools any more. Answer now with the evidence you already have,"
        " and briefly say what you could not check.",
        "tool_config": types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE")),
    }
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
    """Answer one question, switching to FALLBACK_MODEL if MODEL fails for a transient reason.

    Two errors trigger the fallback:
    - 5xx: Google's side is busy (503) or timed out (504), so another model is likely to work.
    - 429: the free-tier quota is used up. Quotas are counted per model (20 requests/day each),
      so the fallback model may still have quota left.
    The question is restarted from scratch on the fallback model rather than continued, so
    one conversation never mixes two models. Other 4xx errors (e.g. a malformed request)
    would fail the same way on any model, so they are raised instead.
    """
    for model in (MODEL, FALLBACK_MODEL):
        try:
            return _run_loop(question, model)
        except errors.ServerError as exc:
            print(f"[{model}] server error {exc.code}, trying the next model...")
        except errors.ClientError as exc:
            if exc.code != 429:
                raise
            print(f"[{model}] free-tier quota used up (429), trying the next model...")
    return "Mosaic's AI models are overloaded or out of free quota right now. Please try again later."


def _run_loop(question: str, model: str) -> str:
    """Answer one question by looping between the LLM and the tools.

    Each round sends the whole conversation so far (the question, every earlier tool call
    and every tool result) because the LLM is stateless and remembers nothing between
    requests. If the reply contains function calls, we run them, append the results and
    loop again. A reply with no function calls is the final answer.

    The loop is capped at MAX_STEPS rounds so a confused model can't loop forever. The last
    round runs with tools disabled, which forces a best-effort answer from the evidence
    gathered so far (graceful degradation) instead of an empty failure.
    """
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]

    for step in range(1, MAX_STEPS + 1):
        config = FINAL_STEP_CONFIG if step == MAX_STEPS else CONFIG
        response = client.models.generate_content(model=model, contents=contents, config=config)

        if not response.function_calls:
            print(f"[step {step}] final answer, no tool calls")
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
