"""Run the eval set (evals/cases.json) against the agent and score it.

Two separate scores per case:
- Answer accuracy: every `must_include` regex appears in the answer and no `must_not_include`
  regex does. Patterns allow wording variants, e.g. "30 Oct", "30th October", "October 30".
- Citation accuracy: every group in `expected_sources` has at least one cited id (the right
  source was cited), and every cited id is in `relevant_sources` (nothing unrelated was cited).
  Cases with no expected sources and `relevant_sources: null` (unanswerable questions) are
  not scored for citations.

Every case runs under the same conditions, so numbers are comparable between runs: mock data
with known answers (whatever MOSAIC_DATA says in .env), a pinned "today", and a fresh, empty
memory database holding only the case's own `facts` and `history`.

The free tier allows ~80 requests a day, so results are appended to evals/results.jsonl as
each case finishes, and a re-run skips cases that already have a result. When every model is
out of quota the run stops; run it again later (e.g. the next day) and it resumes.

Usage (from the project root):
    .venv/bin/python evals/run_eval.py                  # run the cases without a result yet
    .venv/bin/python evals/run_eval.py --only dbms_latest_deadline swiggy_mismatch
    .venv/bin/python evals/run_eval.py --rerun          # run every case again
    .venv/bin/python evals/run_eval.py --model gemini-3.5-flash-lite   # one model only
    .venv/bin/python evals/run_eval.py --report         # just print the scores so far
"""

import argparse
import asyncio
import json
import re
import sys
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `python evals/run_eval.py` can import the app modules

import agent  # noqa: E402
import memory  # noqa: E402

CASES_FILE = Path(__file__).parent / "cases.json"
RESULTS_FILE = Path(__file__).parent / "results.jsonl"
# The mock data was written as if today were 24 Oct 2026; cases can override it.
DEFAULT_TODAY = "2026-10-24"


def score(case: dict, answer: str, cited_ids: list[str]) -> dict:
    """Grade one answer against its case with rule-based checks (no LLM, no quota).

    Markdown asterisks are removed first, so "**30 Oct**" matches like "30 Oct". Returns both
    pass/fail flags plus what exactly failed, which is what you read when reviewing failures
    by hand to tell a real mistake from a pattern that was too strict.
    """
    text = answer.replace("*", "")
    missing = [p for p in case.get("must_include", []) if not re.search(p, text, re.IGNORECASE)]
    forbidden = [p for p in case.get("must_not_include", []) if re.search(p, text, re.IGNORECASE)]

    expected = case.get("expected_sources", [])
    relevant = case.get("relevant_sources")
    uncited = [group for group in expected if not set(group) & set(cited_ids)]
    irrelevant = [i for i in cited_ids if relevant is not None and i not in relevant]
    scored_for_citations = bool(expected) or relevant is not None

    return {
        "answer_pass": not missing and not forbidden,
        "citation_pass": (not uncited and not irrelevant) if scored_for_citations else None,
        "missing": missing,
        "forbidden": forbidden,
        "uncited": uncited,
        "irrelevant": irrelevant,
    }


async def run_case(case: dict, hub: agent.ToolHub) -> dict:
    """Run one case in isolation and return its scored result.

    A throwaway SQLite file holds this case's memory, so long-term facts saved while chatting
    (or by earlier cases) can't change the answer.
    """
    with tempfile.TemporaryDirectory() as tmp:
        memory.DB_FILE = Path(tmp) / "eval.db"
        memory.init()
        for fact in case.get("facts", []):
            memory.remember(fact)
        conversation_id = f"eval-{case['id']}"
        for role, text in case.get("history", []):
            memory.add_message(conversation_id, role, text)

        started = time.monotonic()
        reply = await agent.run_agent(
            case["question"], hub, conversation_id, date.fromisoformat(case.get("today", DEFAULT_TODAY))
        )
        seconds = round(time.monotonic() - started, 1)

    cited_ids = [source["id"] for source in reply.sources if source.get("id")]
    return {
        "id": case["id"],
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": reply.model,
        "seconds": seconds,
        "answer": reply.answer,
        "cited": cited_ids,
        "trace": reply.trace,
        **score(case, reply.answer, cited_ids),
    }


def load_results() -> dict[str, dict]:
    """The latest saved result for each case id (later lines in the file win)."""
    results: dict[str, dict] = {}
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text().splitlines():
            if line.strip():
                result = json.loads(line)
                results[result["id"]] = result
    return results


def report(cases: list[dict]) -> None:
    """Print one line per case, the reasons for every failure, and the two accuracy numbers."""
    results = load_results()
    answer_passes = answer_total = citation_passes = citation_total = 0
    models: dict[str, int] = {}

    print(f"\n{'case':28} {'category':22} {'answer':7} {'cite':5} model")
    for case in cases:
        result = results.get(case["id"])
        if result is None:
            print(f"{case['id']:28} {case.get('category', ''):22} {'-':7} {'-':5} (not run yet)")
            continue
        answer_total += 1
        answer_passes += result["answer_pass"]
        if result["citation_pass"] is not None:
            citation_total += 1
            citation_passes += result["citation_pass"]
        models[result["model"]] = models.get(result["model"], 0) + 1
        cite = "-" if result["citation_pass"] is None else ("PASS" if result["citation_pass"] else "FAIL")
        print(f"{case['id']:28} {case.get('category', ''):22} {'PASS' if result['answer_pass'] else 'FAIL':7} "
              f"{cite:5} {result['model']}")
        for key in ("missing", "forbidden", "uncited", "irrelevant"):
            if result[key]:
                print(f"{'':30}{key}: {result[key]}")

    print()
    if answer_total:
        print(f"Answer accuracy:   {answer_passes}/{answer_total} = {answer_passes / answer_total:.0%}")
    if citation_total:
        print(f"Citation accuracy: {citation_passes}/{citation_total} = {citation_passes / citation_total:.0%}")
    print(f"Cases run: {answer_total}/{len(cases)}   Models used: {models}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run and score the Mosaic eval set.")
    parser.add_argument("--only", nargs="+", metavar="CASE_ID", help="run only these cases")
    parser.add_argument("--rerun", action="store_true", help="run cases again even if they have a result")
    parser.add_argument("--model", help="use only this Gemini model instead of the fallback chain")
    parser.add_argument("--pause", type=float, default=5, help="seconds to wait between cases (default 5)")
    parser.add_argument("--report", action="store_true", help="only print the scores saved so far")
    args = parser.parse_args()

    cases = json.loads(CASES_FILE.read_text())
    if args.report:
        report(cases)
        return

    done = load_results()
    todo = [
        case for case in cases
        if (args.rerun or case["id"] not in done) and (not args.only or case["id"] in args.only)
    ]
    if args.model:
        agent.MODELS = [args.model]  # comparable numbers: every case answered by the same model

    hub = agent.ToolHub(data_mode="mock")
    await hub.start()
    try:
        for number, case in enumerate(todo, start=1):
            print(f"\n[{number}/{len(todo)}] {case['id']}: {case['question']}")
            result = await run_case(case, hub)
            if not result["model"]:
                print("Every model is out of quota (or overloaded). Stopping; run again later to resume.")
                break
            with RESULTS_FILE.open("a") as f:
                f.write(json.dumps(result) + "\n")
            print(f"  answer {'PASS' if result['answer_pass'] else 'FAIL'}, citations "
                  f"{'-' if result['citation_pass'] is None else 'PASS' if result['citation_pass'] else 'FAIL'}"
                  f" ({result['model']}, {result['seconds']}s)")
            if number < len(todo):
                await asyncio.sleep(args.pause)
    finally:
        await hub.stop()

    report(cases)


if __name__ == "__main__":
    asyncio.run(main())
