// Behaviour for the landing page: the product demo (tabs that replay an example agent run),
// scroll-in reveals, and the sticky nav border. The live chat lives in chat.ts.

type Source = "gmail" | "notion";

/** One tool call in the agent trace, shown as `name(args)` with its result. */
interface TraceStep {
  round: number;
  source: Source;
  call: string;
  result: string;
}

/** A piece of the answer: plain text, or a citation number like [1]. */
type AnswerPart = string | { cite: number };

interface SourceCard {
  source: Source;
  title: string;
  meta: string;
}

interface Scenario {
  question: string;
  trace: TraceStep[];
  answer: AnswerPart[];
  sources: SourceCard[]; // sources[0] is citation 1, sources[1] is citation 2
}

// Example runs on the mock data. They mirror real runs of the agent, so the page never
// promises behaviour the agent doesn't have.
const SCENARIOS: Record<string, Scenario> = {
  deadline: {
    question: "When is my DBMS assignment due?",
    trace: [
      { round: 1, source: "notion", call: 'query_notion("tasks")', result: "9 rows" },
      { round: 1, source: "gmail", call: 'search_emails("DBMS")', result: "2 mails" },
    ],
    answer: [
      "30 October, 11:59 PM. It was extended from 25 Oct",
      { cite: 1 },
      ". Your Notion planner still shows the old date",
      { cite: 2 },
      ", so update it.",
    ],
    sources: [
      { source: "gmail", title: "Re: DBMS Assignment 3: Deadline extended", meta: "Dr. Meera Iyer · 22 Oct" },
      { source: "notion", title: "DBMS Assignment 3", meta: "Study planner · due 25 Oct" },
    ],
  },
  mismatch: {
    question: "What's the status of my Swiggy application?",
    trace: [
      { round: 1, source: "notion", call: 'query_notion("applications")', result: "10 rows" },
      { round: 1, source: "gmail", call: 'search_emails("swiggy")', result: "2 mails" },
    ],
    answer: [
      "Rejected. Swiggy Careers wrote on 20 Oct",
      { cite: 1 },
      ", but your tracker still says Applied",
      { cite: 2 },
      ". The other Swiggy mail is a food offer, so I ignored it.",
    ],
    sources: [
      { source: "gmail", title: "Update on your Backend Engineering Intern application", meta: "Swiggy Careers · 20 Oct" },
      { source: "notion", title: "Swiggy · Backend Engineering Intern", meta: "Applications · Applied" },
    ],
  },
  buried: {
    question: "Anything about Google I shouldn't miss?",
    trace: [
      { round: 1, source: "gmail", call: 'search_emails("google")', result: "4 mails" },
      { round: 1, source: "notion", call: 'query_notion("applications")', result: "10 rows" },
    ],
    answer: [
      "Yes: register on the T&P portal by 26 Oct, 6 PM for Google's on-campus test on 1 Nov",
      { cite: 1 },
      ". It landed in Promotions, so it's easy to miss. Your application itself is still at Applied",
      { cite: 2 },
      ".",
    ],
    sources: [
      { source: "gmail", title: "Google on-campus online test: register by 26 October", meta: "T&P Cell · Promotions · 22 Oct" },
      { source: "notion", title: "Google · Software Engineering Intern", meta: "Applications · Applied" },
    ],
  },
};

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>("[data-scenario]"));
const convo = document.getElementById("demo-convo");
const traceList = document.getElementById("demo-trace");

// Incremented on every new run, so timers from a run the user switched away from do nothing.
let runId = 0;

/** Small helper: create an element with a class and optional text (always as textContent). */
function el<K extends keyof HTMLElementTagNameMap>(tag: K, className = "", text = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function citeBadge(n: number): HTMLSpanElement {
  return el("span", "cite", String(n));
}

function dot(source: Source): HTMLElement {
  return el("i", `dot ${source}`);
}

/** Resolve after `ms` milliseconds (instantly when the user prefers reduced motion). */
function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, reduceMotion ? 0 : ms));
}

/**
 * Replay one scenario like a live agent run: the question, then each tool call appearing as
 * "running" before its result, then the final answer with citations and its source cards.
 * Stops early if the user picks another tab mid-run.
 */
async function play(scenario: Scenario): Promise<void> {
  if (!convo || !traceList) return;
  const id = ++runId;
  convo.replaceChildren(el("div", "demo-q appear", scenario.question));
  traceList.replaceChildren();

  for (const step of scenario.trace) {
    await wait(450);
    if (id !== runId) return;
    const item = el("li", "running appear");
    item.append(el("span", "round", `Round ${step.round}`), dot(step.source), el("code", "", step.call), el("span", "result"));
    traceList.append(item);
    await wait(650);
    if (id !== runId) return;
    item.classList.remove("running");
    item.querySelector(".result")!.textContent = step.result;
  }

  await wait(500);
  if (id !== runId) return;
  const lastRound = scenario.trace[scenario.trace.length - 1]?.round ?? 0;
  const final = el("li", "final appear");
  final.append(el("span", "round", `Round ${lastRound + 1}`), el("span", "", "✓"), el("code", "", "final answer"), el("span", "result", "no tool calls"));
  traceList.append(final);

  const answer = el("div", "demo-a appear");
  for (const part of scenario.answer) {
    answer.append(typeof part === "string" ? document.createTextNode(part) : citeBadge(part.cite));
  }
  convo.append(answer);

  await wait(350);
  if (id !== runId) return;
  const sources = el("div", "demo-sources appear");
  scenario.sources.forEach((source, index) => {
    const card = el("div", "source-card");
    const head = el("div", "source-head");
    head.append(citeBadge(index + 1), dot(source.source), document.createTextNode(source.source === "gmail" ? "Gmail" : "Notion"));
    card.append(head, el("b", "", source.title), el("span", "meta", source.meta));
    sources.append(card);
  });
  convo.append(sources);
}

function selectTab(selected: HTMLButtonElement): void {
  for (const tab of tabs) tab.setAttribute("aria-selected", String(tab === selected));
  const scenario = SCENARIOS[selected.dataset.scenario ?? ""];
  if (scenario) void play(scenario);
}

for (const tab of tabs) {
  tab.addEventListener("click", () => selectTab(tab));
}

// Fade sections in as they scroll into view. The demo starts playing the first time it's seen.
let demoStarted = false;
const observer = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add("visible");
      observer.unobserve(entry.target);
      if (entry.target.id === "demo" && !demoStarted && tabs[0]) {
        demoStarted = true;
        selectTab(tabs[0]);
      }
    }
  },
  { threshold: 0.15 },
);
document.querySelectorAll(".reveal").forEach((node) => observer.observe(node));

// Show a hairline under the nav only once the page has scrolled.
const nav = document.getElementById("nav");
window.addEventListener("scroll", () => nav?.classList.toggle("scrolled", window.scrollY > 8), { passive: true });
