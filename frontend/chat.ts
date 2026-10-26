export {}; // makes this file a module, so its names don't clash with home.ts in the global scope

// Mirrors the Pydantic models in main.py, so the compiler catches a mismatch between
// what the page sends/reads and what the API expects (e.g. a typo like `data.answr`).
interface ChatRequest {
  question: string;
  conversation_id: string;
}

interface Source {
  n: number;
  source: "gmail" | "notion";
  title: string;
  url: string | null;
  date: string | null;
}

interface TraceStep {
  round: number;
  call: string;
  result: string;
}

interface ChatResponse {
  answer: string;
  sources: Source[];
  plan: string[];
  trace: TraceStep[];
}

type Role = "user" | "bot";

/** Look up a required element by id, failing loudly if the HTML and the script drift apart. */
function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing #${id} in index.html`);
  return element as T;
}

const messages = byId<HTMLDivElement>("messages");
const form = byId<HTMLFormElement>("chat-form");
const input = byId<HTMLInputElement>("question");
const sendButton = byId<HTMLButtonElement>("send");

// One id per page load = one conversation. Every question sends it, so the server can load
// this chat's earlier turns (short-term memory). Reloading the page starts a fresh chat.
const conversationId = crypto.randomUUID();

/** Create an element with a class and optional text. Text always goes in via textContent. */
function el<K extends keyof HTMLElementTagNameMap>(tag: K, className = "", text = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

/** Only real web links become hrefs, so a javascript: URL from data can never be clicked. */
function safeUrl(url: string | null): string | null {
  return url && /^https?:\/\//.test(url) ? url : null;
}

function formatDate(value: string | null): string {
  if (!value) return "";
  const parsed = new Date(value);
  return isNaN(parsed.getTime()) ? value : parsed.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

/**
 * Add one chat bubble and scroll to it. textContent (not innerHTML) is used on purpose, so
 * text from emails or the LLM is always shown as plain text and can never inject HTML or
 * scripts (XSS).
 */
function addMessage(text: string, role: Role, pending = false): HTMLDivElement {
  document.getElementById("empty")?.remove(); // the welcome screen goes away with the first message
  const bubble = el("div", `msg ${role}${pending ? " pending" : ""}`, text);
  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

/**
 * Render the inline parts of one line: **bold**, *italic*, `code` and [n] citations, everything
 * else as text. This is a tiny Markdown subset built from DOM nodes, never innerHTML, so the
 * LLM's output is formatted without ever being able to inject markup. Bold is matched before
 * italic so "**x**" is never read as two italics; italic must start with a non-space, so a
 * "* " bullet marker is never mistaken for it.
 */
function renderInline(line: string, sources: Map<number, Source>): Node[] {
  const nodes: Node[] = [];
  for (const piece of line.split(/(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[\d+\])/g)) {
    if (!piece) continue;
    const bold = piece.match(/^\*\*([^*]+)\*\*$/);
    const italic = piece.match(/^\*([^*]+)\*$/);
    const code = piece.match(/^`([^`]+)`$/);
    const cite = piece.match(/^\[(\d+)\]$/);
    if (bold) {
      nodes.push(el("strong", "", bold[1]));
    } else if (italic) {
      nodes.push(el("em", "", italic[1]));
    } else if (code) {
      nodes.push(el("code", "inline", code[1]));
    } else if (cite && sources.has(Number(cite[1]))) {
      const source = sources.get(Number(cite[1]))!;
      const url = safeUrl(source.url);
      const chip = el(url ? "a" : "span", "cite", cite[1]);
      chip.title = `${source.source === "gmail" ? "Gmail" : "Notion"}: ${source.title}`;
      if (chip instanceof HTMLAnchorElement && url) {
        chip.href = url;
        chip.target = "_blank";
        chip.rel = "noopener noreferrer";
      }
      nodes.push(chip);
    } else {
      nodes.push(document.createTextNode(piece));
    }
  }
  return nodes;
}

/**
 * Turn the answer into block elements: "#"-style headings, "---" dividers, bullet / numbered
 * lists (lines starting "- ", "* " or "1. ") and paragraphs for everything else.
 */
function renderAnswer(answer: string, sources: Map<number, Source>): HTMLDivElement {
  const body = el("div", "answer");
  let list: HTMLUListElement | HTMLOListElement | null = null;
  for (const raw of answer.split("\n")) {
    const line = raw.trim();
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    const bullet = line.match(/^[-*]\s+(.*)$/);
    const numbered = line.match(/^\d+\.\s+(.*)$/);
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line)) {
      list = null;
      body.append(el("hr"));
    } else if (heading) {
      list = null;
      const title = el("h4");
      title.append(...renderInline(heading[1], sources));
      body.append(title);
    } else if (bullet || numbered) {
      const tag = bullet ? "ul" : "ol";
      if (!list || list.tagName.toLowerCase() !== tag) {
        list = el(tag);
        body.append(list);
      }
      const item = el("li");
      item.append(...renderInline((bullet ?? numbered)![1], sources));
      list.append(item);
    } else if (line) {
      list = null;
      const paragraph = el("p");
      paragraph.append(...renderInline(line, sources));
      body.append(paragraph);
    } else {
      list = null;
    }
  }
  return body;
}

/** Source cards under the answer, numbered like the [n] chips. Each card links to the original. */
function renderSources(sources: Source[]): HTMLDivElement {
  const grid = el("div", "msg-sources");
  for (const source of sources) {
    const url = safeUrl(source.url);
    const card = el(url ? "a" : "div", "source-card");
    if (card instanceof HTMLAnchorElement && url) {
      card.href = url;
      card.target = "_blank";
      card.rel = "noopener noreferrer";
    }
    const head = el("div", "source-head");
    head.append(el("span", "cite", String(source.n)), el("i", `dot ${source.source}`), document.createTextNode(source.source === "gmail" ? "Gmail" : "Notion"));
    card.append(head, el("b", "", source.title), el("span", "meta", formatDate(source.date)));
    grid.append(card);
  }
  return grid;
}

/** A collapsible "how I got this" panel: the plan (if any) and every tool call per round. */
function renderTrace(data: ChatResponse): HTMLDetailsElement | null {
  if (!data.trace.length && !data.plan.length) return null;
  const details = el("details", "msg-trace");
  const toolCalls = data.trace.filter((step) => !step.call.startsWith("plan(")).length;
  details.append(el("summary", "", `Agent trace · ${toolCalls} tool call${toolCalls === 1 ? "" : "s"}`));
  if (data.plan.length) {
    const plan = el("div", "trace-plan", "Plan");
    const steps = el("ol");
    for (const question of data.plan) steps.append(el("li", "", question));
    plan.append(steps);
    details.append(plan);
  }
  const list = el("ol", "trace-steps");
  for (const step of data.trace) {
    const item = el("li");
    item.append(el("span", "round", `R${step.round}`), el("code", "", step.call), el("span", "result", step.result));
    list.append(item);
  }
  details.append(list);
  return details;
}

/** POST the question to /chat and return the full reply. Throws on network errors and non-200 replies. */
async function askMosaic(question: string): Promise<ChatResponse> {
  const body: ChatRequest = { question, conversation_id: conversationId };
  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`Server returned ${response.status}`);
  return response.json();
}

// On submit: show the question, show "Thinking..." while the agent works, then replace it with
// the formatted answer, its sources and the trace, or a readable error. The button is disabled
// meanwhile to block double sends.
form.addEventListener("submit", async (event: SubmitEvent) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, "user");
  input.value = "";
  sendButton.disabled = true;
  const reply = addMessage("Thinking...", "bot", true);

  try {
    const data = await askMosaic(question);
    const sources = new Map(data.sources.map((source) => [source.n, source]));
    reply.classList.add("rich");
    reply.replaceChildren(renderAnswer(data.answer, sources));
    if (data.sources.length) reply.append(renderSources(data.sources));
    const trace = renderTrace(data);
    if (trace) reply.append(trace);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    reply.textContent = `Something went wrong: ${message}`;
  } finally {
    reply.classList.remove("pending");
    sendButton.disabled = false;
    input.focus();
    messages.scrollTop = messages.scrollHeight;
  }
});

// Suggested questions on the welcome screen: fill the input and submit through the same handler.
for (const button of document.querySelectorAll<HTMLButtonElement>(".suggestion")) {
  button.addEventListener("click", () => {
    input.value = button.textContent ?? "";
    form.requestSubmit();
  });
}
