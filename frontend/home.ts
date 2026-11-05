export {}; // makes this file a module, so its names don't clash with chat.ts in the global scope

// Behaviour for the landing page: the product demo (tabs that replay an example agent run), the
// step-by-step architecture diagram, the agent-loop animation, the eval counters, scroll-in
// reveals and the sticky nav border. The live chat lives in chat.ts.

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

/** A small "GMAIL" / "NOTION" tag naming where an item came from. */
function sourceTag(source: Source): HTMLElement {
  return el("span", `src ${source}`, source === "gmail" ? "Gmail" : "Notion");
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
    item.append(el("span", "round", `Round ${step.round}`), sourceTag(step.source), el("code", "", step.call), el("span", "result"));
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
    head.append(citeBadge(index + 1), sourceTag(source.source));
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


/** Run `onVisible` while `target` is on screen and `onHidden` when it leaves, so animations idle offscreen. */
function whileVisible(target: Element, onVisible: () => void, onHidden: () => void): void {
  new IntersectionObserver(([entry]) => (entry.isIntersecting ? onVisible() : onHidden()), { threshold: 0.2 }).observe(target);
}

// --- Architecture diagram ---------------------------------------------------------------
// Every node and link in the SVG lists the stages it belongs to in data-stages, and the six
// stage buttons describe them, so this code only switches classes. Stages advance on their own
// while the diagram is visible; clicking one jumps there and restarts the timer.

const STAGE_MS = 4500;
const arch = document.getElementById("arch");
const archSvg = arch?.querySelector<SVGSVGElement>(".arch-svg");
const archCanvas = arch?.querySelector<HTMLElement>(".arch-canvas");
const stageButtons = Array.from(arch?.querySelectorAll<HTMLButtonElement>("[data-stage]") ?? []);
let stage = 1;
let stageTimer: number | undefined;

function showStage(next: number): void {
  stage = next;
  archSvg?.querySelectorAll<SVGElement>("[data-stages]").forEach((part) => {
    part.classList.toggle("on", (part.dataset.stages ?? "").split(" ").includes(String(next)));
  });
  for (const button of stageButtons) {
    const active = button.dataset.stage === String(next);
    button.setAttribute("aria-pressed", String(active));
    if (active) {
      // Restart the progress bar's CSS animation for this stage.
      const bar = button.querySelector<HTMLElement>(".bar");
      if (bar) {
        bar.style.animation = "none";
        void bar.offsetWidth;
        bar.style.animation = "";
      }
    }
  }
  panToStage();
}

/**
 * On narrow screens the diagram is wider than its frame and scrolls sideways, so the active
 * part could be off screen. Pan the frame so this stage's nodes sit in the middle.
 */
function panToStage(): void {
  if (!archCanvas || !archSvg || archCanvas.scrollWidth <= archCanvas.clientWidth) return;
  const boxes = Array.from(archSvg.querySelectorAll<SVGGElement>(".node-g.on")).map((node) => node.getBoundingClientRect());
  if (!boxes.length) return;
  const frame = archCanvas.getBoundingClientRect();
  const middle = (Math.min(...boxes.map((b) => b.left)) + Math.max(...boxes.map((b) => b.right))) / 2;
  const left = archCanvas.scrollLeft + middle - frame.left - frame.width / 2;
  archCanvas.scrollTo({ left, behavior: reduceMotion ? "auto" : "smooth" });
}

function playStages(): void {
  window.clearInterval(stageTimer);
  if (reduceMotion) return;
  arch?.classList.remove("paused");
  stageTimer = window.setInterval(() => showStage((stage % stageButtons.length) + 1), STAGE_MS);
}

function pauseStages(): void {
  window.clearInterval(stageTimer);
  arch?.classList.add("paused");
}

if (arch && archSvg && stageButtons.length) {
  arch.style.setProperty("--stage-ms", `${STAGE_MS}ms`);
  showStage(1);
  for (const button of stageButtons) {
    button.addEventListener("click", () => {
      showStage(Number(button.dataset.stage));
      playStages();
    });
  }
  if (reduceMotion) arch.classList.add("paused");
  whileVisible(arch, playStages, pauseStages);
}

// --- Agent loop -----------------------------------------------------------------------------
// A dot circles the ring once per round, lighting the step it passes. After two rounds it
// leaves along the exit path to "Answer", which is what happens when the model stops asking
// for tools. Driven by requestAnimationFrame so the round counter always matches the dot.

const loopDiagram = document.getElementById("loop-diagram");
const loopDot = document.getElementById("loop-dot");
const loopRound = document.getElementById("loop-round");
const loopExit = document.getElementById("loop-exit") as SVGPathElement | null;
const loopNodes = Array.from(loopDiagram?.querySelectorAll<SVGGElement>(".lnode") ?? []);
const answerNode = loopDiagram?.querySelector<SVGGElement>(".answer-node");

const LAP_MS = 3200; // one round
const EXIT_MS = 700; // ring to "Answer"
const HOLD_MS = 1800; // pause on the answer before starting over
const ROUNDS = 2;
const CYCLE_MS = ROUNDS * LAP_MS + EXIT_MS + HOLD_MS;
const [CX, CY, R] = [210, 220, 150];

let loopFrame = 0;
let loopStart = 0;

function placeDot(x: number, y: number): void {
  loopDot?.setAttribute("cx", x.toFixed(1));
  loopDot?.setAttribute("cy", y.toFixed(1));
}

function drawLoop(now: number): void {
  const t = (now - loopStart) % CYCLE_MS;
  const answered = t >= ROUNDS * LAP_MS;
  if (!answered) {
    const angle = ((t % LAP_MS) / LAP_MS) * Math.PI * 2; // 0 at the top, clockwise
    placeDot(CX + R * Math.sin(angle), CY - R * Math.cos(angle));
    const nearest = Math.round(angle / (Math.PI / 2)) % 4;
    loopNodes.forEach((node) => node.classList.toggle("on", node.dataset.q === String(nearest)));
    if (loopRound) loopRound.textContent = `Round ${Math.floor(t / LAP_MS) + 1}`;
  } else if (loopExit) {
    const progress = Math.min((t - ROUNDS * LAP_MS) / EXIT_MS, 1);
    const point = loopExit.getPointAtLength(progress * loopExit.getTotalLength());
    placeDot(point.x, point.y);
    loopNodes.forEach((node) => node.classList.toggle("on", node.dataset.q === "0"));
    if (loopRound) loopRound.textContent = progress < 1 ? `Round ${ROUNDS + 1}` : "Answered";
  }
  answerNode?.classList.toggle("on", answered && t >= ROUNDS * LAP_MS + EXIT_MS);
  loopFrame = requestAnimationFrame(drawLoop);
}

if (loopDiagram && loopDot) {
  if (reduceMotion) {
    loopNodes[0]?.classList.add("on");
  } else {
    whileVisible(
      loopDiagram,
      () => {
        cancelAnimationFrame(loopFrame);
        loopStart = performance.now();
        loopFrame = requestAnimationFrame(drawLoop);
      },
      () => cancelAnimationFrame(loopFrame),
    );
  }
}

// --- Eval counters ----------------------------------------------------------------------------
// Count up from 0 the first time the numbers scroll into view. The real value is already in
// the HTML, so without JavaScript (or with reduced motion) the page still shows it.

function countUp(node: HTMLElement): void {
  const target = Number(node.dataset.count);
  const decimals = (node.dataset.count ?? "").split(".")[1]?.length ?? 0;
  const started = performance.now();
  const tick = (now: number) => {
    const progress = Math.min((now - started) / 1200, 1);
    const eased = 1 - (1 - progress) ** 3;
    node.textContent = (target * eased).toFixed(decimals);
    if (progress < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

const metrics = document.querySelector(".metrics");
if (metrics && !reduceMotion) {
  const counters = Array.from(metrics.querySelectorAll<HTMLElement>("[data-count]"));
  counters.forEach((node) => (node.textContent = "0"));
  const counterObserver = new IntersectionObserver(([entry]) => {
    if (!entry.isIntersecting) return;
    counters.forEach(countUp);
    counterObserver.disconnect();
  }, { threshold: 0.4 });
  counterObserver.observe(metrics);
}

