/** Look up a required element by id, failing loudly if the HTML and the script drift apart. */
function byId(id) {
    const element = document.getElementById(id);
    if (!element)
        throw new Error(`Missing #${id} in index.html`);
    return element;
}
const messages = byId("messages");
const form = byId("chat-form");
const input = byId("question");
const sendButton = byId("send");
// One id per page load = one conversation. Every question sends it, so the server can load
// this chat's earlier turns (short-term memory). Reloading the page starts a fresh chat.
const conversationId = crypto.randomUUID();
/** Create an element with a class and optional text. Text always goes in via textContent. */
function el(tag, className = "", text = "") {
    const node = document.createElement(tag);
    if (className)
        node.className = className;
    if (text)
        node.textContent = text;
    return node;
}
/** Only real web links become hrefs, so a javascript: URL from data can never be clicked. */
function safeUrl(url) {
    return url && /^https?:\/\//.test(url) ? url : null;
}
/** A small "GMAIL" / "NOTION" tag naming where an item came from. */
function sourceTag(source) {
    return el("span", `src ${source}`, source === "gmail" ? "Gmail" : "Notion");
}
function formatDate(value) {
    if (!value)
        return "";
    // A bare date ("2026-10-24") is read by Date() as UTC midnight, which shows as 23 Oct in
    // timezones behind UTC. Build it as a local date instead; full timestamps parse as usual.
    const dateOnly = value.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    const parsed = dateOnly ? new Date(Number(dateOnly[1]), Number(dateOnly[2]) - 1, Number(dateOnly[3])) : new Date(value);
    return isNaN(parsed.getTime()) ? value : parsed.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}
/**
 * Add one chat bubble and scroll to it. textContent (not innerHTML) is used on purpose, so
 * text from emails or the LLM is always shown as plain text and can never inject HTML or
 * scripts (XSS).
 */
function addMessage(text, role, pending = false) {
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
function renderInline(line, sources) {
    const nodes = [];
    for (const piece of line.split(/(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[\d+\])/g)) {
        if (!piece)
            continue;
        const bold = piece.match(/^\*\*([^*]+)\*\*$/);
        const italic = piece.match(/^\*([^*]+)\*$/);
        const code = piece.match(/^`([^`]+)`$/);
        const cite = piece.match(/^\[(\d+)\]$/);
        // Bold and italic text is parsed again, so citations or code inside them still render,
        // e.g. *(Note: Razorpay [5] is a general SDE role)* keeps [5] as a clickable chip.
        if (bold) {
            const strong = el("strong");
            strong.append(...renderInline(bold[1], sources));
            nodes.push(strong);
        }
        else if (italic) {
            const em = el("em");
            em.append(...renderInline(italic[1], sources));
            nodes.push(em);
        }
        else if (code) {
            nodes.push(el("code", "inline", code[1]));
        }
        else if (cite && sources.has(Number(cite[1]))) {
            const source = sources.get(Number(cite[1]));
            const url = safeUrl(source.url);
            const chip = el(url ? "a" : "span", "cite", cite[1]);
            chip.title = `${source.source === "gmail" ? "Gmail" : "Notion"}: ${source.title}`;
            if (chip instanceof HTMLAnchorElement && url) {
                chip.href = url;
                chip.target = "_blank";
                chip.rel = "noopener noreferrer";
            }
            nodes.push(chip);
        }
        else {
            nodes.push(document.createTextNode(piece));
        }
    }
    return nodes;
}
/**
 * Turn the answer into block elements: "#"-style headings, "---" dividers, bullet / numbered
 * lists (lines starting "- ", "* " or "1. ") and paragraphs for everything else.
 */
function renderAnswer(answer, sources) {
    const body = el("div", "answer");
    let list = null;
    for (const raw of answer.split("\n")) {
        const line = raw.trim();
        const heading = line.match(/^#{1,6}\s+(.*)$/);
        const bullet = line.match(/^[-*]\s+(.*)$/);
        const numbered = line.match(/^\d+\.\s+(.*)$/);
        if (/^(-{3,}|\*{3,}|_{3,})$/.test(line)) {
            list = null;
            body.append(el("hr"));
        }
        else if (heading) {
            list = null;
            const title = el("h4");
            title.append(...renderInline(heading[1], sources));
            body.append(title);
        }
        else if (bullet || numbered) {
            const tag = bullet ? "ul" : "ol";
            if (!list || list.tagName.toLowerCase() !== tag) {
                list = el(tag);
                body.append(list);
            }
            const item = el("li");
            item.append(...renderInline((bullet ?? numbered)[1], sources));
            list.append(item);
        }
        else if (line) {
            list = null;
            const paragraph = el("p");
            paragraph.append(...renderInline(line, sources));
            body.append(paragraph);
        }
        else {
            list = null;
        }
    }
    return body;
}
/** Source cards under the answer, numbered like the [n] chips. Each card links to the original. */
function renderSources(sources) {
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
        head.append(el("span", "cite", String(source.n)), sourceTag(source.source));
        card.append(head, el("b", "", source.title), el("span", "meta", formatDate(source.date)));
        grid.append(card);
    }
    return grid;
}
/**
 * A tracker update waiting for the student's decision. Nothing changes until "Yes" is clicked.
 * Every word comes from the server's evidence, not the model's prose, so the card shows exactly
 * what will change and which mail it's based on.
 */
function renderAction(action) {
    const card = el("div", "action-card");
    card.append(el("div", "action-title", `Update ${action.title} in your tracker?`));
    const change = el("div", "action-change");
    change.append(el("span", "chip", action.from_status), document.createTextNode(" → "), el("span", "chip warn", action.to_status));
    card.append(change);
    const proof = action.evidence;
    const url = safeUrl(proof?.url ?? null);
    const basis = el(url ? "a" : "div", "action-basis");
    if (basis instanceof HTMLAnchorElement && url) {
        basis.href = url;
        basis.target = "_blank";
        basis.rel = "noopener noreferrer";
    }
    basis.textContent = proof
        ? `Based on: ${proof.title ?? "a mail"} · ${proof.from ?? "unknown sender"} · ${formatDate(proof.date)}`
        : "You asked for this change.";
    card.append(basis);
    const yes = el("button", "pill dark sm", "Yes, update");
    const no = el("button", "pill sm", "No");
    const buttons = el("div", "action-buttons");
    buttons.append(yes, no);
    card.append(buttons);
    const decide = async (approve) => {
        yes.disabled = no.disabled = true;
        let message;
        try {
            const response = await fetch(`/actions/${encodeURIComponent(action.id)}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ conversation_id: conversationId, approve }),
            });
            message = response.ok
                ? (await response.json()).message
                : "This update has expired or was already decided. Ask again if it's still needed.";
        }
        catch {
            message = "Couldn't reach the server, so nothing was changed.";
        }
        buttons.replaceWith(el("div", "action-result", message));
    };
    yes.type = no.type = "button";
    yes.addEventListener("click", () => void decide(true));
    no.addEventListener("click", () => void decide(false));
    return card;
}
/** A collapsible "how I got this" panel: the plan (if any) and every tool call per round. */
function renderTrace(data) {
    if (!data.trace.length && !data.plan.length)
        return null;
    const details = el("details", "msg-trace");
    const toolCalls = data.trace.filter((step) => !step.call.startsWith("plan(")).length;
    details.append(el("summary", "", `Agent trace · ${toolCalls} tool call${toolCalls === 1 ? "" : "s"}`));
    if (data.plan.length) {
        const plan = el("div", "trace-plan", "Plan");
        const steps = el("ol");
        for (const question of data.plan)
            steps.append(el("li", "", question));
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
/**
 * POST the question to /chat/stream and call `onEvent` for every progress event as it arrives.
 * Resolves with the final answer; throws on network errors, non-200 replies and "error" events.
 *
 * The reply is a stream of Server-Sent Events ("data: {json}" blocks separated by a blank
 * line). EventSource can't send a POST body, so the body is read chunk by chunk with fetch.
 * A chunk can end in the middle of an event, so the unfinished tail is kept for the next one.
 */
async function askMosaic(question, onEvent) {
    const body = { question, conversation_id: conversationId };
    const response = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!response.ok || !response.body)
        throw new Error(`Server returned ${response.status}`);
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
        const { value, done } = await reader.read();
        if (done)
            break;
        buffer += value;
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";
        for (const block of blocks) {
            if (!block.startsWith("data: "))
                continue;
            const event = JSON.parse(block.slice(6));
            if (event.type === "answer")
                return event;
            if (event.type === "error")
                throw new Error(event.text);
            onEvent(event);
        }
    }
    throw new Error("The connection closed before the answer arrived");
}
/** "2 results" → "2 mails" / "10 rows"; errors become "failed"; anything else is hidden. */
function friendlyResult(tool, result) {
    if (result.startsWith("ERROR"))
        return "failed";
    const count = result.match(/^(\d+) results?/);
    if (!count)
        return "";
    const n = Number(count[1]);
    const noun = tool === "search_emails" ? "mail" : "row";
    return `${n} ${noun}${n === 1 ? "" : "s"}`;
}
/** One live step line, e.g. [GMAIL] Searching "swiggy"  2 mails. */
function renderLiveStep(event) {
    const item = el("li", "appear");
    const { tool, args } = event;
    if (tool === "search_emails") {
        item.append(sourceTag("gmail"), el("span", "what", `Searching “${String(args.query ?? "")}”`));
    }
    else if (tool === "query_notion") {
        const status = args.status ? ` · ${String(args.status)}` : "";
        item.append(sourceTag("notion"), el("span", "what", `Reading ${String(args.database ?? "")}${status}`));
    }
    else if (tool === "plan") {
        const count = Array.isArray(args.sub_questions) ? args.sub_questions.length : 0;
        item.append(el("span", "what", `Split the question into ${count} parts`));
    }
    else if (tool === "remember") {
        item.append(el("span", "what", "Saving to memory"));
    }
    else if (tool === "propose_status_update") {
        item.append(sourceTag("notion"), el("span", "what", "Preparing a tracker update for you to approve"));
    }
    else {
        item.append(el("span", "what", tool));
    }
    item.append(el("span", "result", friendlyResult(tool, event.result)));
    return item;
}
/**
 * The "working on it" bubble: live step lines, what the agent is doing now, and a timer.
 * Every line comes from a real event (a model call starting, a tool call finishing), so the
 * page never shows progress that didn't happen.
 */
function startLiveBubble() {
    const bubble = addMessage("", "bot", true);
    bubble.classList.add("live");
    const steps = el("ol", "live-steps");
    const label = el("span", "label", "Understanding your question");
    const timer = el("span", "timer", "0s");
    const status = el("div", "live-status");
    status.append(el("span", "spinner"), label, timer);
    bubble.append(steps, status);
    const started = performance.now();
    const tick = window.setInterval(() => {
        timer.textContent = `${Math.round((performance.now() - started) / 1000)}s`;
    }, 1000);
    const handle = (event) => {
        if (event.type === "thinking") {
            label.textContent = event.round === 1 ? "Understanding your question" : "Reading what it found";
        }
        else if (event.type === "step") {
            steps.append(renderLiveStep(event));
        }
        else if (event.type === "retry") {
            // The next model starts over, so the steps so far no longer apply.
            steps.replaceChildren(el("li", "note appear", "Retrying with a backup model"));
            label.textContent = "Understanding your question";
        }
        messages.scrollTop = messages.scrollHeight;
    };
    return { bubble, handle, stop: () => window.clearInterval(tick) };
}
// Questions that work well on the sample data. After each answer, up to three whose topic
// hasn't come up yet are offered as chips, so a visitor always has an easy next thing to try.
// The topic is matched loosely, so "What's happening with Swiggy?" also retires the Swiggy chip.
const SAMPLE_QUESTIONS = [
    { question: "When is my Razorpay interview?", topic: /razorpay/i },
    { question: "What do I need to submit this week?", topic: /submit|due|deadline/i },
    { question: "Status of my Swiggy application?", topic: /swiggy/i },
    { question: "Which companies have rejected me?", topic: /reject/i },
    { question: "Do I have any internship offers?", topic: /offer/i },
    { question: "Is there anything I need to do for my Google application?", topic: /google/i },
];
const asked = [];
function showFollowUps() {
    document.querySelector(".followups")?.remove();
    const next = SAMPLE_QUESTIONS.filter(({ topic }) => !asked.some((question) => topic.test(question)))
        .map(({ question }) => question)
        .slice(0, 3);
    if (!next.length)
        return;
    const row = el("div", "followups appear");
    for (const question of next)
        row.append(suggestionButton(question));
    messages.append(row);
}
function suggestionButton(question) {
    const button = el("button", "pill sm suggestion", question);
    button.type = "button";
    button.addEventListener("click", () => {
        input.value = question;
        form.requestSubmit();
    });
    return button;
}
// On submit: show the question and a live progress bubble, then replace it with the formatted
// answer, its sources and the trace, or a readable error. The button is disabled meanwhile to
// block double sends.
let busy = false; // one question at a time: side-panel buttons submit the form even while one runs
form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question || busy)
        return;
    busy = true;
    document.querySelector(".followups")?.remove();
    asked.push(question);
    addMessage(question, "user");
    input.value = "";
    sendButton.disabled = true;
    const live = startLiveBubble();
    const reply = live.bubble;
    try {
        const data = await askMosaic(question, live.handle);
        const sources = new Map(data.sources.map((source) => [source.n, source]));
        reply.classList.remove("live");
        reply.classList.add("rich");
        reply.replaceChildren(renderAnswer(data.answer, sources));
        if (data.sources.length)
            reply.append(renderSources(data.sources));
        for (const action of data.actions ?? [])
            reply.append(renderAction(action));
        const trace = renderTrace(data);
        if (trace)
            reply.append(trace);
        showFollowUps();
    }
    catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        reply.classList.remove("live");
        reply.textContent = `Something went wrong: ${message}`;
    }
    finally {
        busy = false;
        live.stop();
        reply.classList.remove("pending");
        sendButton.disabled = false;
        input.focus();
        messages.scrollTop = messages.scrollHeight;
    }
});
// Suggested questions (welcome screen and side panel): fill the input and submit through the same handler.
for (const button of document.querySelectorAll(".suggestion")) {
    button.addEventListener("click", () => {
        input.value = button.textContent ?? "";
        form.requestSubmit();
    });
}
/**
 * Describe what Mosaic is answering from: a "Sample data" badge with the pinned date, and the
 * matching side-panel section. On sample data the date matters: "due today" means 24 Oct 2026,
 * not the visitor's real date, and the page says so.
 */
async function showDataSource() {
    const response = await fetch("/config");
    if (!response.ok)
        return;
    const config = await response.json();
    const badge = byId("data-badge");
    if (config.data === "mock") {
        const day = config.today ? formatDate(config.today) : "";
        badge.textContent = day ? `Sample data · today is ${day}` : "Sample data";
        byId("today-note").textContent = day
            ? `The sample data is set on ${day}, so "today" and "this week" are counted from that date.`
            : "";
    }
    else {
        badge.textContent = "Live · Gmail + Notion";
    }
    badge.hidden = false;
    for (const section of document.querySelectorAll("[data-mode]")) {
        section.hidden = section.dataset.mode !== config.data;
    }
}
showDataSource().catch(() => { }); // the chat works without it; only the description is missing
// A question passed in the link (the landing page's examples open /chat?q=...) is asked right
// away. It's removed from the address bar first, so reloading the page doesn't ask it again.
const linkedQuestion = new URLSearchParams(location.search).get("q");
if (linkedQuestion) {
    history.replaceState(null, "", location.pathname);
    input.value = linkedQuestion;
    form.requestSubmit();
}
export {};
