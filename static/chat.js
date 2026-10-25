"use strict";
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
/**
 * Add one chat bubble and scroll to it. textContent (not innerHTML) is used on purpose, so
 * text from emails or the LLM is always shown as plain text and can never inject HTML or
 * scripts (XSS).
 */
function addMessage(text, role, pending = false) {
    document.getElementById("empty")?.remove(); // the welcome screen goes away with the first message
    const bubble = document.createElement("div");
    bubble.className = `msg ${role}${pending ? " pending" : ""}`;
    bubble.textContent = text;
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;
    return bubble;
}
/** POST the question to /chat and return the answer. Throws on network errors and non-200 replies. */
async function askMosaic(question) {
    const body = { question };
    const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!response.ok)
        throw new Error(`Server returned ${response.status}`);
    const data = await response.json();
    return data.answer;
}
// On submit: show the question, show "Thinking..." while the agent works, then replace it
// with the answer or a readable error. The button is disabled meanwhile to block double sends.
form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question)
        return;
    addMessage(question, "user");
    input.value = "";
    sendButton.disabled = true;
    const reply = addMessage("Thinking...", "bot", true);
    try {
        reply.textContent = await askMosaic(question);
    }
    catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        reply.textContent = `Something went wrong: ${message}`;
    }
    finally {
        reply.classList.remove("pending");
        sendButton.disabled = false;
        input.focus();
    }
});
// Suggested questions on the welcome screen: fill the input and submit through the same handler.
for (const button of document.querySelectorAll(".suggestion")) {
    button.addEventListener("click", () => {
        input.value = button.textContent ?? "";
        form.requestSubmit();
    });
}
