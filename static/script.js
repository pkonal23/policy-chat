const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");
const hero = document.getElementById("hero");
const suggBtns = document.querySelectorAll(".sugg-btn");

marked.setOptions({ breaks: true, gfm: true });

suggBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        input.value = btn.dataset.q;
        form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    });
});

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    if(hero) hero.style.display = 'none';

    const turnEl = document.createElement("div");
    turnEl.className = "turn-container";
    
    // 1. Query
    const queryEl = document.createElement("div");
    queryEl.className = "user-query";
    queryEl.textContent = query;
    turnEl.appendChild(queryEl);

    // 2. Loading State (Magic Spinner)
    const loadingEl = document.createElement("div");
    loadingEl.className = "magic-spinner";
    loadingEl.innerHTML = "<div class='spinner-ring'></div> Searching knowledge base...";
    turnEl.appendChild(loadingEl);
    
    chatBox.insertBefore(turnEl, chatBox.lastElementChild); // insert before spacer
    scrollBottom();
    input.value = "";

    try {
        const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query })
        });
        const data = await res.json();
        loadingEl.remove();

        if (res.ok) {
            renderAnswer(turnEl, data);
        } else {
            renderAnswer(turnEl, { answer: `**Error:** ${data.detail || "Request failed."}`, thinking: "", retrieved_nodes: [] });
        }
    } catch {
        loadingEl.remove();
        renderAnswer(turnEl, { answer: "Connection error. Is the server running?", thinking: "", retrieved_nodes: [] });
    }
});

function renderAnswer(container, { answer, thinking, retrieved_nodes }) {
    let html = "";

    if (thinking && thinking !== "No reasoning provided.") {
        const nodesHtml = (retrieved_nodes || []).map(n =>
            `<span class="tag">Pg ${n.page_index} : ${n.title}</span>`
        ).join("");

        html += `
        <div class="ai-reasoning">
            <button class="rsn-btn" onclick="toggleRsn(this)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
                View vectorless reasoning
            </button>
            <div class="rsn-content">
                <p>${escapeHtml(thinking)}</p>
                ${nodesHtml ? `<div class="tags-row">${nodesHtml}</div>` : ""}
            </div>
        </div>`;
    }

    html += `<div class="ai-answer">${marked.parse(answer)}</div>`;
    
    const ansContainer = document.createElement('div');
    ansContainer.innerHTML = html;
    container.appendChild(ansContainer);
    scrollBottom();
}

window.toggleRsn = function(btn) {
    const content = btn.nextElementSibling;
    const isHidden = window.getComputedStyle(content).display === "none";
    content.style.display = isHidden ? "block" : "none";
}

function scrollBottom() {
    chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: "smooth" });
}

function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
