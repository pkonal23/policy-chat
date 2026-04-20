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
    const magicTexts = [
        "Thinking...",
        "Reading...",
        "Searching...",
        "Analyzing...",
        "Synthesizing..."
    ];
    let magicIdx = 0;
    loadingEl.innerHTML = `<div class='spinner-ring'></div> <span class="magic-dynamic-text" style="transition: opacity 0.3s">${magicTexts[magicIdx]}</span>`;
    turnEl.appendChild(loadingEl);
    
    const magicInterval = setInterval(() => {
        if (magicIdx >= magicTexts.length - 1) {
            clearInterval(magicInterval);
            return;
        }
        magicIdx++;
        const txtSpan = loadingEl.querySelector('.magic-dynamic-text');
        if(txtSpan) {
            txtSpan.style.opacity = 0;
            setTimeout(() => {
                txtSpan.textContent = magicTexts[magicIdx];
                txtSpan.style.opacity = 1;
            }, 300);
        }
    }, 2400);
    
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
        clearInterval(magicInterval);
        loadingEl.remove();

        if (res.ok) {
            renderAnswer(turnEl, data);
        } else {
            renderAnswer(turnEl, { answer: `**Error:** ${data.detail || "Request failed."}`, thinking: "", retrieved_nodes: [] });
        }
    } catch {
        clearInterval(magicInterval);
        loadingEl.remove();
        renderAnswer(turnEl, { answer: "Connection error. Is the server running?", thinking: "", retrieved_nodes: [] });
    }
});

function renderAnswer(container, { answer, thinking, retrieved_nodes }) {
    let html = "";

    if (thinking && thinking !== "No reasoning provided.") {
        const nodesHtml = (retrieved_nodes || []).map(n =>
            `<span class="tag source-tag" style="cursor:pointer;" onclick="openSourceModal(this)" data-title="Page ${n.page_index}: ${escapeHtml(n.title)}" data-text="${escapeHtml(n.text || '')}">Pg ${n.page_index} : ${n.title}</span>`
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

window.openSourceModal = function(el) {
    const title = el.getAttribute("data-title");
    const rawText = el.getAttribute("data-text");
    const modal = document.getElementById("source-modal");
    const titleEl = document.getElementById("modal-title");
    const bodyEl = document.getElementById("modal-text");
    
    const turnEl = el.closest(".turn-container");
    const queryEl = turnEl ? turnEl.querySelector(".user-query") : null;
    const query = queryEl ? queryEl.textContent : "";
    
    const ignoreWords = ["what", "when", "where", "which", "who", "how", "why", "this", "that", "these", "those", "does", "policy", "about", "explain"];
    const rawWords = query.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
    const keywords = [...new Set(rawWords.filter(w => w.length > 3 && !ignoreWords.includes(w)))];
    
    let htmlContent = marked.parse(rawText || "No source text available.");
    
    keywords.forEach(word => {
        const regex = new RegExp(`\\b(${word})\\b(?![^<]*>)`, 'gi');
        htmlContent = htmlContent.replace(regex, '<mark>$1</mark>');
    });
    
    titleEl.textContent = title;
    bodyEl.innerHTML = htmlContent;
    modal.style.display = "flex";
}

window.closeSourceModal = function() {
    document.getElementById("source-modal").style.display = "none";
}
