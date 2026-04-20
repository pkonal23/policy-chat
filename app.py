import os
import re
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")

client = AsyncOpenAI(
    api_key=MINIMAX_API_KEY,
    base_url="https://api.minimax.io/v1"
)

app = FastAPI(title="Vectorless Policy Chatbot")

# ── Load knowledge index (runs once at startup) ──────────────────────────
TREE = {}
NODE_MAP = {}
try:
    with open("index_output/tree.json", "r") as f:
        TREE = json.load(f)
    with open("index_output/node_map.json", "r") as f:
        NODE_MAP = json.load(f)
    print(f"Loaded index: {len(NODE_MAP)} nodes")
except Exception as e:
    print(f"Warning: Index files not found ({e}). Run indexer.py first.")

# ── Build full-page index (groups ALL node text by page number) ───────────
PAGE_INDEX = {}
for _nid, _node in NODE_MAP.items():
    _page = _node.get("page_index", "?")
    _text = _node.get("text", "")
    _title = _node.get("title", "")
    if _page not in PAGE_INDEX:
        PAGE_INDEX[_page] = []
    PAGE_INDEX[_page].append({"node_id": _nid, "title": _title, "text": _text})
print(f"Page index: {len(PAGE_INDEX)} pages")


def build_lightweight_tree(node):
    """
    Recursively build a minimal tree containing ONLY node_id, title, page_index,
    and optional summary (if enriched). Handles both 'children' and 'nodes' keys.
    """
    if isinstance(node, list):
        return [build_lightweight_tree(item) for item in node]

    if not isinstance(node, dict):
        return node

    node_id = node.get("node_id", "")
    slim = {
        "id": node_id,
        "title": node.get("title", ""),
        "page": node.get("page_index", ""),
    }

    # Pull summary from NODE_MAP if it was enriched (not stored in tree.json)
    if node_id and node_id in NODE_MAP:
        summary = NODE_MAP[node_id].get("summary", "")
        if summary:
            slim["summary"] = summary

    # Recurse into whichever child key exists
    kids = node.get("children") or node.get("nodes")
    if kids:
        slim["sub"] = build_lightweight_tree(kids)

    return slim


# Pre-compute the lightweight tree for prompts (tiny compared to full tree)
SLIM_TREE = build_lightweight_tree(TREE) if TREE else {}
SLIM_TREE_JSON = json.dumps(SLIM_TREE, indent=1)
print(f"Slim tree size: {len(SLIM_TREE_JSON)} chars")

# Pre-compute all section titles for query rewriting context
ALL_TITLES = "\n".join(f"- {n.get('title', '')}" for n in NODE_MAP.values() if n.get('title'))
print(f"Titles index: {len(ALL_TITLES)} chars")


def extract_json(text: str) -> dict:
    """Robustly extract the first JSON object from LLM output, ignoring <think> tags and markdown."""
    # Strip <think>...</think> blocks first
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Strip markdown code fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    # Find the first JSON object
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON found in: {text[:300]}")


async def rewrite_query(raw_query: str) -> str:
    """Use LLM to expand the user query with policy-specific terminology.
    Returns the raw LLM output with ZERO parsing — even if wrapped in <think> tags,
    the downstream tree search LLM will still understand the enriched keywords."""
    try:
        res = await client.chat.completions.create(
            model="MiniMax-M2.5",
            messages=[
                {"role": "system", "content": "You rewrite student questions into formal policy search queries. You are given the actual section titles from the university policy manual. Output relevant keywords and matching section titles that would help find the answer. Just output the keywords, nothing else."},
                {"role": "user", "content": f"Student question: {raw_query}\n\nAvailable policy sections:\n{ALL_TITLES}\n\nRelevant search keywords and matching section titles:"}
            ],
            temperature=0.0,
            max_tokens=150
        )
        # NO PARSING — use raw output as-is, think tags and all
        rewritten = res.choices[0].message.content.strip() or raw_query
        print(f"[Query Rewrite] '{raw_query}' → '{rewritten[:120]}...'")
        return rewritten
    except Exception as e:
        print(f"[Query Rewrite] Failed ({e}), using original query")
        return raw_query


def enrich_with_page_context(node_list: list, existing_pieces: list) -> list:
    """For every page touched by retrieved nodes, pull ALL sibling content from that page."""
    seen_node_ids = {nid for nid in node_list}
    pages_touched = set()
    for nid in node_list:
        if nid in NODE_MAP:
            pages_touched.add(NODE_MAP[nid].get("page_index", "?"))

    enriched_pieces = list(existing_pieces)  # keep originals
    extra_nodes = []
    for page in pages_touched:
        if page in PAGE_INDEX:
            for sibling in PAGE_INDEX[page]:
                if sibling["node_id"] not in seen_node_ids and sibling["text"].strip():
                    seen_node_ids.add(sibling["node_id"])
                    enriched_pieces.append(
                        f"[Page {page}] {sibling['title']}\n{sibling['text']}"
                    )
                    extra_nodes.append({
                        "node_id": sibling["node_id"],
                        "title": sibling["title"],
                        "page_index": page,
                        "text": sibling["text"]
                    })
    if extra_nodes:
        print(f"[Page Enrichment] Added {len(extra_nodes)} sibling nodes from {len(pages_touched)} pages")
    return enriched_pieces, extra_nodes


class ChatRequest(BaseModel):
    query: str
    temperature: Optional[float] = 0.5


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not MINIMAX_API_KEY:
        raise HTTPException(status_code=500, detail="MINIMAX_API_KEY is not configured in .env")

    query = req.query

    # ── STEP 0: Query Rewriting ───────────────────────────────────────
    rewritten_query = await rewrite_query(query)

    # ── STEP 1: Tree Search (Vectorless Reasoning) ────────────────────
    search_prompt = f"""You are a document retrieval assistant. Given a user question and a document tree, identify which nodes likely contain the answer.

Original Question: {query}
Search Keywords: {rewritten_query}

Document Tree (id=node id, title=section title, page=page number, sub=children):
{SLIM_TREE_JSON}

Reply with ONLY a JSON object (no markdown, no explanation, no <think> tags):
{{"thinking": "your brief reasoning about which sections are relevant", "node_list": ["id1", "id2"]}}"""

    try:
        search_res = await client.chat.completions.create(
            model="MiniMax-M2.5",
            messages=[
                {"role": "system", "content": "You are a precise JSON-only assistant. Never use <think> tags. Output raw JSON only."},
                {"role": "user", "content": search_prompt}
            ],
            temperature=0.1,
            max_tokens=2048
        )

        raw = search_res.choices[0].message.content or ""
        print(f"[Tree Search] Raw output ({len(raw)} chars): {raw[:200]}...")
        tree_result = extract_json(raw)

    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Full raw output: {raw}")
        raise HTTPException(status_code=500, detail="The AI returned an unparseable response. Please try again.")
    except Exception as e:
        print(f"Tree Search Error: {e}")
        raise HTTPException(status_code=500, detail="Failed during document search. Please try again.")

    node_list = tree_result.get("node_list", [])
    thinking = tree_result.get("thinking", "No reasoning provided.")

    # ── Retrieve full text from local node map ────────────────────────
    retrieved_nodes = []
    relevant_content_pieces = []

    for node_id in node_list:
        if node_id in NODE_MAP:
            node = NODE_MAP[node_id]
            retrieved_nodes.append({
                "node_id": node.get("node_id", "?"),
                "title": node.get("title", "?"),
                "page_index": node.get("page_index", "?"),
                "text": node.get("text", "")
            })
            text = node.get("text", "")
            relevant_content_pieces.append(
                f"[Page {node.get('page_index', '?')}] {node.get('title', '')}\n{text}"
            )

    # ── Enrich with full page context (sibling nodes from same pages) ─
    enriched_pieces, extra_nodes = enrich_with_page_context(node_list, relevant_content_pieces)
    retrieved_nodes.extend(extra_nodes)

    relevant_content = "\n\n---\n\n".join(enriched_pieces)
    if not relevant_content.strip():
        relevant_content = "No relevant context found in document tree."

    # ── STEP 2: Answer Generation ─────────────────────────────────────
    answer_prompt = f"""You are a knowledgeable UPES policy assistant. Answer based ONLY on the provided context.
If the answer is not in the context, say so clearly.
Always cite page numbers and section titles. Use Markdown formatting.

Question: {query}

Context:
{relevant_content}"""

    try:
        ans_res = await client.chat.completions.create(
            model="MiniMax-M2.5",
            messages=[
                {"role": "system", "content": "You are a helpful university policy expert. Give clear, well-structured answers with citations."},
                {"role": "user", "content": answer_prompt}
            ],
            temperature=req.temperature,
            max_tokens=2048
        )

        final_answer = ans_res.choices[0].message.content or ""
        # Strip any <think> tags from final answer too
        final_answer = re.sub(r'<think>.*?</think>', '', final_answer, flags=re.DOTALL).strip()

    except Exception as e:
        print(f"Answer Generation Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate the answer. Please try again.")

    return {
        "answer": final_answer,
        "thinking": thinking,
        "retrieved_nodes": retrieved_nodes
    }


# ── Serve static frontend ────────────────────────────────────────────
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
