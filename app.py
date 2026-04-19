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


def build_lightweight_tree(node):
    """
    Recursively build a minimal tree containing ONLY node_id, title, and page_index.
    Strips text, summary, and all heavy fields. Handles both 'children' and 'nodes' keys,
    as well as bare lists.
    """
    if isinstance(node, list):
        return [build_lightweight_tree(item) for item in node]

    if not isinstance(node, dict):
        return node

    slim = {
        "id": node.get("node_id", ""),
        "title": node.get("title", ""),
        "page": node.get("page_index", ""),
    }

    # Recurse into whichever child key exists
    kids = node.get("children") or node.get("nodes")
    if kids:
        slim["sub"] = build_lightweight_tree(kids)

    return slim


# Pre-compute the lightweight tree for prompts (tiny compared to full tree)
SLIM_TREE = build_lightweight_tree(TREE) if TREE else {}
SLIM_TREE_JSON = json.dumps(SLIM_TREE, indent=1)
print(f"Slim tree size: {len(SLIM_TREE_JSON)} chars")


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


class ChatRequest(BaseModel):
    query: str
    temperature: Optional[float] = 0.5


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not MINIMAX_API_KEY:
        raise HTTPException(status_code=500, detail="MINIMAX_API_KEY is not configured in .env")

    query = req.query

    # ── STEP 1: Tree Search (Vectorless Reasoning) ────────────────────
    search_prompt = f"""You are a document retrieval assistant. Given a user question and a document tree, identify which nodes likely contain the answer.

Question: {query}

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
                "page_index": node.get("page_index", "?")
            })
            text = node.get("text", "")
            relevant_content_pieces.append(
                f"[Page {node.get('page_index', '?')}] {node.get('title', '')}\n{text}"
            )

    relevant_content = "\n\n---\n\n".join(relevant_content_pieces)
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
