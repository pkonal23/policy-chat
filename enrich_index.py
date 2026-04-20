"""
enrich_index.py — Post-processing enrichment for the vectorless index.

Reads index_output/node_map.json, generates a 1-sentence summary for each
node that has substantial text, and writes the summaries back into node_map.json.

Run ONCE after indexer.py:
    python3 enrich_index.py

The enriched node_map.json is then automatically used by app.py on next restart.
"""
import os
import re
import json
import time
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MIN_TEXT_LENGTH = 100   # Skip nodes with fewer chars (titles/headers only)
CONCURRENCY = 5         # How many parallel LLM calls at once
SLEEP_BETWEEN_BATCHES = 1.0  # Seconds between batches to avoid rate limiting


def extract_json(text: str) -> dict:
    """Robustly extract the first JSON object from LLM output, ignoring think tags."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON found in: {text[:200]}")

client = AsyncOpenAI(
    api_key=MINIMAX_API_KEY,
    base_url="https://api.minimax.io/v1"
)


async def generate_summary(node_id: str, title: str, text: str) -> tuple[str, str]:
    """Generate a 1-sentence summary for a single node. Returns (node_id, summary)."""
    prompt = (
        f"Section title: {title}\n"
        f"Section text:\n{text[:1500]}\n\n"
        f"Summarize this policy section in one precise sentence (max 25 words). "
        f"Mention specific rules, thresholds, or key actions if present.\n\n"
        f'Output ONLY this JSON: {{"summary": "your sentence here"}}'
    )
    try:
        res = await client.chat.completions.create(
            model="MiniMax-M2.5",
            messages=[
                {"role": "system", "content": "You are a policy summarizer. Output only a JSON object with a single 'summary' key. No other text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100000
        )
        raw = res.choices[0].message.content.strip()
        # Use extract_json first; if JSON is truncated, extract summary string directly
        summary = ""
        try:
            parsed = extract_json(raw)
            summary = parsed.get("summary", "").strip()
        except ValueError:
            # Fallback: grab whatever came after '"summary": "' even if JSON is cut off
            fallback = re.search(r'"summary"\s*:\s*"([^"]{10,})', raw, re.DOTALL)
            if fallback:
                summary = fallback.group(1).strip()
        if summary and not summary.endswith('.'):
            summary += '.'
        return node_id, summary
    except Exception as e:
        print(f"  ✗ Failed {node_id}: {e}")
        return node_id, ""


async def enrich():
    if not MINIMAX_API_KEY:
        print("Error: MINIMAX_API_KEY not set in .env")
        return

    # Load existing node map
    with open("index_output/node_map.json", "r") as f:
        node_map = json.load(f)

    print(f"Loaded {len(node_map)} nodes from node_map.json")

    # Filter nodes that need summarizing (have enough text and no summary yet)
    to_process = [
        (nid, node)
        for nid, node in node_map.items()
        if len(node.get("text", "")) >= MIN_TEXT_LENGTH and not node.get("summary")
    ]
    already_done = len(node_map) - len(to_process)
    print(f"Nodes already summarized: {already_done}")
    print(f"Nodes to process: {len(to_process)}")

    if not to_process:
        print("Nothing to do — all nodes already have summaries!")
        return

    # Process in batches for rate limiting
    total = len(to_process)
    done = 0
    for i in range(0, total, CONCURRENCY):
        batch = to_process[i:i + CONCURRENCY]
        tasks = [
            generate_summary(nid, node.get("title", ""), node.get("text", ""))
            for nid, node in batch
        ]
        results = await asyncio.gather(*tasks)

        for node_id, summary in results:
            if summary:
                node_map[node_id]["summary"] = summary
                done += 1
                print(f"  ✓ [{done}/{total}] {node_map[node_id].get('title', node_id)[:50]}")
                print(f"      → {summary}")

        # Save incrementally (safe against interruption)
        with open("index_output/node_map.json", "w") as f:
            json.dump(node_map, f, indent=2)

        if i + CONCURRENCY < total:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"\n✅ Done! Enriched {done} nodes.")
    print("Restart app.py to load the summaries into the slim tree.")


if __name__ == "__main__":
    asyncio.run(enrich())
