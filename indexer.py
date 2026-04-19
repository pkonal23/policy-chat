import os
import json
import time
from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()

PAGEINDEX_API_KEY = os.environ.get("PAGEINDEX_API_KEY")


def create_node_mapping(node, node_map=None):
    """
    Recursively walk the tree and add every node to node_map.
    Handles:
      - bare lists (top-level tree is a list of nodes)
      - dicts with children under 'children' key
      - dicts with children under 'nodes' key (PageIndex often uses this)
    Each node stored in node_map has its child arrays REMOVED (we only want the flat text content).
    """
    if node_map is None:
        node_map = {}

    if isinstance(node, list):
        for item in node:
            create_node_mapping(item, node_map)
        return node_map

    if not isinstance(node, dict):
        return node_map

    node_id = node.get("node_id")

    # Get child arrays before we strip them
    children = node.get("children", [])
    sub_nodes = node.get("nodes", [])

    if node_id:
        # Store a flat copy — no child arrays, just the node itself
        flat_node = {k: v for k, v in node.items() if k not in ("children", "nodes")}
        node_map[node_id] = flat_node

    # Recurse into both possible child keys
    for child in children:
        create_node_mapping(child, node_map)
    for child in sub_nodes:
        create_node_mapping(child, node_map)

    return node_map


def index_pdfs():
    if not PAGEINDEX_API_KEY:
        print("Error: PAGEINDEX_API_KEY not set in .env")
        return

    pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    doc_ids = {
        "UPES POLICY  MANUAL V1.pdf": "pi-cmo617hi012sn01r4g9ha22ew",
        "UPES Policy Manual_Addendum A.pdf": "pi-cmo617jtw12sr01r40v58zc31"
    }

    pdf_files = list(doc_ids.keys())

    print("Waiting for documents to be ready...")
    ready = {f: False for f in pdf_files}
    while not all(ready.values()):
        for f, d_id in doc_ids.items():
            if not ready[f]:
                res = pi_client.get_tree_result(d_id)
                if res.get("status") == "success" or res.get("retrieval_ready") is True:
                    print(f"  Ready: {f}")
                    ready[f] = True
        if not all(ready.values()):
            print("  Still processing... retrying in 10s")
            time.sleep(10)

    all_trees = []
    all_node_maps = {}

    for f, d_id in doc_ids.items():
        print(f"Retrieving tree for {f}...")
        full_result = pi_client.get_tree_result(d_id)
        tree = full_result.get("result", full_result.get("tree", full_result))
        all_trees.append(tree)

        node_map = create_node_mapping(tree)
        print(f"  Extracted {len(node_map)} nodes from {f}")
        all_node_maps.update(node_map)

    combined_tree = {
        "node_id": "root_combined",
        "title": "UPES Policies Document Collection",
        "summary": "Root node containing all UPES policy documents including Addendum.",
        "text": "",
        "page_index": 0,
        "children": all_trees
    }
    all_node_maps["root_combined"] = {k: v for k, v in combined_tree.items() if k != "children"}

    os.makedirs("index_output", exist_ok=True)
    with open("index_output/tree.json", "w") as f:
        json.dump(combined_tree, f, indent=2)

    with open("index_output/node_map.json", "w") as f:
        json.dump(all_node_maps, f, indent=2)

    print(f"\nDone! Saved {len(all_node_maps)} total nodes to index_output/")
    print("  tree.json    — full hierarchical tree for navigation")
    print("  node_map.json — flat lookup map for text retrieval")


if __name__ == "__main__":
    index_pdfs()
