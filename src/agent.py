import os
import re
import json
import pickle
import yaml
import numpy as np

# Import config and rules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DOCSTORE_PATH, FAISS_INDEX_PATH, BM25_PATH, TFIDF_PATH, PLAYBOOK_YAML_PATH,
    RULES, NEGATE
)

# --- Model and Retrieval Globals ---
model = None
faiss_index = None
bm25 = None
tfidf_vectorizer = None
tfidf_matrix = None
docstore = None
playbook = None

# --- Helper Functions ---

def tokens(s):
    """Basic tokenizer"""
    s = re.sub(r"[^A-Za-z0-9_./:-]+", " ", str(s).lower())
    return [t for t in s.split() if 2 <= len(t) <= 40][:200]

def excerpt(s, n=500):
    """Return the last n characters of a string."""
    s = str(s) if isinstance(s, str) else ""
    return s[-n:]

# --- Agent Tools ---
def tool_grep_logs(message: str, patterns):
    msg = message.lower()
    found = [p for p in patterns if p.lower() in msg]
    return {"tool": "grep_logs", "found": found, "count": len(found)}

def tool_pip_check(message: str):
    msg = message.lower()
    flags = {
        "modulenotfound": "modulenotfounderror" in msg or "no module named" in msg,
        "resolver_conflict": "resolution" in msg or "conflict" in msg or "cannot find a version" in msg,
        "torch_cuda": ("torch" in msg and ("cuda" in msg or "cu11" in msg or "cu12" in msg)),
        "pip_timeout": "read timed out" in msg or "timeout" in msg,
    }
    hints = [k for k, v in flags.items() if v]
    return {"tool": "pip_check", "flags": flags, "hints": hints}

def tool_docker_inspect(message: str):
    msg = message.lower()
    flags = {
        "copy": "copy " in msg or "no such file or directory" in msg,
        "permission": "permission denied" in msg or "operation not permitted" in msg,
    }
    hints = [k for k, v in flags.items() if v]
    return {"tool": "docker_inspect", "flags": flags, "hints": hints}

def tool_retry_with_flag(message: str):
    msg = message.lower()
    flaky = ("timeout" in msg) or ("timed out" in msg) or ("flake" in msg) or ("flaky" in msg)
    return {"tool": "retry_with_flag", "flaky": flaky, "hints": ["rerun with backoff"] if flaky else []}

# --- Retrieval and Agent Logic ---

def initialize_retriever():
    """Loads all the models and indexes into memory."""
    global model, faiss_index, bm25, tfidf_vectorizer, tfidf_matrix, docstore, playbook
    
    print("--- Initializing Retriever ---")
    # Load docstore
    docstore = [json.loads(l) for l in open(DOCSTORE_PATH, "r", encoding="utf-8")]
    
    # Load dense index
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        print("FAISS retriever initialized.")
    except Exception as e:
        print(f"Could not load FAISS index, dense retrieval will be disabled: {e}")
        # Load TF-IDF as fallback
        with open(TFIDF_PATH, "rb") as f:
            td = pickle.load(f)
        tfidf_vectorizer, tfidf_matrix = td["tfidf"], td["X"]
        print("TF-IDF fallback retriever initialized.")

    # Load sparse index
    from rank_bm25 import BM25Okapi
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)["bm25"]
    print("BM25 retriever initialized.")

    # Load playbook for actions
    playbook = yaml.safe_load(open(PLAYBOOK_YAML_PATH, "r", encoding="utf-8").read())["categories"]
    print("Playbook loaded.")

def zscore(a):
    a = np.array(a, dtype=float)
    return (a - a.mean()) / (a.std() + 1e-9)

def retrieve(query, topk=8, k_dense=6, k_bm25=6, alpha_dense=0.6, fuse="zsum"):
    """
    Performs hybrid retrieval over the dense and sparse indexes.
    """
    id2doc = {i: d for i, d in enumerate(docstore)}
    
    # Dense branch
    if faiss_index and model:
        q = model.encode([query], normalize_embeddings=True)
        sims, idx = faiss_index.search(q.astype("float32"), k_dense)
        d_ids, d_sc = idx[0].tolist(), sims[0].tolist()
    else: # Fallback to TF-IDF
        from sklearn.metrics.pairwise import cosine_similarity
        qv = tfidf_vectorizer.transform([query])
        sims = cosine_similarity(qv, tfidf_matrix)[0]
        d_ids = np.argsort(-sims)[:k_dense].tolist()
        d_sc = sims[d_ids].tolist()

    # Sparse branch
    bm_all = bm25.get_scores(query.split())
    b_ids = np.argsort(-bm_all)[:k_bm25].tolist()

    # Fuse
    cand = sorted(set(d_ids + b_ids))
    cand = [c for c in cand if c != -1] # Filter out invalid FAISS index
    d_map = {i: (d_sc[d_ids.index(i)] if i in d_ids else 0.0) for i in cand}
    s_map = {i: bm_all[i] for i in cand}
    if fuse == "zsum":
        dz = zscore([d_map[i] for i in cand])
        sz = zscore([s_map[i] for i in cand])
        fused = {i: dz[cand.index(i)] + sz[cand.index(i)] for i in cand}
    else: # weighted
        fused = {i: alpha_dense*d_map[i] + (1-alpha_dense)*s_map[i] for i in cand}

    ranked = sorted(cand, key=lambda i: fused[i], reverse=True)[:topk]
    return [{"id": id2doc[i]["id"], "source": id2doc[i]["source"], "text": id2doc[i]["text"], "score": fused[i]} for i in ranked]

def classify_message_weakly(msg: str):
    """Applies regex rules to get a 'weak' category label and confidence."""
    if not isinstance(msg, str) or not msg.strip():
        return "unknown", 0.0
    
    m = msg.lower()
    # Assign confidence based on rule specificity
    if ("modulenotfounderror" in m) or ("no module named" in m) or ("could not find a version" in m):
        return "dependency", 0.9
    if any(x in m for x in ["econnreset", "temporary failure in name resolution"]):
        return "network", 0.85
    if any(x in m for x in ["exceeded time limit", "timed out"]):
        return "timeout", 0.8
    if any(x in m for x in ["403 forbidden", "permission denied", "unauthorized"]):
        return "auth", 0.9
    if any(x in m for x in ["no space left", "oom-killed"]):
        return "infra", 0.9
    if any(x in m for x in ["typeerror", "assertionerror", "segmentation fault"]):
        return "code", 0.75
    
    # Generic rules from config with lower confidence
    compiled_rules = {k: [re.compile(p, re.I) for p in pats] for k, pats in RULES.items()}
    for cat, regexes in compiled_rules.items():
        if any(r.search(m) for r in regexes):
            return cat, 0.6 # Lower confidence for generic matches

    return "unknown", 0.0

def decide_category(message, weak_cat, weak_conf, retrieved, threshold=0.8):
    """Decide category based on weak label confidence or retrieval results."""
    if weak_conf >= threshold:
        return weak_cat
    
    txt = " ".join(r["text"].lower() for r in retrieved[:3])
    for k in ["dependency","network","timeout","auth","infra","code","flake"]:
        if k in txt: return k
    return "unknown"

def propose_actions(category):
    acts = playbook.get(category, {}).get("actions", [])[:3]
    root = {
        "dependency": "missing/incompatible package or resolver conflict",
        "network": "proxy/DNS/SSL blocking downloads or connections",
        "timeout": "job/test exceeded time limit",
        "auth": "insufficient scope/expired token/forbidden resource",
        "infra": "disk/mem/workspace limit or quota exceeded",
        "code": "logic/runtime error in code/test",
        "flake": "nondeterministic timing/network/test behavior"
    }.get(category, "unknown")
    return root, acts

def agent_triage(message, weak_cat, weak_conf, tool_budget=3):
    retrieved = retrieve(str(message), topk=8)
    category = decide_category(message, weak_cat, weak_conf, retrieved)
    root, actions = propose_actions(category)
    cites = [r["id"] for r in retrieved[:3]]
    tools_used = []

    # Tool selection policy
    if category in ["dependency", "network"] and tool_budget > 0:
        res = tool_pip_check(message)
        tools_used.append(res)
        tool_budget -= 1
        if "torch_cuda" in res["hints"]:
            actions.append("Check CUDA/toolkit compatibility for torch build")

    if category == "timeout" and tool_budget > 0:
        res = tool_retry_with_flag(message)
        tools_used.append(res)
        tool_budget -= 1
        if res["flaky"] and not any("rerun" in a.lower() for a in actions):
            actions.insert(0, "Add retry/backoff or reduce parallelism")

    # Opportunistic grep to confirm signatures
    if tool_budget > 0:
        patterns_to_grep = ["ModuleNotFoundError", "ECONNRESET", "timeout", "permission denied", "no space left"]
        res = tool_grep_logs(message, patterns_to_grep)
        if res["count"] > 0:
            tools_used.append(res)
        tool_budget -= 1

    # Optional docker hint pass
    if category in ["infra", "code", "unknown"] and tool_budget > 0:
        res = tool_docker_inspect(message)
        if res["hints"]:
            tools_used.append(res)
        tool_budget -= 1

    # Finalize
    return {
        "category": category,
        "root_cause": root,
        "actions": actions[:5],
        "citations": cites,
        "tools_run": tools_used,
        "weak_label": weak_cat,
        "message_excerpt": excerpt(message)
    }