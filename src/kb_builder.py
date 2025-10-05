import os
import re
import json
import pickle
import yaml
from glob import glob

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KB_DIR, PLAYBOOK_YAML_PATH, DOCSTORE_PATH, FAISS_INDEX_PATH, BM25_PATH, TFIDF_PATH

def chunk_text(s, size=1200, overlap=150):
    """Chunk text into smaller, overlapping pieces."""
    out, i = [], 0
    while i < len(s):
        ch = s[i:i+size].strip()
        if ch: out.append(ch)
        i += max(1, size - overlap)
    return out

def create_knowledge_base():
    """
    Creates the playbook and guide documents that form the knowledge base.
    """
    print("--- Creating Knowledge Base ---")
    playbook_md_path = os.path.join(KB_DIR, "playbook.md")
    guides_dir = os.path.join(KB_DIR, "guides")
    os.makedirs(guides_dir, exist_ok=True)

    playbook_content = """
# CI Triage Playbook (starter)

## dependency
**Symptoms:** ImportError, ModuleNotFoundError, resolver errors, 'cannot find module'.
**Checks:** requirements.txt/pip freeze; lockfile; build cache.
**Actions:** pin versions; add missing dep; clear cache; rebuild.

## network
**Symptoms:** ECONNRESET, EAI_AGAIN, DNS/timeout.
**Checks:** runner egress/proxy; DNS; flaky endpoints; retry policy.
**Actions:** add retry/backoff; fix proxy/DNS; cache downloads.

## timeout
**Symptoms:** 'timed out', 'exceeded time limit'.
**Checks:** longest tests; parallelism; resource limits.
**Actions:** raise timeout; split/parallelize; cache artifacts.

## auth
**Symptoms:** 401/403, 'permission denied', token errors.
**Checks:** CI secrets/permissions; token scope/expiry; repo/registry access.
**Actions:** rotate tokens; fix secret names; least-privilege scopes.

## infra
**Symptoms:** 'no space left', OOM.
**Checks:** disk/mem; workspace size; job concurrency; container limits.
**Actions:** prune caches; increase resources; limit workers.

## code
**Symptoms:** AssertionError, TypeError, NameError.
**Checks:** failing test locally; recent diffs; static analysis.
**Actions:** fix logic; add guards; narrow PRs.

## flake
**Symptoms:** intermittent failures that pass on rerun.
**Checks:** nondeterministic order/time; network calls in tests.
**Actions:** seed tests; rerun-on-fail; quarantine & deflake.
"""
    with open(playbook_md_path, "w", encoding="utf-8") as f:
        f.write(playbook_content.strip())

    # Create playbook.yml from the markdown
    with open(playbook_md_path, "r", encoding="utf-8") as f:
        text = f.read()
    sections = re.split(r"^##\s+", text, flags=re.M)
    cats = {}
    for sec in sections[1:]:
        head, *body = sec.splitlines()
        cat = head.strip().lower()
        block = "\n".join(body)
        def grab(label):
            m = re.search(rf"\*\*{label}:\*\*(.+?)(?:\n\s*\n|\Z)", block, flags=re.S|re.I)
            if not m: return []
            parts = re.split(r"[;,]\s*", m.group(1).strip())
            return [re.sub(r"^[‘’'“”\s]+|[‘’'“”\s]+", "", p.strip()) for p in parts if p.strip()]
        cats[cat] = {"symptoms": grab("Symptoms"), "checks": grab("Checks"), "actions": grab("Actions")}
    
    with open(PLAYBOOK_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump({"categories": cats}, f, allow_unicode=True)

    # Create guide files
    open(os.path.join(guides_dir, "common_errors.md"), "w", encoding="utf-8").write(
        "# Common CI Failures\nDependency: import errors, resolver conflicts.\nNetwork: proxy/DNS/SSL issues.\nTimeout: long tests, resource limits."
    )
    open(os.path.join(guides_dir, "docker_notes.md"), "w", encoding="utf-8").write(
        "# Docker Notes\nCOPY path mistakes; .dockerignore excludes needed files.\nRUN permission denied -> chmod +x scripts."
    )
    print("Knowledge base created.")

def build_indexes():
    """
    Chunks the KB documents and builds the FAISS, BM25, and TF-IDF indexes.
    """
    print("--- Building Retrieval Indexes ---")
    # 1. Chunk all documents
    chunk_rows = []
    with open(PLAYBOOK_YAML_PATH, "r", encoding="utf-8") as f:
        pb_txt = f.read()
    for i, t in enumerate(chunk_text(pb_txt)):
        chunk_rows.append({"id": f"playbook.yml#{i}", "source": "playbook.yml", "text": t})

    for p in glob(os.path.join(KB_DIR, "guides", "*.md")):
        txt = open(p, "r", encoding="utf-8", errors="ignore").read()
        name = os.path.basename(p)
        for j, ch in enumerate(chunk_text(txt)):
            chunk_rows.append({"id": f"{name}#{j}", "source": name, "text": ch})

    with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
        for r in chunk_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    local_docstore = [json.loads(l) for l in open(DOCSTORE_PATH, "r", encoding="utf-8")]
    texts = [d["text"] for d in local_docstore]
    
    # 2. Build Dense Index (FAISS)
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embs = dense_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        faiss_index = faiss.IndexFlatIP(embs.shape[1])
        faiss_index.add(embs.astype("float32"))
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        print("FAISS index built successfully.")
    except Exception as e:
        print(f"Dense index (FAISS) failed, will rely on sparse only: {e}")

    # 3. Build Sparse Index (BM25)
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [doc.split() for doc in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25}, f)
    print("BM25 index built successfully.")

    # 4. Build TF-IDF as a fallback
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=20000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump({"tfidf": tfidf_vectorizer, "X": tfidf_matrix}, f)
    print("TF-IDF index built successfully.")
