import os
import re
import json
import pickle
import yaml
from glob import glob

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KB_DIR, PLAYBOOK_YAML_PATH, DOCSTORE_PATH, FAISS_INDEX_PATH, BM25_PATH, TFIDF_PATH, ARTIFACTS_DIR, EMBEDDING_MODEL_NAME

def get_kb_last_modified():
    """Returns the last modification time of the playbook.md file."""
    playbook_md_path = os.path.join(KB_DIR, "playbook.md")
    if os.path.exists(playbook_md_path):
        return os.path.getmtime(playbook_md_path)
    return 0

def semantic_chunk_markdown(text, header_level="##"):
    """
    Chunks markdown text by headers to ensure semantic units (Symptom+Action) 
    stay together.
    """
    # Split by the header level (e.g., "## ")
    # The lookahead assertion (?=...) keeps the delimiter
    sections = re.split(f"(?={header_level} )", text)
    
    chunks = []
    current_chunk = ""
    
    for sec in sections:
        if not sec.strip(): continue
        
        # If adding this section exceeds a safe size limit (e.g. 2000 chars), 
        # we treat the current_chunk as finished and start a new one.
        # Otherwise, we can group small adjacent sections if desired, 
        # but for the playbook, usually 1 section = 1 concept.
        if len(sec) > 2000:
             # Fallback for massive sections: split them by paragraph
             sub_parts = chunk_text(sec, size=1200, overlap=150)
             chunks.extend(sub_parts)
        else:
            chunks.append(sec.strip())
            
    return chunks

def chunk_text(s, size=1200, overlap=150):
    """Fallback chunker for non-structured text."""
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
# CI Triage Playbook (Comprehensive)

## dependency
**Symptoms:** 
- `ModuleNotFoundError: No module named 'xyz'`
- `ImportError: cannot import name 'abc'`
- `pip install` fails with `No matching distribution found`
- `npm ERR! code ERESOLVE` / `upstream dependency conflict`
- `Could not find a version that satisfies the requirement`
**Checks:** 
- Check `requirements.txt`, `setup.py`, or `package.json` for typos.
- Verify the package exists on PyPI/npm registry.
- Check if the package supports the python/node version used in CI.
- Inspect `pip freeze` or `npm list` output in CI logs.
**Actions:** 
- Pin exact versions in `requirements.txt` / `package.json`.
- Add the missing dependency to the manifest.
- Clear CI dependency cache (e.g., `actions/cache`).
- Upgrade pip/npm: `pip install --upgrade pip` / `npm install -g npm`.

## network
**Symptoms:** 
- `ECONNRESET`, `Connection refused`, `Connection timed out`
- `EAI_AGAIN`, `Temporary failure in name resolution`
- `502 Bad Gateway`, `503 Service Unavailable` from artifact registry
- `SSLError: HTTPSConnectionPool(host='xyz', port=443)`
**Checks:** 
- Is the URL correct and reachable from the CI runner?
- Are there corporate proxy settings (`http_proxy`, `no_proxy`) required?
- Is the external service (PyPI, Docker Hub) down?
- Check DNS resolution in the runner: `nslookup google.com`.
**Actions:** 
- Implement retry logic with exponential backoff (e.g., `curl --retry 5`).
- Whitelist the domain in the firewall.
- Use a local mirror or cache for dependencies.
- Fix proxy environment variables.

## timeout
**Symptoms:** 
- `Job exceeded maximum allowed time`
- `The job was canceled because it exceeded the time limit`
- Test suite hangs indefinitely without output.
- `command timed out` after X seconds.
**Checks:** 
- Identify the specific step taking too long (timestamps in logs).
- Are tests waiting on a network resource or database lock?
- Did a recent change increase test coverage significantly?
**Actions:** 
- Increase the job timeout limit (e.g., `timeout-minutes: 60`).
- Split tests into parallel jobs (sharding).
- Ensure integration tests tear down resources correctly.
- Cache large assets (Docker layers, dependencies) to speed up build.

## auth
**Symptoms:** 
- `401 Unauthorized`, `403 Forbidden`
- `fatal: Authentication failed for 'https://github.com/...'`
- `Access Denied`, `Permission denied (publickey)`
- `docker login` fails.
**Checks:** 
- Are the secrets/tokens correctly exposed to the CI job?
- Has the Personal Access Token (PAT) expired?
- Does the token have the correct scopes (e.g., `repo`, `read:packages`)?
- Is the CI runner authorized to access the private registry?
**Actions:** 
- Rotate the expired API token/secret.
- Grant "Read/Write" permissions to the GITHUB_TOKEN.
- Use SSH keys instead of HTTPS for git operations if possible.
- Verify secret names match in CI config and repo settings.

## infra
**Symptoms:** 
- `OSError: [Errno 28] No space left on device`
- `Exit code 137` (OOM Killed)
- `java.lang.OutOfMemoryError: Java heap space`
- `CUDA out of memory`
**Checks:** 
- Check disk usage: `df -h`.
- Check memory usage: `free -m`.
- Are old Docker images/containers filling up the disk?
- Is the build generating massive log files or artifacts?
**Actions:** 
- Run `docker system prune -af` before the job.
- Increase the runner size (CPU/RAM).
- Set JVM heap limits: `JAVA_OPTS="-Xmx4g"`.
- Limit the number of parallel workers (e.g., `pytest -n 4` -> `-n 2`).

## code
**Symptoms:** 
- `AssertionError: expected X but got Y`
- `TypeError`, `ValueError`, `NullPointerException`
- `Segmentation fault (core dumped)`
- `SyntaxError: invalid syntax`
**Checks:** 
- Does the test pass locally?
- Check the git diff for recent logic changes.
- Are there environment differences (OS, versions) between local and CI?
- Look for uninitialized variables or null references.
**Actions:** 
- Fix the logic error in the source code.
- Add null checks or guard clauses.
- Update the test case if the requirements changed.
- Run static analysis (linting) to catch syntax errors early.

## docker
**Symptoms:**
- `manifest for xyz not found: manifest unknown`
- `COPY failed: file not found in build context`
- `standard_init_linux.go:211: exec user process caused "exec format error"`
- `docker: Error response from daemon: conflict`
**Checks:**
- Check `.dockerignore` - is the file excluded?
- Verify the base image tag exists on Docker Hub.
- Architecture mismatch (building ARM64 on AMD64 runner)?
- Are you trying to overwrite an existing container name?
**Actions:**
- Fix `COPY` paths relative to the build context.
- Use multi-arch builds (buildx).
- Remove conflicting containers before running.
- Ensure the script entrypoint has `chmod +x`.
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
            
            # Split by lines, then clean each line
            raw_parts = m.group(1).strip().split('\n')
            cleaned_parts = []
            for p in raw_parts:
                p_stripped = p.strip()
                if not p_stripped: continue
                # Remove common list prefixes (-, * , numbers like 1.)
                p_cleaned = re.sub(r"^(- |\* |\d+\.\s*)", "", p_stripped).strip()
                cleaned_parts.append(p_cleaned)
            return cleaned_parts
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
    os.makedirs(ARTIFACTS_DIR, exist_ok=True) # Ensure artifacts directory exists
    # 1. Chunk all documents
    chunk_rows = []
    with open(PLAYBOOK_YAML_PATH, "r", encoding="utf-8") as f:
        pb_txt = f.read()
    
    # Use Semantic Chunking for the main playbook
    # Note: PLAYBOOK_YAML_PATH actually contains YAML structure, but we want to chunk the Markdown
    # for better retrieval context. 
    # Let's read the markdown file instead for semantic chunking source.
    playbook_md_path = os.path.join(KB_DIR, "playbook.md")
    if os.path.exists(playbook_md_path):
        pb_md_txt = open(playbook_md_path, "r", encoding="utf-8").read()
        for i, t in enumerate(semantic_chunk_markdown(pb_md_txt)):
            chunk_rows.append({"id": f"playbook.md#{i}", "source": "playbook.md", "text": t})
    else:
        # Fallback if MD missing
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
        dense_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
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
