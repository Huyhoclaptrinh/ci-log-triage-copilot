import os

# --- Global Settings ---
# Use a dedicated output directory for all generated artifacts
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")
KB_DIR = os.path.join(OUTPUT_DIR, "kb")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval") # New line
DOCSTORE_PATH = os.path.join(ARTIFACTS_DIR, "docstore.jsonl")
FAISS_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")
BM25_PATH = os.path.join(ARTIFACTS_DIR, "bm25.pkl")
TFIDF_PATH = os.path.join(ARTIFACTS_DIR, "tfidf.pkl")
PLAYBOOK_YAML_PATH = os.path.join(KB_DIR, "playbook.yml")

# Embedding Model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" # High performance open model

# LLM Configuration for Triage
LLM_API_KEY = os.environ.get("GEMINI_API_KEY") # Use environment variable for API key
LLM_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash") # Default to gemini-2.0-flash

# Rules for weak labeling
RULES = {
    "dependency": [
        r"\bmodulenotfounderror\b", r"\bimporterror\b", r"cannot find module",
        r"no matching distribution", r"package .* not found", r"npm err! code eresolve",
        r"pip( |3)? (install|resolve).* (failed|error)"
    ],
    "network": [
        r"\beconnreset\b", r"\beai_again\b", r"temporary failure in name resolution",
        r"\b(connection|network)\b .* (timed out|unreachable|reset)",
        r"proxy|dns|name resolution"
    ],
    "timeout": [
        r"\btimeout\b", r"timed out", r"exceeded (the )?time limit", r"job (exceeded|timed)"
    ],
    "auth": [
        r"permission denied", r"unauthorized", r"forbidden", r"invalid credential",
        r"\b(401|403)\b", r"access( is)? denied", r"not authorized"
    ],
    "infra": [
        r"no space left on device", r"disk quota exceeded",
        r"\b(oom|out of memory|cuda out of memory)\b", r"insufficient resources"
    ],
    "code": [
        r"assertionerror", r"typeerror", r"indexerror", r"nameerror", r"nullpointerexception",
        r"segmentation fault", r"stack overflow"
    ],
    "flake": [
        r"\bflak(y|e)\b", r"intermittent", r"race condition", r"retry.*passed", r"non-deterministic"
    ],
}
NEGATE = {} # Optional negative patterns
