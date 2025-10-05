import os

# --- Global Settings ---
# Use a dedicated output directory for all generated artifacts
BASE_DIR = "/home/kita/Documents/ci-log-triage-copilot"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")
KB_DIR = os.path.join(OUTPUT_DIR, "kb")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
DOCSTORE_PATH = os.path.join(ARTIFACTS_DIR, "docstore.jsonl")
FAISS_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")
BM25_PATH = os.path.join(ARTIFACTS_DIR, "bm25.pkl")
TFIDF_PATH = os.path.join(ARTIFACTS_DIR, "tfidf.pkl")
PLAYBOOK_YAML_PATH = os.path.join(KB_DIR, "playbook.yml")

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
