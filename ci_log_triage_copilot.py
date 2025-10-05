import os
import argparse

# Import from our new modules
from config import FAISS_INDEX_PATH, BM25_PATH
from src.data_utils import prepare_data
from src.kb_builder import create_knowledge_base, build_indexes
from src.agent import initialize_retriever, agent_triage, classify_message_weakly, excerpt

def display_triage_card(result):
    """Prints a formatted triage card to the console."""
    print("\n" + "="*20 + " TRIAGE CARD " + "="*20)
    print(f"Message   : {result['message_excerpt']}")
    print("-" * 53)
    print(f"Category  : {result['category']} (Weak Label: {result['weak_label']})")
    print(f"Root Cause: {result['root_cause']}")
    print("Actions   :")
    for i, a in enumerate(result["actions"], 1):
        print(f"  {i}. {a}")
    print("Citations : " + ", ".join(result["citations"]))
    if result["tools_run"]:
        print("Tools Run :")
        for tool_res in result["tools_run"]:
            # Gracefully handle different tool outputs
            details = tool_res.get('hints')
            if details is None:
                details = tool_res.get('found', 'No details')
            print(f"  - {tool_res['tool']}: {details}")
    print("="*53 + "\n")

def main():
    parser = argparse.ArgumentParser(description="CI Log Triage Copilot using RAG and a lightweight agent.")
    parser.add_argument("log_file", nargs='?', default='ci_cd_logs.csv', help="Path to the input CI/CD log file (e.g., ci_cd_logs.csv).")
    parser.add_argument("--build", action="store_true", help="Force rebuild of KB and indexes.")
    parser.add_argument("--limit", type=int, default=5, help="Number of failure logs to process.")
    parser.add_argument("--demo", action="store_true", help="Run the agent on a diverse set of demo messages.")
    args = parser.parse_args()

    # Build KB and indexes if they don't exist or if forced
    if args.build or not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(BM25_PATH):
        print("Building Knowledge Base and Indexes...")
        create_knowledge_base()
        build_indexes()
    else:
        print("KB and Indexes already exist. Skipping build.")

    # Initialize the retrieval system
    initialize_retriever()

    if args.demo:
        print("\n--- Running Triage Agent in DEMO mode ---")
        demo_messages = [
            "Error: ModuleNotFoundError: No module named 'requests'",
            "Job failed: exceeded time limit of 60 minutes.",
            "fatal: Authentication failed for 'https://github.com/user/repo.git/'",
            "OSError: [Errno 28] No space left on device",
            "EAI_AGAIN: temporary failure in name resolution",
            "TypeError: NoneType has no attribute 'split'"
        ]
        for msg in demo_messages:
            weak_cat, weak_conf = classify_message_weakly(msg)
            result = agent_triage(msg, weak_cat, weak_conf)
            display_triage_card(result)
    else:
        # Prepare the data
        fail_df = prepare_data(args.log_file)
        
        # Add weak labels
        weak_labels = fail_df["message"].apply(classify_message_weakly)
        fail_df["category_weak"] = weak_labels.apply(lambda x: x[0])
        fail_df["confidence_weak"] = weak_labels.apply(lambda x: x[1])

        print(f"\n--- Running Triage Agent on {min(args.limit, len(fail_df))} unique Failures ---")
        # Get unique failure messages to ensure a diverse demonstration
        unique_failures_df = fail_df.drop_duplicates(subset=['message']).head(args.limit)
        
        for _, row in unique_failures_df.iterrows():
            result = agent_triage(row["message"], row["category_weak"], row["confidence_weak"])
            display_triage_card(result)

if __name__ == "__main__":
    main()
