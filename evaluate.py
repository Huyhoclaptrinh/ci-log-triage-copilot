import json
import numpy as np
from rapidfuzz import fuzz

# Import from our new modules
from src.agent import initialize_retriever, retrieve, agent_triage, classify_message_weakly

# --- Metric Calculation Functions ---

def recall_at_k(ranked, rel, k):
    top = [r["id"] for r in ranked[:k]]
    return len(set(top) & set(rel)) / max(1, len(set(rel)))

def mrr(ranked, rel):
    for i, r in enumerate(ranked, 1):
        if r["id"] in rel: return 1.0/i
    return 0.0

def ndcg_at_k(ranked, rel, k):
    def dcg(items):
        s = 0.0
        for i, it in enumerate(items[:k], 1):
            gain = 1.0 if it["id"] in rel else 0.0
            s += (2**gain - 1)/np.log2(i+1)
        return s
    ideal = [{"id": rid} for rid in rel]
    return dcg(ranked)/(dcg(ideal)+1e-9)

def action_f1(pred, gold_actions, thresh=70):
    hits, used = 0, set()
    for pa in pred:
        best, jbest = -1, -1
        for j, ga in enumerate(gold_actions):
            if j in used: continue
            s = fuzz.token_set_ratio(pa, ga)
            if s > best: best, jbest = s, j
        if best >= thresh: hits += 1; used.add(jbest)
    p = hits / max(1, len(pred)); r = hits / max(1, len(gold_actions))
    return (0,0,0) if p+r==0 else (p, r, 2*p*r/(p+r))

def evaluate_retrieval(qrels):
    print("--- Evaluating Retrieval Performance ---")
    rows = []
    for r in qrels:
        ranked = retrieve(r["query"], topk=8)
        rows.append({
            "qid": r["qid"],
            "R@5": recall_at_k(ranked, r["relevant_ids"], 5),
            "MRR": mrr(ranked, r["relevant_ids"]),
            "nDCG@5": ndcg_at_k(ranked, r["relevant_ids"], 5),
        })
    avg = {k: float(np.mean([x[k] for x in rows])) for k in ["R@5","MRR","nDCG@5"]}
    
    print("Retrieval Metrics (Avg):")
    for key, value in avg.items():
        print(f"  {key}: {value:.3f}")
    return avg

def evaluate_triage(tri_gold):
    print("\n--- Evaluating Triage Performance ---")
    rows = []
    for g in tri_gold:
        # Get weak label and confidence first
        weak_cat, weak_conf = classify_message_weakly(g["failure"])
        
        # Pass them to the agent
        pred = agent_triage(g["failure"], weak_cat, weak_conf)
        
        cat_acc = 1.0 if pred["category"] == g["gold"]["category"] else 0.0
        p,r,f1 = action_f1(pred["actions"], g["gold"]["actions"])
        rows.append({"id": g["id"], "cat_acc": cat_acc, "act_p": p, "act_r": r, "act_f1": f1})
    
    avg = {k: float(np.mean([s[k] for s in rows])) for k in ["cat_acc","act_p","act_r","act_f1"]}

    print("Triage Metrics (Avg):")
    print(f"  Category Accuracy: {avg['cat_acc']:.3f}")
    print(f"  Action Precision: {avg['act_p']:.3f}")
    print(f"  Action Recall: {avg['act_r']:.3f}")
    print(f"  Action F1-Score: {avg['act_f1']:.3f}")
    return avg

def main():
    # Load gold standard files
    qrels = [json.loads(l) for l in open("/home/kita/Documents/ci-log-triage-copilot/output/eval/retrieval_qrels.jsonl", "r", encoding="utf-8")]
    tri_gold = [json.loads(l) for l in open("/home/kita/Documents/ci-log-triage-copilot/output/eval/triage_gold.jsonl", "r", encoding="utf-8")]

    # Initialize the retrieval system (loads models, etc.)
    initialize_retriever()

    # Run evaluations
    evaluate_retrieval(qrels)
    evaluate_triage(tri_gold)

if __name__ == "__main__":
    main()