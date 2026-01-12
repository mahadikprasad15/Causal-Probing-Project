import sys
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import ACTIVATIONS_DIR, RESULTS_DIR
from src.probe_training import load_probes
from src.evaluation import get_probe_predictions, evaluate_ensemble, select_top_k_probes

def main():
    print(">>> 1. Loading Artifacts...")
    probes = load_probes("probes_logistic.pkl")

    # Load validation accuracies (for baseline selection)
    with open(RESULTS_DIR / "val_accs.pkl", 'rb') as f:
        val_accs = pickle.load(f)

    # Load Activations
    # Validation set
    X_val = np.load(ACTIVATIONS_DIR / "X_val.npy")
    y_val = np.load(ACTIVATIONS_DIR / "y_val.npy")

    # Test sets
    X_id = np.load(ACTIVATIONS_DIR / "X_test_id.npy")
    y_id = np.load(ACTIVATIONS_DIR / "y_test_id.npy")

    # X_ood, y_ood (QA)
    try:
        X_ood_qa = np.load(ACTIVATIONS_DIR / "X_test_ood_qa.npy")
        y_ood_qa = np.load(ACTIVATIONS_DIR / "y_test_ood_qa.npy")
    except:
        X_ood_qa, y_ood_qa = None, None

    # X_ood_cf (CounterFact)
    try:
        X_ood_cf = np.load(ACTIVATIONS_DIR / "X_test_ood_cf.npy")
        y_ood_cf = np.load(ACTIVATIONS_DIR / "y_test_ood_cf.npy")
    except:
        X_ood_cf, y_ood_cf = None, None

    # Load Causal Scores
    causal_scores = torch.load(RESULTS_DIR / "causal_scores_ablation.pt") # [Layers, Heads]

    n_layers, n_heads = causal_scores.shape
    
    print(">>> 2. Pre-computing Probe Predictions (Batch inference)...")
    # To speed up ensemble sweep
    probs_val = get_probe_predictions(probes, X_val, n_layers, n_heads)
    probs_id = get_probe_predictions(probes, X_id, n_layers, n_heads)

    probs_ood_qa = get_probe_predictions(probes, X_ood_qa, n_layers, n_heads) if X_ood_qa is not None else None
    probs_ood_cf = get_probe_predictions(probes, X_ood_cf, n_layers, n_heads) if X_ood_cf is not None else None

    # Define Datasets for Eval Loop
    eval_sets = {'Val': (probs_val, y_val), 'ID': (probs_id, y_id)}
    if probs_ood_qa is not None: eval_sets['OOD_QA'] = (probs_ood_qa, y_ood_qa)
    if probs_ood_cf is not None: eval_sets['OOD_CF'] = (probs_ood_cf, y_ood_cf)

    results = []

    # --- Pre-computation of All Accuracies ---
    print("Computing all probe accuracies...")
    # Shape: [Layers, Heads]
    acc_matrix = {}

    for name, (probs, y) in eval_sets.items():
        mat = np.zeros((n_layers, n_heads))
        for l in range(n_layers):
            for h in range(n_heads):
                p = probs[:, l, h]
                pred = (p > 0.5).astype(int)
                mat[l, h] = np.mean(pred == y)
        acc_matrix[name] = mat

    # Save these matrices
    with open(RESULTS_DIR / "probe_acc_matrices.pkl", 'wb') as f:
        pickle.dump(acc_matrix, f)
    print(f"Saved probe accuracy matrices to {RESULTS_DIR}/probe_acc_matrices.pkl")

    # --- Strategy A: Best Single Probe (Baseline) ---
    # Find best probe on VALIDATION set (not test!)
    print("Finding Best Single Probe on Validation Set...")
    # Use stored validation accuracies
    best_probe_idx = max(val_accs.items(), key=lambda x: x[1])[0]
    best_val_acc = val_accs[best_probe_idx]

    print(f"Best Baseline Probe: L{best_probe_idx[0]}.H{best_probe_idx[1]} (Val Acc: {best_val_acc:.4f})")

    # Eval Baseline on all sets
    for name, (probs, y) in eval_sets.items():
        p = probs[:, best_probe_idx[0], best_probe_idx[1]]
        pred = (p > 0.5).astype(int)
        acc = np.mean(pred == y)
        results.append({
            "Method": "Baseline (Best Val)",
            "Dataset": name,
            "K": 1,
            "Aggregation": "Single",
            "Accuracy": acc
        })

    # --- Strategy B: Causal Ensembles ---
    # Select Top-K Causal Heads
    # Flatten and sort
    flat_scores = causal_scores.flatten()
    # Sort indices descending
    sorted_indices = torch.argsort(flat_scores, descending=True)

    k_values = [3, 5, 10, 20]
    agg_methods = ['mean', 'vote']  # Using equal weights (removed 'weighted')

    for k in k_values:
        top_k_flat = sorted_indices[:k]
        top_k_tuples = []

        for idx in top_k_flat:
            l = (idx // n_heads).item()
            h = (idx % n_heads).item()
            top_k_tuples.append((l, h))

        for agg in agg_methods:
            # Eval on all sets with equal weights
            for name, (probs, y) in eval_sets.items():
                acc, auc = evaluate_ensemble(probs, y, top_k_tuples, method=agg, weights=None)
                results.append({
                    "Method": "Causal Ensemble",
                    "Dataset": name,
                    "K": k,
                    "Aggregation": agg,
                    "Accuracy": acc,
                    "AUC": auc
                })
                
    # --- Strategy C: Top-K Accuracy Ensembles (Naive Ensemble) ---
    # Just ensemble the top K probes by VALIDATION accuracy
    # To see if Causal selection is better than Accuracy selection

    # Use stored validation accuracies
    top_acc_probes = select_top_k_probes(val_accs, k=max(k_values))

    for k in k_values:
        # Take top k from that list
        subset = top_acc_probes[:k]
        agg = 'mean' # Just compare mean for fair fight

        for name, (probs, y) in eval_sets.items():
            acc, auc = evaluate_ensemble(probs, y, subset, method=agg)
            results.append({
                "Method": "Accuracy Ensemble", # Naive
                "Dataset": name,
                "K": k,
                "Aggregation": agg,
                "Accuracy": acc,
                "AUC": auc
            })

    # Save Results
    df_res = pd.DataFrame(results)
    print("\nResults Summary:")
    print(df_res.groupby(["Method", "Dataset", "K", "Aggregation"])["Accuracy"].mean())
    
    df_res.to_csv(RESULTS_DIR / "final_results.csv", index=False)
    print(f"Saved results to {RESULTS_DIR}/final_results.csv")

if __name__ == "__main__":
    main()
