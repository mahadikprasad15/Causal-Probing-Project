import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Config
BASE_DIR = Path(__file__).parent.parent.resolve()
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

def main():
    print("Loading Results...")
    # Load Acc Matrices
    with open(RESULTS_DIR / "probe_acc_matrices.pkl", 'rb') as f:
        acc_matrices = pickle.load(f)
        
    id_mat = acc_matrices['ID']
    ood_mat = acc_matrices.get('OOD_QA', None)
    
    # Load Causal Scores
    causal_scores = torch.load(RESULTS_DIR / "causal_scores_ablation.pt").cpu().numpy()
    
    n_layers, n_heads = causal_scores.shape # 24, 16 for Qwen 0.5B
    
    # --- Plot 1: Heatmaps Side-by-Side ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    sns.heatmap(id_mat, ax=axes[0], cmap="viridis", vmin=0.4, vmax=0.8)
    axes[0].set_title("ID Validation Accuracy")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    
    if ood_mat is not None:
        sns.heatmap(ood_mat, ax=axes[1], cmap="viridis", vmin=0.4, vmax=0.8)
        axes[1].set_title("OOD Accuracy")
        axes[1].set_xlabel("Head")
        axes[1].set_ylabel("Layer")
        
    sns.heatmap(causal_scores, ax=axes[2], cmap="hot") # Use different cmap for causal
    axes[2].set_title("Causal Importance (Ablation)")
    axes[2].set_xlabel("Head")
    axes[2].set_ylabel("Layer")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "heatmaps_id_ood_causal.png")
    print("Saved heatmaps_id_ood_causal.png")
    
    # --- Plot 2: Layer-wise Performance (Max & Mean) ---
    plt.figure(figsize=(10, 6))
    
    layers = np.arange(n_layers)
    
    # ID
    plt.plot(layers, np.max(id_mat, axis=1), label="ID Max", color="blue", linestyle="-")
    plt.plot(layers, np.mean(id_mat, axis=1), label="ID Mean", color="blue", linestyle="--", alpha=0.5)
    
    # OOD
    if ood_mat is not None:
        plt.plot(layers, np.max(ood_mat, axis=1), label="OOD Max", color="red", linestyle="-")
        plt.plot(layers, np.mean(ood_mat, axis=1), label="OOD Mean", color="red", linestyle="--", alpha=0.5)
        
    plt.xlabel("Layer")
    plt.ylabel("Probe Accuracy")
    plt.title("Layer-wise Probe Information Content")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / "layer_wise_performance.png")
    print("Saved layer_wise_performance.png")
    
    # --- Plot 3: Scatter: Causal Score vs OOD Accuracy ---
    if ood_mat is not None:
        plt.figure(figsize=(8, 8))
        
        flat_causal = causal_scores.flatten()
        flat_ood = ood_mat.flatten()
        
        # Color by layer?
        flat_layers = np.repeat(np.arange(n_layers), n_heads)
        
        scatter = plt.scatter(flat_causal, flat_ood, c=flat_layers, cmap="magma", alpha=0.7)
        plt.colorbar(scatter, label="Layer")
        
        plt.xlabel("Causal Importance Score")
        plt.ylabel("OOD Accuracy")
        plt.title("Do Causal Heads Generalize Better?")
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        m, b = np.polyfit(flat_causal, flat_ood, 1)
        plt.plot(flat_causal, m*flat_causal + b, color='black', linestyle='--', label=f"Fit (m={m:.2f})")
        plt.legend()
        
        plt.savefig(PLOTS_DIR / "scatter_causal_vs_ood.png")
        print("Saved scatter_causal_vs_ood.png")

if __name__ == "__main__":
    main()
