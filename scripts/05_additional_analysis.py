import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, PLOTS_DIR

def plot_k_sweep_comparison():
    """Plot K vs OOD Accuracy for all methods"""
    df = pd.read_csv(RESULTS_DIR / "final_results.csv")

    # Filter for OOD_QA dataset only
    df_ood = df[df['Dataset'] == 'OOD_QA'].copy()

    plt.figure(figsize=(12, 7))

    # Baseline (horizontal line)
    baseline_acc = df_ood[df_ood['Method'] == 'Baseline (Best Val)']['Accuracy'].values[0]
    plt.axhline(y=baseline_acc, color='black', linestyle='--', linewidth=2,
                label=f'Baseline (Single Best): {baseline_acc:.3f}')

    # Causal Ensemble (mean aggregation)
    causal_mean = df_ood[(df_ood['Method'] == 'Causal Ensemble') &
                         (df_ood['Aggregation'] == 'mean')]
    plt.plot(causal_mean['K'], causal_mean['Accuracy'],
             marker='o', linewidth=2, markersize=8, label='Causal Ensemble (mean)', color='red')

    # Causal Ensemble (vote aggregation)
    causal_vote = df_ood[(df_ood['Method'] == 'Causal Ensemble') &
                         (df_ood['Aggregation'] == 'vote')]
    plt.plot(causal_vote['K'], causal_vote['Accuracy'],
             marker='s', linewidth=2, markersize=8, label='Causal Ensemble (vote)',
             color='red', alpha=0.5, linestyle=':')

    # Accuracy Ensemble (mean aggregation)
    acc_ensemble = df_ood[df_ood['Method'] == 'Accuracy Ensemble']
    plt.plot(acc_ensemble['K'], acc_ensemble['Accuracy'],
             marker='o', linewidth=2, markersize=8, label='Accuracy Ensemble (mean)', color='blue')

    plt.xlabel('K (Number of Heads in Ensemble)', fontsize=12)
    plt.ylabel('OOD Accuracy', fontsize=12)
    plt.title('Ensemble Performance vs Number of Heads (OOD Generalization)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(causal_mean['K'])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "k_sweep_comparison.png", dpi=150)
    print("Saved k_sweep_comparison.png")
    plt.close()


def analyze_probe_overlap():
    """Analyze overlap between top causal and accuracy heads"""
    # Load validation accuracies
    with open(RESULTS_DIR / "val_accs.pkl", 'rb') as f:
        val_accs = pickle.load(f)

    # Load causal scores
    causal_scores = torch.load(RESULTS_DIR / "causal_scores_ablation.pt")
    n_layers, n_heads = causal_scores.shape

    # Get top-20 by accuracy
    top_acc = sorted(val_accs.items(), key=lambda x: x[1], reverse=True)[:20]
    top_acc_set = set([k for k, v in top_acc])

    # Get top-20 by causal score
    flat_scores = causal_scores.flatten()
    sorted_indices = torch.argsort(flat_scores, descending=True)
    top_causal = []
    for idx in sorted_indices[:20]:
        l = (idx // n_heads).item()
        h = (idx % n_heads).item()
        top_causal.append((l, h))
    top_causal_set = set(top_causal)

    # Calculate overlap
    overlap = top_acc_set & top_causal_set

    print("\n" + "="*60)
    print("PROBE OVERLAP ANALYSIS (Top-20)")
    print("="*60)
    print(f"Overlap: {len(overlap)}/20 heads ({len(overlap)/20*100:.1f}%)")
    print(f"\nShared heads: {sorted(overlap)}")
    print(f"\nOnly in Accuracy top-20: {sorted(top_acc_set - top_causal_set)}")
    print(f"\nOnly in Causal top-20: {sorted(top_causal_set - top_acc_set)}")

    # Create Venn diagram-style visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Overlap counts
    categories = ['Accuracy\nOnly', 'Both', 'Causal\nOnly']
    counts = [len(top_acc_set - top_causal_set), len(overlap), len(top_causal_set - top_acc_set)]
    colors = ['blue', 'purple', 'red']
    ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Heads', fontsize=12)
    ax1.set_title('Overlap Between Top-20 Accuracy and Causal Heads', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 22)
    for i, v in enumerate(counts):
        ax1.text(i, v + 0.5, str(v), ha='center', fontsize=14, fontweight='bold')

    # Right: Contribution to accuracy
    # For overlap heads, show their individual contributions
    overlap_accs = [val_accs[head] for head in overlap]
    acc_only_accs = [val_accs[head] for head in (top_acc_set - top_causal_set)]
    causal_only_accs = [val_accs[head] for head in (top_causal_set - top_acc_set)]

    data_to_plot = [acc_only_accs, overlap_accs, causal_only_accs]
    positions = [1, 2, 3]
    bp = ax2.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                     labels=categories)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy Distribution by Group', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "probe_overlap_analysis.png", dpi=150)
    print("Saved probe_overlap_analysis.png")
    plt.close()

    return overlap, top_acc_set, top_causal_set


def plot_vote_vs_mean():
    """Compare vote vs mean aggregation"""
    df = pd.read_csv(RESULTS_DIR / "final_results.csv")

    # Filter for causal ensemble only
    df_causal = df[df['Method'] == 'Causal Ensemble'].copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    datasets = ['Val', 'ID', 'OOD_QA']
    titles = ['Validation', 'In-Distribution Test', 'Out-of-Distribution Test']

    for ax, dataset, title in zip(axes, datasets, titles):
        df_dataset = df_causal[df_causal['Dataset'] == dataset]

        mean_data = df_dataset[df_dataset['Aggregation'] == 'mean']
        vote_data = df_dataset[df_dataset['Aggregation'] == 'vote']

        ax.plot(mean_data['K'], mean_data['Accuracy'],
                marker='o', linewidth=2, markersize=8, label='Mean', color='green')
        ax.plot(vote_data['K'], vote_data['Accuracy'],
                marker='s', linewidth=2, markersize=8, label='Vote', color='orange')

        ax.set_xlabel('K (Number of Heads)', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(mean_data['K'])

    plt.suptitle('Vote vs Mean Aggregation (Causal Ensemble)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "vote_vs_mean_comparison.png", dpi=150)
    print("Saved vote_vs_mean_comparison.png")
    plt.close()


def plot_auc_comparison():
    """Plot AUC scores"""
    df = pd.read_csv(RESULTS_DIR / "final_results.csv")

    # Filter for OOD and methods with AUC
    df_ood = df[(df['Dataset'] == 'OOD_QA') & (df['AUC'].notna())].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Accuracy vs AUC scatter
    colors = {'Causal Ensemble': 'red', 'Accuracy Ensemble': 'blue'}
    for method in colors:
        subset = df_ood[df_ood['Method'] == method]
        ax1.scatter(subset['Accuracy'], subset['AUC'],
                   label=method, color=colors[method], s=100, alpha=0.7)

        # Add K labels
        for _, row in subset.iterrows():
            ax1.annotate(f"K={int(row['K'])}",
                        (row['Accuracy'], row['AUC']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax1.plot([0.5, 1.0], [0.5, 1.0], 'k--', alpha=0.3)  # Diagonal
    ax1.set_xlabel('Accuracy', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('Accuracy vs AUC (OOD)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right: AUC vs K
    for method in colors:
        subset = df_ood[(df_ood['Method'] == method) & (df_ood['Aggregation'] == 'mean')]
        ax2.plot(subset['K'], subset['AUC'],
                marker='o', linewidth=2, markersize=8,
                label=method, color=colors[method])

    ax2.set_xlabel('K (Number of Heads)', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('AUC vs Ensemble Size (OOD)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(subset['K'])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "auc_analysis.png", dpi=150)
    print("Saved auc_analysis.png")
    plt.close()


def analyze_layer_contributions():
    """Analyze layer-wise contributions for best ensembles"""
    # Load artifacts
    with open(RESULTS_DIR / "val_accs.pkl", 'rb') as f:
        val_accs = pickle.load(f)
    causal_scores = torch.load(RESULTS_DIR / "causal_scores_ablation.pt")
    with open(RESULTS_DIR / "probe_acc_matrices.pkl", 'rb') as f:
        acc_matrices = pickle.load(f)

    n_layers, n_heads = causal_scores.shape

    # Get top-20 heads for each method
    top_acc = sorted(val_accs.items(), key=lambda x: x[1], reverse=True)[:20]

    flat_scores = causal_scores.flatten()
    sorted_indices = torch.argsort(flat_scores, descending=True)
    top_causal = []
    for idx in sorted_indices[:20]:
        l = (idx // n_heads).item()
        h = (idx % n_heads).item()
        top_causal.append((l, h))

    # Count heads per layer
    acc_layer_counts = {}
    causal_layer_counts = {}

    for layer, head in top_acc:
        acc_layer_counts[layer] = acc_layer_counts.get(layer, 0) + 1

    for layer, head in top_causal:
        causal_layer_counts[layer] = causal_layer_counts.get(layer, 0) + 1

    # Get OOD accuracy per head
    ood_mat = acc_matrices['OOD_QA']
    acc_ood_scores = [ood_mat[l, h] for l, h in top_acc]
    causal_ood_scores = [ood_mat[l, h] for l, h in top_causal]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Head count per layer
    layers = range(n_layers)
    acc_counts = [acc_layer_counts.get(l, 0) for l in layers]
    causal_counts = [causal_layer_counts.get(l, 0) for l in layers]

    x = np.arange(n_layers)
    width = 0.35
    axes[0, 0].bar(x - width/2, acc_counts, width, label='Accuracy Top-20', color='blue', alpha=0.7)
    axes[0, 0].bar(x + width/2, causal_counts, width, label='Causal Top-20', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Layer', fontsize=11)
    axes[0, 0].set_ylabel('Number of Heads Selected', fontsize=11)
    axes[0, 0].set_title('Layer Distribution of Selected Heads', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Top-right: OOD accuracy per selected head
    axes[0, 1].scatter(range(20), sorted(acc_ood_scores, reverse=True),
                      label='Accuracy Heads', color='blue', s=100, alpha=0.7)
    axes[0, 1].scatter(range(20), sorted(causal_ood_scores, reverse=True),
                      label='Causal Heads', color='red', s=100, alpha=0.7)
    axes[0, 1].set_xlabel('Head Rank', fontsize=11)
    axes[0, 1].set_ylabel('OOD Accuracy', fontsize=11)
    axes[0, 1].set_title('OOD Accuracy of Individual Selected Heads', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Mean OOD accuracy per layer (for selected heads)
    acc_layer_ood = {}
    causal_layer_ood = {}

    for layer, head in top_acc:
        if layer not in acc_layer_ood:
            acc_layer_ood[layer] = []
        acc_layer_ood[layer].append(ood_mat[layer, head])

    for layer, head in top_causal:
        if layer not in causal_layer_ood:
            causal_layer_ood[layer] = []
        causal_layer_ood[layer].append(ood_mat[layer, head])

    acc_layer_means = [np.mean(acc_layer_ood.get(l, [0])) for l in layers]
    causal_layer_means = [np.mean(causal_layer_ood.get(l, [0])) for l in layers]

    axes[1, 0].plot(layers, acc_layer_means, marker='o', linewidth=2,
                   label='Accuracy Heads', color='blue')
    axes[1, 0].plot(layers, causal_layer_means, marker='s', linewidth=2,
                   label='Causal Heads', color='red')
    axes[1, 0].set_xlabel('Layer', fontsize=11)
    axes[1, 0].set_ylabel('Mean OOD Accuracy', fontsize=11)
    axes[1, 0].set_title('Mean OOD Accuracy per Layer (Selected Heads Only)', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Causal score vs validation accuracy for top-20
    acc_causal_scores = [causal_scores[l, h].item() for l, h in top_acc]
    acc_val_accs = [val_accs[k] for k in top_acc]

    causal_causal_scores = [causal_scores[l, h].item() for l, h in top_causal]
    causal_val_accs = [val_accs[(l, h)] for l, h in top_causal]

    axes[1, 1].scatter(acc_causal_scores, acc_val_accs,
                      label='Accuracy Top-20', color='blue', s=100, alpha=0.7)
    axes[1, 1].scatter(causal_causal_scores, causal_val_accs,
                      label='Causal Top-20', color='red', s=100, alpha=0.7)
    axes[1, 1].set_xlabel('Causal Importance Score', fontsize=11)
    axes[1, 1].set_ylabel('Validation Accuracy', fontsize=11)
    axes[1, 1].set_title('Causal Score vs Val Accuracy (Top-20 Heads)', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "layer_contributions_analysis.png", dpi=150)
    print("Saved layer_contributions_analysis.png")
    plt.close()


def main():
    print("="*60)
    print("ADDITIONAL ANALYSIS")
    print("="*60)

    print("\n1. K-Sweep Comparison...")
    plot_k_sweep_comparison()

    print("\n2. Probe Overlap Analysis...")
    analyze_probe_overlap()

    print("\n3. Vote vs Mean Comparison...")
    plot_vote_vs_mean()

    print("\n4. AUC Analysis...")
    plot_auc_comparison()

    print("\n5. Layer Contributions Analysis...")
    analyze_layer_contributions()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Additional analysis and visualizations')
    parser.add_argument('--base_dir', type=str, default=None, help='Base directory for data')

    args = parser.parse_args()

    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
