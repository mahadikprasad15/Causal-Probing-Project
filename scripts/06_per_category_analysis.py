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

from src.config import ACTIVATIONS_DIR, RESULTS_DIR, PLOTS_DIR
from src.data_loader import get_truthful_qa_data, create_truthfulqa_splits, format_truthfulqa_for_probing
from src.probe_training import load_probes
from src.evaluation import get_probe_predictions, evaluate_ensemble, select_top_k_probes

def analyze_per_category():
    """Analyze performance per OOD category"""
    print("Loading data with categories...")

    # Load data to get categories
    df = get_truthful_qa_data()
    _, _, test_ood_df = create_truthfulqa_splits(df)
    test_ood_data = format_truthfulqa_for_probing(test_ood_df)

    # Group by category
    categories = {}
    for item in test_ood_data:
        cat = item['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)

    print(f"Found {len(categories)} OOD categories:")
    for cat, items in categories.items():
        print(f"  {cat}: {len(items)} samples")

    # Load artifacts
    probes = load_probes("probes_logistic.pkl")
    with open(RESULTS_DIR / "val_accs.pkl", 'rb') as f:
        val_accs = pickle.load(f)
    causal_scores = torch.load(RESULTS_DIR / "causal_scores_ablation.pt")

    n_layers, n_heads = causal_scores.shape

    # Load activations and labels
    X_ood = np.load(ACTIVATIONS_DIR / "X_test_ood_qa.npy")
    y_ood = np.load(ACTIVATIONS_DIR / "y_test_ood_qa.npy")

    # Get category indices
    # Each item creates 2 samples (correct + incorrect)
    category_indices = {}
    idx = 0
    for item in test_ood_data:
        cat = item['category']
        if cat not in category_indices:
            category_indices[cat] = []
        category_indices[cat].extend([idx, idx+1])  # Both augmented samples
        idx += 2

    # Get predictions for all probes
    probs_ood = get_probe_predictions(probes, X_ood, n_layers, n_heads)

    # Get top-20 for each method
    top_acc = sorted(val_accs.items(), key=lambda x: x[1], reverse=True)[:20]
    top_acc_heads = [k for k, v in top_acc]

    flat_scores = causal_scores.flatten()
    sorted_indices = torch.argsort(flat_scores, descending=True)
    top_causal_heads = []
    for idx in sorted_indices[:20]:
        l = (idx // n_heads).item()
        h = (idx % n_heads).item()
        top_causal_heads.append((l, h))

    # Best single probe
    best_probe_idx = max(val_accs.items(), key=lambda x: x[1])[0]

    # Evaluate per category
    results = []

    for cat, indices in category_indices.items():
        cat_y = y_ood[indices]
        cat_probs = probs_ood[indices]

        # Baseline
        p_baseline = cat_probs[:, best_probe_idx[0], best_probe_idx[1]]
        pred_baseline = (p_baseline > 0.5).astype(int)
        acc_baseline = np.mean(pred_baseline == cat_y)

        # Causal ensemble (K=20, mean)
        causal_ensemble_probs = []
        for l, h in top_causal_heads:
            causal_ensemble_probs.append(cat_probs[:, l, h])
        causal_ensemble_probs = np.stack(causal_ensemble_probs, axis=1)
        final_causal = np.mean(causal_ensemble_probs, axis=1)
        pred_causal = (final_causal > 0.5).astype(int)
        acc_causal = np.mean(pred_causal == cat_y)

        # Accuracy ensemble (K=20, mean)
        acc_ensemble_probs = []
        for l, h in top_acc_heads:
            acc_ensemble_probs.append(cat_probs[:, l, h])
        acc_ensemble_probs = np.stack(acc_ensemble_probs, axis=1)
        final_acc = np.mean(acc_ensemble_probs, axis=1)
        pred_acc = (final_acc > 0.5).astype(int)
        acc_acc = np.mean(pred_acc == cat_y)

        results.append({
            'Category': cat,
            'N_Samples': len(indices),
            'Baseline': acc_baseline,
            'Causal_Ensemble': acc_causal,
            'Accuracy_Ensemble': acc_acc
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('N_Samples', ascending=False)

    print("\n" + "="*80)
    print("PER-CATEGORY OOD PERFORMANCE")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)

    # Save results
    df_results.to_csv(RESULTS_DIR / "per_category_results.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/per_category_results.csv")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Bar chart
    categories_sorted = df_results['Category'].values
    x = np.arange(len(categories_sorted))
    width = 0.25

    axes[0, 0].bar(x - width, df_results['Baseline'], width, label='Baseline', color='gray', alpha=0.7)
    axes[0, 0].bar(x, df_results['Causal_Ensemble'], width, label='Causal Ens.', color='red', alpha=0.7)
    axes[0, 0].bar(x + width, df_results['Accuracy_Ensemble'], width, label='Accuracy Ens.', color='blue', alpha=0.7)

    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Performance by OOD Category', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(categories_sorted, rotation=45, ha='right', fontsize=9)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=0.5, color='black', linestyle='--', alpha=0.3, linewidth=1)

    # Top-right: Improvement over baseline
    df_results['Causal_Gain'] = df_results['Causal_Ensemble'] - df_results['Baseline']
    df_results['Accuracy_Gain'] = df_results['Accuracy_Ensemble'] - df_results['Baseline']

    axes[0, 1].bar(x - width/2, df_results['Causal_Gain'], width, label='Causal Ens.', color='red', alpha=0.7)
    axes[0, 1].bar(x + width/2, df_results['Accuracy_Gain'], width, label='Accuracy Ens.', color='blue', alpha=0.7)

    axes[0, 1].set_ylabel('Improvement over Baseline', fontsize=12)
    axes[0, 1].set_title('Ensemble Gain by Category', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(categories_sorted, rotation=45, ha='right', fontsize=9)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # Bottom-left: Scatter - Accuracy Ens vs Causal Ens
    axes[1, 0].scatter(df_results['Causal_Ensemble'], df_results['Accuracy_Ensemble'],
                      s=df_results['N_Samples']*2, alpha=0.6, c=range(len(df_results)), cmap='viridis')

    for _, row in df_results.iterrows():
        axes[1, 0].annotate(row['Category'],
                           (row['Causal_Ensemble'], row['Accuracy_Ensemble']),
                           fontsize=8, alpha=0.7)

    # Add diagonal
    lims = [0.45, 0.85]
    axes[1, 0].plot(lims, lims, 'k--', alpha=0.3, linewidth=1)

    axes[1, 0].set_xlabel('Causal Ensemble Accuracy', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy Ensemble Accuracy', fontsize=12)
    axes[1, 0].set_title('Method Comparison by Category', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Difficulty vs Performance
    df_results['Difficulty'] = 1 - df_results['Baseline']

    axes[1, 1].scatter(df_results['Difficulty'], df_results['Causal_Ensemble'],
                      label='Causal Ens.', color='red', s=100, alpha=0.7)
    axes[1, 1].scatter(df_results['Difficulty'], df_results['Accuracy_Ensemble'],
                      label='Accuracy Ens.', color='blue', s=100, alpha=0.7)

    axes[1, 1].set_xlabel('Category Difficulty (1 - Baseline)', fontsize=12)
    axes[1, 1].set_ylabel('Ensemble Accuracy', fontsize=12)
    axes[1, 1].set_title('Performance vs Difficulty', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_category_analysis.png", dpi=150)
    print(f"Saved visualization to {PLOTS_DIR}/per_category_analysis.png")
    plt.close()

    # Statistical summary
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    print(f"Mean Accuracy:")
    print(f"  Baseline:          {df_results['Baseline'].mean():.3f} ± {df_results['Baseline'].std():.3f}")
    print(f"  Causal Ensemble:   {df_results['Causal_Ensemble'].mean():.3f} ± {df_results['Causal_Ensemble'].std():.3f}")
    print(f"  Accuracy Ensemble: {df_results['Accuracy_Ensemble'].mean():.3f} ± {df_results['Accuracy_Ensemble'].std():.3f}")

    print(f"\nHardest Categories (by baseline):")
    hardest = df_results.nsmallest(3, 'Baseline')
    for _, row in hardest.iterrows():
        print(f"  {row['Category']:25s}: {row['Baseline']:.3f}")

    print(f"\nEasiest Categories (by baseline):")
    easiest = df_results.nlargest(3, 'Baseline')
    for _, row in easiest.iterrows():
        print(f"  {row['Category']:25s}: {row['Baseline']:.3f}")

    print(f"\nBiggest Gains from Accuracy Ensemble:")
    biggest_gain = df_results.nlargest(3, 'Accuracy_Gain')
    for _, row in biggest_gain.iterrows():
        print(f"  {row['Category']:25s}: +{row['Accuracy_Gain']:.3f}")

    print("="*80)


def main():
    print("="*80)
    print("PER-CATEGORY OOD ANALYSIS")
    print("="*80)

    analyze_per_category()

    print("\nANALYSIS COMPLETE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Per-category OOD analysis')
    parser.add_argument('--base_dir', type=str, default=None, help='Base directory for data')

    args = parser.parse_args()

    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
