import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Config
BASE_DIR = Path(__file__).parent.parent.resolve()
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

def main():
    df = pd.read_csv(RESULTS_DIR / "final_results.csv")
    
    # Filter for OOD_QA
    df_ood = df[df['Dataset'] == 'OOD_QA']
    
    # Setup Plot
    plt.figure(figsize=(10, 6))
    
    # We want to compare methods. 
    # Baseline is a single line (K=1).
    baseline_acc = df_ood[df_ood['Method'] == 'Baseline (Best ID)']['Accuracy'].max()
    
    # Filter Ensembles
    df_ens = df_ood[df_ood['Method'].isin(['Causal Ensemble', 'Accuracy Ensemble'])]
    
    sns.barplot(data=df_ens, x='K', y='Accuracy', hue='Method')
    
    # Add Baseline Line
    plt.axhline(y=baseline_acc, color='r', linestyle='--', label=f'Baseline (Best ID): {baseline_acc:.3f}')
    
    plt.title("OOD Generalization: Causal Ensemble vs Baseline")
    plt.ylabel("OOD Accuracy")
    plt.ylim(0.4, 0.6) # Zoom in around random
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = PLOTS_DIR / "ood_generalization.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
