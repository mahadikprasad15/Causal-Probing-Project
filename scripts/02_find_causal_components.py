import sys
import torch
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR
from src.model_utils import load_model
from src.data_loader import get_truthful_qa_data, create_truthfulqa_splits, format_truthfulqa_for_probing
from src.causal_tracing import run_causal_tracing

def main(model_name=None, token=None):
    print(">>> 1. Preparing Data for Causal Tracing...")
    # Ideally we find causal components on the TRAIN distribution?
    # Or on a held-out ID set?
    # Usually we use the training distribution to find WHAT matters.
    df = get_truthful_qa_data()
    train_df, _, _ = create_truthfulqa_splits(df)
    train_data = format_truthfulqa_for_probing(train_df)

    # Use a subset of 100 examples for tracing to keep it fast
    subset = train_data[:100]

    print(">>> 2. Loading Model...")
    model = load_model(model_name=model_name, token=token)
    
    print(">>> 3. Running Causal Tracing (Head Ablation)...")
    # This might take a while
    scores = run_causal_tracing(model, subset, batch_size=4) 
    # scores shape: [n_layers, n_heads]
    
    print(">>> 4. Saving Causal Scores...")
    torch.save(scores, RESULTS_DIR / "causal_scores_ablation.pt")
    
    # Print top heads
    flat_scores = scores.flatten()
    top_k_indices = torch.topk(flat_scores, k=10).indices
    n_heads = model.cfg.n_heads
    
    print("\nTop 10 Causal Heads (Layer.Head):")
    for idx in top_k_indices:
        layer = idx // n_heads
        head = idx % n_heads
        score = scores[layer, head]
        print(f"L{layer}.H{head}: {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find causal components via ablation')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name/path (default: from config)')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--base_dir', type=str, default=None, help='Base directory for data')

    args = parser.parse_args()

    try:
        main(model_name=args.model, token=args.token)
    except Exception as e:
        import traceback
        traceback.print_exc()
