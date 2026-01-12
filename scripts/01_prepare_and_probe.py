import sys
import os
import numpy as np
import pickle
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import ACTIVATIONS_DIR, PROBES_DIR, RESULTS_DIR
from src.model_utils import load_model
from src.data_loader import get_truthful_qa_data, create_truthfulqa_splits, format_truthfulqa_for_probing, get_counterfact_data
from src.probe_training import extract_activations_for_probing, train_probes, save_probes
from src.evaluation import get_probe_predictions

def main(pooling='last'):
    print(f">>> Running with pooling method: {pooling}")
    print(">>> 1. Loading Data...")
    df = get_truthful_qa_data()
    train_df, test_id_df, test_ood_qa_df = create_truthfulqa_splits(df)

    # Format
    train_data_full = format_truthfulqa_for_probing(train_df)
    test_id_data = format_truthfulqa_for_probing(test_id_df)
    test_ood_qa_data = format_truthfulqa_for_probing(test_ood_qa_df)

    # Split train into train/val (80:20)
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(train_data_full, test_size=0.2, random_state=42)

    # Counterfact
    test_ood_cf_data = get_counterfact_data()

    print(f"Data Sizes: Train={len(train_data)}, Val={len(val_data)}, Test_ID={len(test_id_data)}, Test_QA_OOD={len(test_ood_qa_data)}, Test_CF_OOD={len(test_ood_cf_data)}")

    # Load Model
    print(">>> 2. Loading Model...")
    model = load_model()

    # Extract Activations
    # We save these to disk to save time in next steps

    splits = {
        "train": train_data,
        "val": val_data,
        "test_id": test_id_data,
        "test_ood_qa": test_ood_qa_data,
        "test_ood_cf": test_ood_cf_data
    }

    # Suffix for files based on pooling method
    suffix = f"_{pooling}" if pooling != 'last' else ""

    for name, data in splits.items():
        if not data:
            continue
        print(f">>> Extracting activations for {name} with {pooling} pooling...")
        X, y = extract_activations_for_probing(model, data, batch_size=8, pooling=pooling)

        # Save with pooling suffix
        np.save(ACTIVATIONS_DIR / f"X_{name}{suffix}.npy", X)
        np.save(ACTIVATIONS_DIR / f"y_{name}{suffix}.npy", y)
        print(f"Saved {name} activations.")

    # Free memory?
    del model

    # Train Probes
    print(">>> 3. Training Probes on Train Set, Evaluating on Val Set...")
    X_train = np.load(ACTIVATIONS_DIR / f"X_train{suffix}.npy")
    y_train = np.load(ACTIVATIONS_DIR / f"y_train{suffix}.npy")
    X_val = np.load(ACTIVATIONS_DIR / f"X_val{suffix}.npy")
    y_val = np.load(ACTIVATIONS_DIR / f"y_val{suffix}.npy")

    n_layers, n_heads = X_train.shape[1], X_train.shape[2]

    probes, train_accs, val_accs = train_probes(X_train, y_train, X_val, y_val, n_layers, n_heads)
    save_probes(probes, f"probes_logistic{suffix}.pkl")

    # Save training and validation metrics
    with open(RESULTS_DIR / f"train_accs{suffix}.pkl", 'wb') as f:
        pickle.dump(train_accs, f)
    with open(RESULTS_DIR / f"val_accs{suffix}.pkl", 'wb') as f:
        pickle.dump(val_accs, f)

    print(f"Done with Stage 1 ({pooling} pooling).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract activations and train probes')
    parser.add_argument('--pooling', type=str, default='last', choices=['last', 'mean'],
                        help='Pooling method for activation extraction (default: last)')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--base_dir', type=str, default=None, help='Base directory for data')

    args = parser.parse_args()

    try:
        main(pooling=args.pooling)
    except Exception as e:
        import traceback
        traceback.print_exc()
