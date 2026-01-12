import sys
import os
import numpy as np
import pickle
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import ACTIVATIONS_DIR, PROBES_DIR, RESULTS_DIR
from src.model_utils import load_model
from src.data_loader import get_truthful_qa_data, create_truthfulqa_splits, format_truthfulqa_for_probing, get_counterfact_data
from src.probe_training import extract_activations_for_probing, train_probes, save_probes
from src.evaluation import get_probe_predictions

def main():
    print(">>> 1. Loading Data...")
    df = get_truthful_qa_data()
    train_df, test_id_df, test_ood_qa_df = create_truthfulqa_splits(df)
    
    # Format
    train_data = format_truthfulqa_for_probing(train_df)
    test_id_data = format_truthfulqa_for_probing(test_id_df)
    test_ood_qa_data = format_truthfulqa_for_probing(test_ood_qa_df)
    
    # Counterfact
    test_ood_cf_data = get_counterfact_data()
    
    print(f"Data Sizes: Train={len(train_data)}, Test_ID={len(test_id_data)}, Test_QA_OOD={len(test_ood_qa_data)}, Test_CF_OOD={len(test_ood_cf_data)}")
    
    # Load Model
    print(">>> 2. Loading Model...")
    model = load_model()
    
    # Extract Activations
    # We save these to disk to save time in next steps
    
    splits = {
        "train": train_data,
        "test_id": test_id_data,
        "test_ood_qa": test_ood_qa_data,
        "test_ood_cf": test_ood_cf_data
    }
    
    for name, data in splits.items():
        if not data:
            continue
        print(f">>> Extracting activations for {name}...")
        X, y = extract_activations_for_probing(model, data, batch_size=8)
        
        # Save
        np.save(ACTIVATIONS_DIR / f"X_{name}.npy", X)
        np.save(ACTIVATIONS_DIR / f"y_{name}.npy", y)
        print(f"Saved {name} activations.")
        
    # Free memory?
    del model
    
    # Train Probes
    print(">>> 3. Training Probes on Train Set...")
    X_train = np.load(ACTIVATIONS_DIR / "X_train.npy")
    y_train = np.load(ACTIVATIONS_DIR / "y_train.npy")
    
    n_layers, n_heads = X_train.shape[1], X_train.shape[2]
    
    probes, train_accs = train_probes(X_train, y_train, n_layers, n_heads)
    save_probes(probes, "probes_logistic.pkl")
    
    # Save training metrics
    with open(RESULTS_DIR / "train_accs.pkl", 'wb') as f:
        pickle.dump(train_accs, f)
        
    print("Done with Stage 1.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
