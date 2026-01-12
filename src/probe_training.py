import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import os
from src.config import PROBES_DIR

def extract_activations_for_probing(model, dataset, batch_size=8, pooling='last'):
    """
    Extracts activations for every head in the model.
    Data augmentation: Creates 2 examples per dataset item:
      1. Prompt + Correct Answer (Label 1)
      2. Prompt + Incorrect Answer (Label 0)

    Args:
        model: HookedTransformer model
        dataset: List of dicts with 'prompt', 'correct', 'incorrect'
        batch_size: Batch size for processing
        pooling: Pooling method - 'last' (last token) or 'mean' (mean over all tokens)

    Returns:
        activations: [N_samples, n_layers, n_heads, d_head]
        labels: [N_samples]
    """
    
    # Prepare text
    texts = []
    labels = []
    
    for item in dataset:
        # Prompt + Correct
        texts.append(item['prompt'] + " " + item['correct'])
        labels.append(1)
        
        # Prompt + Incorrect
        texts.append(item['prompt'] + " " + item['incorrect'])
        labels.append(0)
        
    # Run in batches
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    
    # Initialize storage
    # Using a numpy array might be better for memory than a dict of lists
    # Shape: (N_samples, n_layers, n_heads, d_head)
    # Warning: 1000 samples * 32 * 32 * 64 * 4 bytes could be large ~ 250MB. Fine for 1B model.
    
    all_acts = []
    
    print(f"Extracting activations for {len(texts)} samples...")

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]

        with torch.no_grad():
            # Run model and cache activations
            logits, cache = model.run_with_cache(
                batch_texts,
                names_filter=lambda name: name.endswith("attn.hook_z"),
                return_type=None
            )

            # Get tokens and attention mask to find actual last token position
            tokens = model.to_tokens(batch_texts, prepend_bos=True)  # [batch, seq_len]

            # Find last non-padding token for each sample
            # Check if model has a pad token
            pad_token_id = model.tokenizer.pad_token_id if hasattr(model.tokenizer, 'pad_token_id') else None

            # Find actual last token positions (accounting for padding)
            last_token_positions = []
            for b in range(tokens.shape[0]):
                if pad_token_id is not None:
                    # Find last non-pad token
                    non_pad_mask = tokens[b] != pad_token_id
                    non_pad_indices = torch.where(non_pad_mask)[0]
                    if len(non_pad_indices) > 0:
                        last_pos = non_pad_indices[-1].item()
                    else:
                        last_pos = tokens.shape[1] - 1  # Fallback to last position
                else:
                    # No padding token defined, use last position
                    last_pos = tokens.shape[1] - 1
                last_token_positions.append(last_pos)

            # Extract activations based on pooling method
            batch_acts = np.zeros((len(batch_texts), n_layers, n_heads, d_head))

            for layer in range(n_layers):
                act = cache[f"blocks.{layer}.attn.hook_z"]  # [batch, seq_len, n_heads, d_head]

                if pooling == 'last':
                    # Extract activation at the actual last token for each sample
                    for b in range(len(batch_texts)):
                        last_pos = last_token_positions[b]
                        batch_acts[b, layer] = act[b, last_pos, :, :].cpu().numpy()

                elif pooling == 'mean':
                    # Mean pool over all non-padding tokens
                    for b in range(len(batch_texts)):
                        if pad_token_id is not None:
                            # Create mask for non-padding tokens
                            non_pad_mask = tokens[b] != pad_token_id
                            # Get indices of non-padding tokens
                            valid_positions = torch.where(non_pad_mask)[0]
                            if len(valid_positions) > 0:
                                # Mean over valid positions
                                batch_acts[b, layer] = act[b, valid_positions, :, :].mean(dim=0).cpu().numpy()
                            else:
                                # Fallback: mean over all positions
                                batch_acts[b, layer] = act[b, :, :, :].mean(dim=0).cpu().numpy()
                        else:
                            # No padding: mean over all positions
                            batch_acts[b, layer] = act[b, :, :, :].mean(dim=0).cpu().numpy()

                else:
                    raise ValueError(f"Unknown pooling method: {pooling}")

            all_acts.append(batch_acts)

            del cache, logits, tokens
            torch.cuda.empty_cache()
            
    # Concatenate
    all_acts = np.concatenate(all_acts, axis=0) # [Total_Samples, Layers, Heads, Dim]
    labels = np.array(labels)
    
    return all_acts, labels

def train_probes(X_train, y_train, X_val, y_val, n_layers, n_heads):
    """
    Trains a Logistic Regression probe for each head.
    X_train: [Samples_train, Layers, Heads, Dim]
    y_train: [Samples_train]
    X_val: [Samples_val, Layers, Heads, Dim]
    y_val: [Samples_val]

    Returns:
        Dict {(layer, head): sklearn_model}
        Dict {(layer, head): train_accuracy}
        Dict {(layer, head): val_accuracy}
    """
    probes = {}
    train_accuracies = {}
    val_accuracies = {}

    total = n_layers * n_heads
    print(f"Training {total} probes...")

    for layer in tqdm(range(n_layers), desc="Layer Loop"):
        for head in range(n_heads):
            X_head_train = X_train[:, layer, head, :]
            X_head_val = X_val[:, layer, head, :]

            clf = LogisticRegression(max_iter=1000, solver='liblinear')
            clf.fit(X_head_train, y_train)

            train_acc = clf.score(X_head_train, y_train)
            val_acc = clf.score(X_head_val, y_val)

            probes[(layer, head)] = clf
            train_accuracies[(layer, head)] = train_acc
            val_accuracies[(layer, head)] = val_acc

    return probes, train_accuracies, val_accuracies

def save_probes(probes, filename="all_probes.pkl"):
    path = PROBES_DIR / filename
    with open(path, 'wb') as f:
        pickle.dump(probes, f)
    print(f"Saved probes to {path}")

def load_probes(filename="all_probes.pkl"):
    path = PROBES_DIR / filename
    with open(path, 'rb') as f:
        return pickle.load(f)
