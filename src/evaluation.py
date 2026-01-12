import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import torch

def get_probe_predictions(probes, X, n_layers, n_heads):
    """
    Get probabilities from ALL probes on dataset X.
    Returns tensor: [Samples, Layers, Heads, 2] (prob for class 0 and 1) or just class 1.
    """
    n_samples = X.shape[0]
    # Prediction matrix: [Samples, Layers, Heads]
    probs = np.zeros((n_samples, n_layers, n_heads))
    
    # We Iterate keys
    for (layer, head), clf in probes.items():
        X_head = X[:, layer, head, :]
        p = clf.predict_proba(X_head)[:, 1] # Probability of True
        probs[:, layer, head] = p
        
    return probs

def evaluate_ensemble(probs_all, y, selected_indices, method='mean', weights=None):
    """
    Evaluates an ensemble of probes defined by selected_indices.
    
    probs_all: [Samples, Layers, Heads] - Precomputed probabilities
    y: [Samples] - Labels
    selected_indices: List of tuples [(layer, head), ...]
    method: 'mean', 'vote', 'weighted'
    weights: Dict {(layer, head): weight} for weighted mean.
    """
    
    # Extract relevant probs: [Samples, K]
    ensemble_probs = []
    w_list = []
    
    for (l, h) in selected_indices:
        ensemble_probs.append(probs_all[:, l, h])
        if weights:
            w_list.append(weights.get((l, h), 1.0))
            
    ensemble_probs = np.stack(ensemble_probs, axis=1) # [Samples, K]
    
    if method == 'mean':
        final_pred_probs = np.mean(ensemble_probs, axis=1)
        final_preds = (final_pred_probs > 0.5).astype(int)
        
    elif method == 'vote':
        # Hard vote
        votes = (ensemble_probs > 0.5).astype(int)
        vote_sum = np.sum(votes, axis=1)
        final_preds = (vote_sum > (len(selected_indices) / 2)).astype(int)
        final_pred_probs = vote_sum / len(selected_indices) # Pseudo-prob
        
    elif method == 'weighted':
        if not w_list:
            w_list = np.ones(len(selected_indices))
        w = np.array(w_list)
        w = w / np.sum(w) # Normalize
        
        # Weighted sum
        final_pred_probs = np.matmul(ensemble_probs, w)
        final_preds = (final_pred_probs > 0.5).astype(int)
        
    else:
        raise ValueError(f"Unknown method {method}")
        
    acc = accuracy_score(y, final_preds)
    try:
        auc = roc_auc_score(y, final_pred_probs)
    except:
        auc = 0.5
        
    return acc, auc

def select_top_k_probes(metrics_dict, k=5, metric='acc'):
    """
    Selects top K probes based on a metric dictionary {(l,h): score}.
    """
    sorted_items = sorted(metrics_dict.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_items[:k]]

