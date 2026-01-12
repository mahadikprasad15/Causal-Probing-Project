import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import os
from src.config import PROBES_DIR

def extract_activations_for_probing(model, dataset, batch_size=8):
    """
    Extracts activations for every head in the model.
    Data augmentation: Creates 2 examples per dataset item:
      1. Prompt + Correct Answer (Label 1)
      2. Prompt + Incorrect Answer (Label 0)
    
    Returns:
        activations: Dict {(layer, head): [N_samples, d_head]}
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
        
        # We need the activation at the LAST token.
        # cache['z'] has shape [batch, pos, n_heads, d_head]
        
        with torch.no_grad():
            # We want to catch 'blocks.{layer}.attn.hook_z'
            # To settle on a single cache object, we might run out of memory if we cache everything.
            # But we only need the LAST position.
            
            # We can use a hook to just grab the last token and store it.
            
            logits, cache = model.run_with_cache(
                batch_texts, 
                names_filter=lambda name: name.endswith("attn.hook_z"),
                return_type=None
            )
            
            # Process cache
            # Stack layers
            # We need to know the sequence lengths to pick the last token
            # Actually, if we padding is involved, strict last token might be padding.
            # TransformerLens handles padding if we use model.tokenizer correctly, but typically we just supply a list of strings and it handles it.
            # However, if strings are different lengths, left padding is standard for generation, but for acts we want the last REAL token.
            # TransformerLens `to_tokens` right pads by default? No, usually left.
            # Let's check tokenization.
            
            # Safest: Tokenize first, finding the index of the last token.
            tokens = model.to_tokens(batch_texts) # [batch, seq_len]
            # Since TL right-pads by default or handled internally? 
            # Actually TL usually assumes left padding for generation, but `to_tokens` usually pads right if asked?
            # Default `to_tokens` pads on the *left* if using a model that expects it?
            # Let's just use the index of the last non-pad token.
            
            # Actually, `run_with_cache` on a list of strings will pad.
            # We need to extract the correct index.
            
            # Simple hack: use batch_size=1 to avoid padding issues during extraction if speed permits.
            # Or just use the fact that `model.tokenizer.padding_side` matters.
            
            # Let's extract per layer
            batch_acts = np.zeros((len(batch_texts), n_layers, n_heads, d_head))
            
            for layer in range(n_layers):
                act = cache[f"blocks.{layer}.attn.hook_z"] # [batch, inputs_len, n_heads, d_head]
                
                # Extract last token. 
                # If padding is on the right, we need `eos_token`?
                # If padding is on the left, we take the last index.
                # Llama tokenizer typically no padding token by default?
                # Let's assume the last index in the tensor corresponds to the end of the text.
                # If they are padded, we might grab a pad token.
                
                # Check tokenizer config
                # For safety in this script, let's just create a tensor of (batch, n_layers, n_heads, d_head) from the last position
                # Assuming simple right-alignment or equal length? No.
                
                # Let's rely on finding the last attention mask or just batch_size=1 for safety if memory is concern/padding complex.
                # But batch_1 is slow.
                
                # Correct way with TL:
                # `batch_tokens = model.to_tokens(batch_texts)`
                # Find length of each.
                
                # Let's stick to taking the `-1` index, assuming the model handles the passed strings without excessive padding artifacts if we don't manually pad.
                # When passing list of str, TL tokenizes and pads.
                # For Llama, padding is usually LEFT?
                # If LEFT padding, the last token is indeed the last token.
                
                batch_acts[:, layer] = act[:, -1, :, :].cpu().numpy()
            
            all_acts.append(batch_acts)
            
            del cache, logits
            torch.cuda.empty_cache()
            
    # Concatenate
    all_acts = np.concatenate(all_acts, axis=0) # [Total_Samples, Layers, Heads, Dim]
    labels = np.array(labels)
    
    return all_acts, labels

def train_probes(X, y, n_layers, n_heads):
    """
    Trains a Logistic Regression probe for each head.
    X: [Samples, Layers, Heads, Dim]
    y: [Samples]
    
    Returns:
        Dict {(layer, head): sklearn_model}
        Dict {(layer, head): accuracy}
    """
    probes = {}
    accuracies = {}
    
    # Validation split inside training? Or we passed Train data only?
    # We should assume X, y are TRAIN data.
    # We can do an internal CV or just train on all.
    # User plan implies: "Train on ID_Train".
    
    total = n_layers * n_heads
    print(f"Training {total} probes...")
    
    for layer in tqdm(range(n_layers), desc="Layer Loop"):
        for head in range(n_heads):
            X_head = X[:, layer, head, :]
            
            clf = LogisticRegression(max_iter=1000, solver='liblinear') # Liblinear is fast for small datasets
            clf.fit(X_head, y)
            
            acc = clf.score(X_head, y) # Training accuracy
            
            probes[(layer, head)] = clf
            accuracies[(layer, head)] = acc
            
    return probes, accuracies

def save_probes(probes, filename="all_probes.pkl"):
    path = PROBES_DIR / filename
    with open(path, 'wb') as f:
        pickle.dump(probes, f)
    print(f"Saved probes to {path}")

def load_probes(filename="all_probes.pkl"):
    path = PROBES_DIR / filename
    with open(path, 'rb') as f:
        return pickle.load(f)
