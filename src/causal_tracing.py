import torch
from transformer_lens import HookedTransformer
from transformer_lens import utils
from tqdm import tqdm
import numpy as np

def get_logit_diff(logits, answer_token_indices):
    """
    Calculates logit difference between correct and incorrect answers.
    answer_token_indices: Tensor of shape (batch, 2) where [:, 0] is correct, [:, 1] is incorrect.
    """
    if len(logits.shape) == 3:
        # Get the logits at the last token position
        last_token_logits = logits[:, -1, :]
    else:
        last_token_logits = logits
        
    correct_logits = last_token_logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = last_token_logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    
    return (correct_logits - incorrect_logits).mean()

def run_causal_tracing(model, dataset, batch_size=4):
    """
    Runs activation patching to find causal heads.
    
    dataset: list of dicts {'prompt', 'correct', 'incorrect'}
    
    Returns:
        Tensor of shape (n_layers, n_heads) with importance scores.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    # We need to tokenize the data and find answer tokens
    # For simplicity, let's assume single-token answers for now or take the first token of the answer.
    # Ideally, we should handle multi-token, but for 'TruthfulQA' generated multiple choice, usually single token diff is okay proxy.
    
    # Prepare clean (correct) and corrupted (incorrect) inputs? 
    # Actually, for "Truthfulness", we often patch:
    #   Corrupted Input: Prompt + [Incorrect Answer] -> wait, we want to see which HEADS compute the truth.
    #   Usually we have ONE prompt, but we want to see which heads distinguish True/False or build the representation.
    #   Standard Causal Tracing (ROME/Meng et al) corrupts the SUBJECT.
    #   Here, we have a binary classification setting.
    #   Let's define:
    #      Clean Run: The model processing the prompt.
    #      Metric: Logit Difference (Correct - Incorrect).
    #      Intervention: We want to know which heads *contribute* to this diff.
    #      Naive approach: Zero ablation (Resample Ablation is better).
    #      Let's use Mean-Ablation (patch in mean activations from the dataset) and see if Logit Diff drops.
    #      If Logit Diff drops significantly, the head is important.
    
    print("Computing mean activations for ablation...")
    # 1. Compute Mean Activations over the dataset (or a subset) to use as 'Corrupted' baseline
    # This is "Resample Ablation" but using the dataset mean as the resample source.
    
    # Actually, let's do a simpler "Head Ablation" sweep. 
    # For each head, ablate it (replace with zeroes or mean) and measure drop in Logit Diff.
    # Higher drop = Higher Importance.
    
    importances = torch.zeros((n_layers, n_heads))
    
    # Pre-compute baseline logit diff
    baseline_diffs = []
    
    # Prepare batches
    # We only need 'prompt' and the target tokens.
    
    prompts = [d['prompt'] for d in dataset]
    correct_strs = [d['correct'] for d in dataset]
    incorrect_strs = [d['incorrect'] for d in dataset]
    
    # Tokenize
    # We need single token targets for logit diff
    # We will pick the first token of the answer.
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Baseline Pass"):
        batch_prompts = prompts[i:i+batch_size]
        batch_correct = correct_strs[i:i+batch_size]
        batch_incorrect = incorrect_strs[i:i+batch_size]
        
        # Get answer tokens (First token of the answer)
        # Note: We prepend space as is common for continuation
        correct_tokens = [model.to_tokens(" " + c, prepend_bos=False)[0, 0] for c in batch_correct]
        incorrect_tokens = [model.to_tokens(" " + c, prepend_bos=False)[0, 0] for c in batch_incorrect]
        
        answer_indices = torch.tensor(list(zip(correct_tokens, incorrect_tokens)), device=model.cfg.device)
        
        logits = model(batch_prompts)
        diff = get_logit_diff(logits, answer_indices)
        baseline_diffs.append(diff.item())
        
    avg_baseline_diff = np.mean(baseline_diffs)
    print(f"Average Baseline Logit Diff: {avg_baseline_diff:.4f}")
    
    # 2. Iterative Ablation
    # We will use 'mean ablation' at the head output.
    # To do this efficiently, we can use model.run_with_hooks
    
    # For efficiency on 1B model, we might want to batch this differently or use only a subset of data for attribution.
    # Let's use a subset of 100 examples for finding causal components to save time.
    subset_size = min(len(dataset), 64)
    subset_indices = np.random.choice(len(dataset), subset_size, replace=False)
    
    sub_prompts = [prompts[j] for j in subset_indices]
    sub_correct = [correct_strs[j] for j in subset_indices]
    sub_incorrect = [incorrect_strs[j] for j in subset_indices]
    
    sub_correct_tokens = [model.to_tokens(" " + c, prepend_bos=False)[0, 0] for c in sub_correct]
    sub_incorrect_tokens = [model.to_tokens(" " + c, prepend_bos=False)[0, 0] for c in sub_incorrect]
    sub_answers = torch.tensor(list(zip(sub_correct_tokens, sub_incorrect_tokens)), device=model.cfg.device)

    # Cache mean activations (optional, or just zero-ablate). 
    # Zero-ablation is standard fast proxy.
    
    print(f"Running ablation on subset of {subset_size} examples...")
    
    for layer in tqdm(range(n_layers), desc="Layers"):
        for head in range(n_heads):
            
            def ablate_head_hook(value, hook):
                # value shape: [batch, pos, head_index, d_head]
                # We want to zero out the specific head
                value[:, :, head, :] = 0.0
                return value
            
            # Run with hook
            # Hook point: blocks.{layer}.attn.hook_z (output of heads before mixing) 
            # or hook_result. hook_z is standardized.
            
            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    sub_prompts,
                    fwd_hooks=[(utils.get_act_name("z", layer), ablate_head_hook)]
                )
            
            ablated_diff = get_logit_diff(ablated_logits, sub_answers).item()
            
            # Impact = Baseline - Ablated
            # If positive, the head was helping.
            # If negative, the head was harmful (suppressing truth).
            # We care about magnitude usually, or signed?
            # 'Causal Importance' usually implies positive contribution to the correct answer.
            
            importances[layer, head] = avg_baseline_diff - ablated_diff

    return importances
