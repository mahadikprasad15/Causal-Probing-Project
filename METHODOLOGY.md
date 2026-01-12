# Methodological Choices and Limitations

This document explains the key experimental design decisions, methodological choices, and limitations of the causal probing project.

## Table of Contents
1. [Activation Extraction](#activation-extraction)
2. [Probe Training](#probe-training)
3. [Causal Discovery](#causal-discovery)
4. [Ensemble Evaluation](#ensemble-evaluation)
5. [Known Limitations](#known-limitations)
6. [Future Extensions](#future-extensions)

---

## Activation Extraction

### Token Position Selection

**Current Implementation:**
- **Default**: Extracts activations at the **last non-padding token** of the input sequence
- **Alternative**: `--pooling mean` extracts **mean-pooled** activations over all non-padding tokens

**Rationale:**
- The last token position is where the model has processed the entire question
- This is where "truthfulness" representations should be most concentrated
- This is standard practice for sequence classification tasks

**How it works:**
```python
# Find last non-padding token
non_pad_mask = tokens[b] != pad_token_id
last_pos = torch.where(non_pad_mask)[0][-1]
activation = cache[layer][b, last_pos, :, :]
```

**Alternatives to consider:**
- First token of the answer (requires generating)
- Max-pooling over sequence
- Attention-weighted pooling
- Position-specific extraction (e.g., subject tokens only)

### Attention Head Outputs (hook_z)

**What we extract:**
- `hook_z`: Output of attention heads **before** mixing via $W_O$ projection
- Shape: `[batch, seq_len, n_heads, d_head]`

**Why not residual stream?**
- We want to probe **individual attention heads**, not the full residual stream
- This allows us to attribute truthfulness to specific heads
- Residual stream contains information from all layers, making attribution unclear

**Why not MLP activations?**
- Current focus is on attention mechanisms
- MLPs can be added in future work

### Data Augmentation

**Binary Classification Setup:**
Each dataset item is converted into **2 examples**:
1. `prompt + correct_answer` → Label 1 (truthful)
2. `prompt + incorrect_answer` → Label 0 (untruthful)

**Benefits:**
- Creates balanced dataset (50/50 class distribution)
- Standard approach in probing literature
- Tests if model can distinguish correct vs incorrect continuations

**Limitation:**
- Doubles dataset size (memory overhead)
- Assumes independence between correct/incorrect versions

---

## Probe Training

### Linear Probes (Logistic Regression)

**Current Implementation:**
```python
LogisticRegression(max_iter=1000, solver='liblinear')
```

**Why linear probes?**
- Tests if representation is **linearly separable**
- Standard in interpretability literature (Alain & Bengio 2017)
- Fast to train (one per head)
- If linear probe works, information is "easily accessible"

**Hyperparameters:**
- **Solver**: `liblinear` (fast for small datasets)
- **Max iterations**: 1000 (usually converges earlier)
- **Regularization**: Default `C=1.0` (no tuning currently)

**Train/Validation Split:**
- **80% train / 20% validation** split from training data
- Probes trained on train split only
- Validation accuracy used for probe selection (avoids overfitting)

**Limitations:**
- No regularization tuning (could improve generalization)
- No cross-validation (single split might be unstable)
- Linear assumption (nonlinear probes might find more information)

**Future Extensions:**
- Sweep over regularization strength `C`
- Compare L1 vs L2 regularization (sparse vs dense probes)
- Try MLP probes (test nonlinearity)
- K-fold cross-validation

---

## Causal Discovery

### Ablation Method: Zero Ablation

**Current Implementation:**
```python
def ablate_head_hook(value, hook):
    value[:, :, head, :] = 0.0  # Set head output to zero
    return value
```

**What is measured:**
```
Causal Importance = Baseline Logit Diff - Ablated Logit Diff
```

**Interpretation:**
- **Positive score**: Head helps the model prefer the correct answer
- **Negative score**: Head suppresses the correct answer (might encode falsehoods)
- **Zero**: Head is irrelevant for this task

**Why zero ablation?**
- Simple and fast
- Standard baseline in mechanistic interpretability

**⚠️ Limitation: Out-of-Distribution Activations**
Zero is **not in the natural activation distribution**. This can create artifacts:
- Model might never see zero activations during training
- Could trigger unexpected behaviors
- May overestimate importance

**Better alternatives:**
1. **Mean Ablation**: Replace with mean activation across dataset
2. **Resample Ablation**: Replace with activation from another example (ROME paper)
3. **Activation Patching**: Patch from corrupted run into clean run

### Single-Token Logit Difference

**Current Implementation:**
```python
# Extract first token of each answer
correct_token = model.to_tokens(" " + correct)[0, 0]
incorrect_token = model.to_tokens(" " + incorrect)[0, 0]

# Measure logit difference at last prompt position
logit_diff = logits[-1, correct_token] - logits[-1, incorrect_token]
```

**What this measures:**
- At the end of the prompt, does the model assign higher probability to the first token of the correct answer vs the first token of the incorrect answer?

**Example:**
- Prompt: "Q: What is the capital of France?\nA:"
- Correct: "Paris" → tokenized as `[" Paris"]` or `[" Par", "is"]`
- Incorrect: "London" → tokenized as `[" London"]` or `[" Lon", "don"]`
- Measure: `logit[" Par"] - logit[" Lon"]` or `logit[" Paris"] - logit[" London"]`

**✅ When this works well:**
- Answers are single tokens
- Answers have distinct first tokens ("Paris" vs "London")
- Logit difference at first token is predictive of full answer quality

**❌ Limitations:**

1. **Multi-token answers**: Only first token matters
   - "New York" → Only " New" is measured
   - "Machine learning" → Only " Machine" is measured
   - Rest of answer ignored

2. **Same-prefix answers**: Breaks completely
   - "California" vs "Canada" → Both start with " C"
   - Would measure `logit[" C"] - logit[" C"] = 0`

3. **Tokenization variability**:
   - "Paris" might be `[" Paris"]` (single token) or `[" Par", "is"]` (two tokens)
   - Depends on tokenizer vocabulary
   - Only first token considered

**Better alternatives:**
1. **Full sequence log-probability**:
   ```python
   log_prob_correct = sum(log P(token_i | prompt + tokens[:i]))
   log_prob_incorrect = sum(log P(token_i | prompt + tokens[:i]))
   metric = log_prob_correct - log_prob_incorrect
   ```

2. **Perplexity ratio**:
   ```python
   perplexity_diff = perplexity(incorrect) / perplexity(correct)
   ```

3. **Exact match after generation**:
   ```python
   generated = model.generate(prompt, max_tokens=10)
   metric = 1 if generated == correct else 0
   ```

### Sample Size for Causal Tracing

**Current Implementation:**
- Uses **64 random samples** from training set (line 102 in causal_tracing.py)
- Rationale: Speed vs accuracy tradeoff

**⚠️ Limitation: High Variance**
- 64 samples might not be representative
- Causal scores could vary significantly with different random seeds
- Some heads might be important but not captured in this subset

**Recommendation:**
- Use larger subset (e.g., 200-500 samples) if memory allows
- Or use full training set for more stable estimates

---

## Ensemble Evaluation

### Three Strategies Compared

**A. Baseline: Best Single Probe**
- Select probe with highest **validation accuracy**
- Represents best individual head for truthfulness
- No ensemble effects

**B. Causal Ensemble**
- Select top-K heads by **causal importance score**
- Aggregate with **equal weights** (mean or vote)
- Tests hypothesis: "Causally important heads generalize better"

**C. Accuracy Ensemble**
- Select top-K heads by **validation accuracy**
- Aggregate with **equal weights** (mean or vote)
- Control condition: "Does causal selection beat accuracy selection?"

### Aggregation Methods

**1. Mean (Equal Weights)**
```python
final_prob = mean(probe_probs)  # Average probabilities
prediction = final_prob > 0.5
```
- **Pro**: Simple, well-calibrated
- **Con**: Gives equal weight to all probes (even mediocre ones)

**2. Vote (Majority Voting)**
```python
votes = (probe_probs > 0.5).astype(int)
prediction = sum(votes) > K/2
```
- **Pro**: Robust to outliers
- **Con**: Loses calibration, threshold-dependent

### Why NOT Weighted by Causal Scores?

**Previous implementation (removed):**
```python
weights = relu(causal_scores)  # Use causal scores as weights
final_prob = weighted_mean(probe_probs, weights)
```

**Problem: Double-dipping**
- Using causal scores **both** for selection **and** weighting
- Unfair comparison with accuracy ensemble (which uses equal weights)
- Confounds selection effect with weighting effect

**Current implementation:**
- All ensembles use **equal weights**
- Isolates effect of **selection criterion** (causal vs accuracy)
- Fair comparison

### Future: Learned Ensemble Weights

**Ideas:**
1. **Stacking**: Train meta-classifier on probe outputs
   ```python
   meta_clf = LogisticRegression()
   meta_clf.fit(probe_outputs_val, labels_val)
   ```

2. **Attention-weighted ensemble**:
   ```python
   attention_weights = softmax(learnable_params @ probe_features)
   final_prob = sum(attention_weights * probe_probs)
   ```

3. **Bayesian Model Averaging**: Weight by posterior probability

---

## Known Limitations

### 1. Padding Handling (✅ Fixed)
- **Issue**: Previously used last token blindly, might grab padding
- **Fix**: Now finds actual last non-padding token using attention mask

### 2. Data Leakage (✅ Fixed)
- **Issue**: Previously selected baseline using test accuracy
- **Fix**: Now uses validation accuracy for all selections

### 3. Single-Token Logit Diff (⚠️ Documented)
- **Issue**: Only measures first token of answers
- **Status**: Acceptable proxy for most TruthfulQA answers, but documented
- **Future**: Implement full sequence log-probability

### 4. Zero Ablation (⚠️ Documented)
- **Issue**: Out-of-distribution activation values
- **Status**: Standard baseline, but mean ablation would be better
- **Future**: Implement mean or resample ablation

### 5. No Regularization Tuning
- **Issue**: Using default `C=1.0` for all probes
- **Status**: Works reasonably, but might not be optimal
- **Future**: Grid search or nested CV for `C`

### 6. Small Causal Subset (⚠️ Documented)
- **Issue**: Only 64 samples for causal tracing
- **Status**: Fast but potentially unstable
- **Future**: Use larger subset or full training set

---

## Future Extensions

### Aggregation Methods
- [ ] Max-voting (most confident probe)
- [ ] Stacking (meta-classifier)
- [ ] Attention-weighted ensemble with learned weights
- [ ] Bayesian model averaging

### Probe Architectures
- [ ] MLP probes (test nonlinearity)
- [ ] Sparse probes (L1 regularization)
- [ ] Probes at different positions (first token, all tokens)
- [ ] Regularization hyperparameter sweep

### Token Positions
- [ ] First answer token (instead of last question token)
- [ ] Max-pooling over sequence
- [ ] Attention-weighted pooling
- [ ] Position-specific (e.g., subject entity only)

### Ablation Methods
- [ ] Mean ablation
- [ ] Resample ablation (ROME-style)
- [ ] Activation patching (path patching)
- [ ] Integrated gradients

### Causal Discovery
- [ ] Larger subset for causal tracing (200-500 samples)
- [ ] Full sequence log-probability (not just first token)
- [ ] Direct Logit Attribution (DLA)
- [ ] Attention pattern analysis

### Datasets
- [ ] Natural Questions (different domain)
- [ ] SciQ (science-specific)
- [ ] Other factual QA datasets

---

## References

- **Probing**: Alain & Bengio (2017), "Understanding intermediate layers using linear classifier probes"
- **Causal Tracing**: Meng et al. (2022), "Locating and Editing Factual Associations in GPT"
- **Mechanistic Interpretability**: Elhage et al. (2021), "A Mathematical Framework for Transformer Circuits"

---

## Running Experiments with Different Pooling

```bash
# Default: Last token
python scripts/01_prepare_and_probe.py --pooling last

# Mean pooling over all tokens
python scripts/01_prepare_and_probe.py --pooling mean
```

This will save activations and probes with different suffixes:
- `X_train.npy` (last token, default)
- `X_train_mean.npy` (mean pooling)
- `probes_logistic.pkl` (last token)
- `probes_logistic_mean.pkl` (mean pooling)
