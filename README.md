# Causal Probe Ensemble

Comparison of Standard Linear Probes vs Causal Ensembles for OOD Generalization.

## Setup

### Requirements
```bash
pip install transformer_lens datasets scikit-learn pandas accelerate matplotlib seaborn torch
```

### HuggingFace Access
For Llama-3.2 models, you need a HuggingFace token.
1. Accept license at [https://huggingface.co/meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B).
2. Get token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Usage

### Local
```bash
# Run the pipeline
python scripts/01_prepare_and_probe.py --token YOUR_HF_TOKEN
python scripts/02_find_causal_components.py --token YOUR_HF_TOKEN
python scripts/03_run_comparison.py

# Visualize
python scripts/04_visualize_analysis.py
```

### Google Colab
Mount Drive and point `base_dir` to this folder.

```python
# In Colab Cell
!git clone https://github.com/yourusername/causal_probing_project.git
%cd causal_probing_project
!pip install transformer_lens datasets scikit-learn pandas accelerate

# Run
!python scripts/01_prepare_and_probe.py --token "hf_..." --base_dir "/content/drive/MyDrive/causal_project"
!python scripts/02_find_causal_components.py --token "hf_..." --base_dir "/content/drive/MyDrive/causal_project"
!python scripts/03_run_comparison.py --base_dir "/content/drive/MyDrive/causal_project"
```

## Structure
- `src/`: Core logic (loading, probing, causal tracing).
- `scripts/`: Execution scripts (ordered 01-04).
- `data/`: Datasets.
- `results/`: Saved scores and metrics.
