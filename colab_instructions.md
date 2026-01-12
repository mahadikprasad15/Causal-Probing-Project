# Google Colab Run Instructions

## 1. Mount Google Drive
This allows you to save results, models, and activations persistently.
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Install Dependencies
Install the required libraries.
```bash
!pip install transformer_lens datasets scikit-learn pandas accelerate matplotlib seaborn torch
```

## 3. Clone Repository
Clone the code into the Colab runtime (fast execution).
```bash
%cd /content
!rm -rf Causal-Probing-Project # Clean up if exists
!git clone https://github.com/mahadikprasad15/Causal-Probing-Project.git
%cd Causal-Probing-Project
```

## 4. Run Experiment Pipeline
Replace `hf_YOUR_TOKEN` with HuggingFace token.
We use `--base_dir` to point to Google Drive. This ensures all huge files (activations) and results are saved to Drive, not the temporary Colab VM.

### Step 1: Prepare Data & Train Initial Probes
```bash
!python scripts/01_prepare_and_probe.py \
    --model "meta-llama/Llama-3.2-1B" \
    --token "hf_YOUR_TOKEN_HERE" \
    --base_dir "/content/drive/MyDrive/Causal_Probing_Experiment"
```

### Step 2: Find Causal Components (Activation Patching)
```bash
!python scripts/02_find_causal_components.py \
    --model "meta-llama/Llama-3.2-1B" \
    --token "hf_YOUR_TOKEN_HERE" \
    --base_dir "/content/drive/MyDrive/Causal_Probing_Experiment"
```

### Step 3: Run Comparison & Ensembling
```bash
!python scripts/03_run_comparison.py \
    --base_dir "/content/drive/MyDrive/Causal_Probing_Experiment"
```

### Step 4: Visualize Results
```bash
!python scripts/04_visualize_analysis.py \
    --base_dir "/content/drive/MyDrive/Causal_Probing_Experiment"
```

## 5. View Results
The plots and results will be saved in your Google Drive folder:
- `/content/drive/MyDrive/Causal_Probing_Experiment/plots/`
- `/content/drive/MyDrive/Causal_Probing_Experiment/results/`
