import os
import torch
import argparse
from pathlib import Path

# Defaults
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_BASE_DIR = str(Path(__file__).parent.parent.resolve())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model Name")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace Token")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR, help="Base output directory")
    
    # We use parse_known_args to allow other flags if needed, or to avoid crashing if called from notebook
    args, _ = parser.parse_known_args()
    return args

# Parse immediately to set global constants (a bit hacky but convenient for this script structure)
args = parse_args()

MODEL_NAME = args.model
HF_TOKEN = args.token
BASE_DIR = Path(args.base_dir)

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
PROBES_DIR = BASE_DIR / "models" / "probes"
ACTIVATIONS_DIR = BASE_DIR / "activations"

# Ensure directories exist
for d in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, PROBES_DIR, ACTIVATIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42

BATCH_SIZE = 16

print(f"Config :: Model: {MODEL_NAME} | Device: {DEVICE} | Base Dir: {BASE_DIR}")
