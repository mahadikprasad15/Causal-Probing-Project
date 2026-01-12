import torch
import os
from transformer_lens import HookedTransformer
from src.config import MODEL_NAME, DEVICE, HF_TOKEN

def load_model(model_name=None, device=DEVICE, token=None):
    """
    Loads the HookedTransformer model.
    """
    # Use defaults if not provided
    if model_name is None: model_name = MODEL_NAME
    if token is None: token = HF_TOKEN

    print(f"Loading model: {model_name} on {device}")
    
    # Set token in env if provided, as transformer_lens often looks there or uses huggingface_hub
    if token:
        os.environ["HF_TOKEN"] = token
    
    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            dtype="float16" if device != "cpu" else "float32"
        )
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have access to the model and are logged in to Hugging Face or provided a token.")
        raise e

if __name__ == "__main__":
    model = load_model()
    print("Model loaded successfully.")
