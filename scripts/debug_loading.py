import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import MODEL_NAME
from src.model_utils import load_model
from src.data_loader import get_truthful_qa_data, get_counterfact_data

print(f"Testing Model Loading: {MODEL_NAME}")
try:
    model = load_model()
    print("Model loaded OK.")
except Exception as e:
    print(f"Model Load Failed: {e}")

print("Testing TruthfulQA Loading...")
try:
    df = get_truthful_qa_data()
    print(f"TruthfulQA loaded OK. Shape: {df.shape}")
except Exception as e:
    print(f"TruthfulQA Load Failed: {e}")

print("Testing CounterFact Loading...")
try:
    cf = get_counterfact_data()
    print(f"CounterFact loaded OK. Size: {len(cf)}")
except Exception as e:
    print(f"CounterFact Load Failed: {e}")
