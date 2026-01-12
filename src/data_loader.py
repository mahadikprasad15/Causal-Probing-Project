import json
import random
import pandas as pd
from datasets import load_dataset
from src.config import DATA_DIR, SEED

def get_truthful_qa_data():
    """
    Loads TruthfulQA generation subset.
    Returns a dataframe.
    """
    ds = load_dataset("truthful_qa", "generation", split="validation")
    df = ds.to_pandas()
    return df

def create_truthfulqa_splits(df):
    """
    Splits TruthfulQA by CATEGORY.
    
    ID Train/Test Categories: 
      - Law, Health, Economics, Politics, Psychology
    
    OOD Test Categories:
      - Superstitions, Religion, Myths and Fairytales, Weather, Other
    """
    
    # Check categories
    # The dataset has a 'category' column
    
    all_categories = df['category'].unique()
    # Manual selection for distribution shift
    id_categories = [
        'Law', 'Health', 'Economics', 'Politics', 'Psychology', 
        'Science', 'History', 'Sociology'
    ]
    ood_categories = [
        'Superstitions', 'Religion', 'Myths and Fairytales', 
        'Weather', 'Misconceptions', 'Proverbs', 'Mandela Effect', 
        'Confusion', 'Indexical Error: Identity', 'Indexical Effect: Identity', 
        'Indexical Error: Location', 'Indexical Error: Time'
    ]
    # Fallback: if category names don't match exactly, we might need to inspect them.
    # For now, let's split by simple inclusion.
    
    df_id = df[df['category'].isin(id_categories)]
    df_ood = df[~df['category'].isin(id_categories)] # Use everything else as OOD or strict subset?
    
    # Further split ID into Train and Test
    # Random shuffle
    df_id = df_id.sample(frac=1, random_state=SEED).reset_index(drop=True)
    split_idx = int(0.8 * len(df_id))
    
    train_df = df_id.iloc[:split_idx]
    test_id_df = df_id.iloc[split_idx:]
    
    print(f"Train Size: {len(train_df)}")
    print(f"Test ID Size: {len(test_id_df)}")
    print(f"Test OOD Size: {len(df_ood)}")
    
    return train_df, test_id_df, df_ood

def format_truthfulqa_for_probing(df):
    """
    Formats the dataframe into (prompt, correct_answer, incorrect_answer) triplets.
    We want to probe if the model 'knows' the truth.
    
    For binary probing on 'Truthfulness', we often look at the residual stream 
    at the last token of the question, or the first token of the answer.
    
    Here, let's format as:
    Prompt: "Q: [Question] A:"
    Target (True): [Best Correct Answer]
    Target (False): [Best Incorrect Answer]
    """
    samples = []
    for _, row in df.iterrows():
        question = row['question']
        # best_correct is usually index 0 in 'correct_answers' list? No, dataset structure varies.
        # In 'generation' subset: 
        # 'best_answer': string
        # 'correct_answers': list of strings
        # 'incorrect_answers': list of strings
        
        correct = row['best_answer']
        # Pick one random incorrect answer
        incorrect_list = row['incorrect_answers']
        if len(incorrect_list) == 0:
            continue
        incorrect = random.choice(incorrect_list)
        
        samples.append({
            "prompt": f"Q: {question}\nA:",
            "correct": correct,
            "incorrect": incorrect,
            "category": row['category']
        })
    return samples

def get_counterfact_data():
    """
    Loads CounterFact dataset for Far-OOD testing.
    """
    # CounterFact is large, we might just want a subset for testing
    try:
        ds = load_dataset("nehulag/counterfact", split="train[:1000]") # Subset
    except:
        print("Could not load CounterFact from HF, returning empty.")
        return []
        
    samples = []
    for item in ds:
        # Structure: prompt, target_true, target_false
        # Example: "The capital of France is" -> "Paris" (True), "Berlin" (False)
        samples.append({
            "prompt": item['prompt'],
            "correct": item['target_true'],
            "incorrect": item['target_false'],
            "category": "CounterFact"
        })
    return samples

if __name__ == "__main__":
    df = get_truthful_qa_data()
    train, test_id, test_ood = create_truthfulqa_splits(df)
    
    formatted_train = format_truthfulqa_for_probing(train)
    print(f"Sample formatted: {formatted_train[0]}")
