import os
import random
import pandas as pd
from datasets import load_dataset

def main():
    # Ensure directories exist
    os.makedirs('data/outputs', exist_ok=True)
    
    print("Downloading dataset...")
    dataset = load_dataset("imdb")
    
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Save datasets
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved train dataset to {train_path}")
    print(f"Saved test dataset to {test_path}")
    


if __name__ == "__main__":
    main()
