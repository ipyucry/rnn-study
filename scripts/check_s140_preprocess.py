import pandas as pd
import numpy as np

# Define paths
BIGRU_TRAIN_PATH = "data/sentiment140/train_biGRU_tokenized.csv"
BIGRU_TEST_PATH = "data/sentiment140/test_biGRU_tokenized.csv"
DISTILBERT_TRAIN_PATH = "data/sentiment140/train_distilBERT_tokenized.csv"
DISTILBERT_TEST_PATH = "data/sentiment140/test_distilBERT_tokenized.csv"

# Load datasets
bigrus_train = pd.read_csv(BIGRU_TRAIN_PATH)
bigrus_test = pd.read_csv(BIGRU_TEST_PATH)
distilbert_train = pd.read_csv(DISTILBERT_TRAIN_PATH)
distilbert_test = pd.read_csv(DISTILBERT_TEST_PATH)

# Convert tokenized sequences from string to list
bigrus_train["input_ids"] = bigrus_train["input_ids"].apply(eval)
bigrus_test["input_ids"] = bigrus_test["input_ids"].apply(eval)
distilbert_train["input_ids"] = distilbert_train["input_ids"].apply(eval)
distilbert_test["input_ids"] = distilbert_test["input_ids"].apply(eval)

# Check dataset validity
def check_dataset(name, dataset):
    print(f"Checking {name} dataset...")
    num_samples = len(dataset)
    avg_length = np.mean([len(seq) for seq in dataset["input_ids"]])
    max_length = max([len(seq) for seq in dataset["input_ids"]])
    num_zero_padded = sum([seq.count(0) for seq in dataset["input_ids"]])

    print(f"  - Total samples: {num_samples}")
    print(f"  - Avg sequence length: {avg_length:.2f}")
    print(f"  - Max sequence length: {max_length}")
    print(f"  - Total padding tokens: {num_zero_padded}")
    
    if num_zero_padded / (num_samples * max_length) > 0.5:
        print("Warning: More than 50% of tokens are padding! Consider reducing max length.")

# Check datasets
check_dataset("BiGRU Train", bigrus_train)
check_dataset("BiGRU Test", bigrus_test)
check_dataset("DistilBERT Train", distilbert_train)
check_dataset("DistilBERT Test", distilbert_test)

# Check if dataset sizes match
if len(bigrus_train) != len(distilbert_train) or len(bigrus_test) != len(distilbert_test):
    print("Mismatch in dataset sizes between BiGRU and DistilBERT! Ensure fairness.")

print("Check complete.")