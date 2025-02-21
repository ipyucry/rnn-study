import pandas as pd

# Load tokenized files
def load_tokenized_data(file_path):
    return pd.read_csv(file_path)

# Check if tokenized files are valid
def check_tokenized_data(file_path, model_type="BiGRU"):
    print(f"Checking tokenized dataset: {file_path}")
    
    # Load file
    df = load_tokenized_data(file_path)
    
    # Check if necessary columns are present
    assert 'input_ids' in df.columns, "'input_ids' column is missing"
    assert 'labels' in df.columns, "'labels' column is missing"
    
    # Check basic data structure
    print(f"Number of samples: {len(df)}")
    print(f"First few rows:\n{df.head()}")

    # Convert 'input_ids' from string to list of integers
    df['input_ids'] = df['input_ids'].apply(eval)  # converts string representation of list back to list
    
    # Check for empty sequences
    empty_sequences = df[df['input_ids'].apply(len) == 0]
    print(f"Number of empty sequences: {len(empty_sequences)}")

    # Check sequence lengths
    max_length = df['input_ids'].apply(len).max()
    min_length = df['input_ids'].apply(len).min()
    
    print(f"Max sequence length: {max_length}")
    print(f"Min sequence length: {min_length}")

    # For BiGRU, check if padding is correct (dynamic padding)
    if model_type == "BiGRU":
        avg_length = df['input_ids'].apply(len).mean()
        print(f"Average sequence length (BiGRU): {avg_length:.2f}")
        lengths = df['input_ids'].apply(len)
        
        if len(set(lengths)) == 1:
            print("Sequences are uniformly padded to the same length.")
        else:
            print(f"Sequences vary in length, but average is {avg_length:.2f}")
    
    # For DistilBERT, check if any sequence exceeds the max token limit (512)
    if model_type == "DistilBERT":
        max_distilbert_length = 512
        long_sequences = df[df['input_ids'].apply(len) > max_distilbert_length]
        print(f"Number of sequences longer than {max_distilbert_length} tokens: {len(long_sequences)}")

    # Check if labels are balanced
    label_counts = df['labels'].value_counts()
    print(f"Label distribution:\n{label_counts}")

    print("Check complete.")

check_tokenized_data("data/imdb/train_biGRU_tokenized.csv", model_type="BiGRU")

check_tokenized_data("data/imdb/train_distilBERT_tokenized.csv", model_type="DistilBERT")