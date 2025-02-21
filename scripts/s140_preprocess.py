import pandas as pd
from datasets import load_dataset
from transformers import DistilBertTokenizer

# Initialize tokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load dataset
print("Loading Sentiment140 dataset...")
sentiment140_dataset = load_dataset("sentiment140")

# Reduce dataset size (10K train/test)
sentiment140_train = sentiment140_dataset["train"].shuffle(seed=42).select(range(10000))
sentiment140_test = sentiment140_dataset["train"].shuffle(seed=42).select(range(10000))
sentiment140_dataset = {"train": sentiment140_train, "test": sentiment140_test}

# Get original text lengths
text_lengths = [len(text.split()) for text in sentiment140_dataset["train"]["text"]]

# Print statistics
import numpy as np
print(f"Sentence Lengths: Min={min(text_lengths)}, Max={max(text_lengths)}, Avg={np.mean(text_lengths):.2f}")
print(f"Percentiles (25%, 50%, 75%, 90%): {np.percentile(text_lengths, [25, 50, 75, 90])}")

# Set a fixed sequence length for fair comparison
MAX_LENGTH = int(np.percentile(text_lengths, 90))  # using 90th percentile as max length

# Preprocessing for BiGRU
def bigru_tokenize(batch):
    tokenized_texts = [distilbert_tokenizer.tokenize(text)[:MAX_LENGTH] for text in batch["text"]]  # Truncate
    token_ids = [distilbert_tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_texts]  # Convert to IDs
    return {"input_ids": token_ids}

# Preprocessing for DistilBERT
def distilbert_tokenize(batch):
    encodings = distilbert_tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    return {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]}

# Padding function for BiGRU
def pad_bigrus(batch):
    max_len = max(len(seq) for seq in batch["input_ids"])  # Use actual batch max length
    batch["input_ids"] = [seq + [0] * (max_len - len(seq)) for seq in batch["input_ids"]]
    return batch

# General preprocessing function
def preprocess_and_tokenize(dataset, tokenizer_function, is_bigrus=False):
    tokenized_data = dataset.map(tokenizer_function, batched=True)  # Handles batch input
    
    if is_bigrus:
        tokenized_data = tokenized_data.map(pad_bigrus, batched=True)  # Apply padding
    
    return tokenized_data

# Preprocess and save datasets
def preprocess_sentiment140():
    sentiment140_bigrus_train = preprocess_and_tokenize(sentiment140_dataset["train"], bigru_tokenize, is_bigrus=True)
    sentiment140_bigrus_test = preprocess_and_tokenize(sentiment140_dataset["test"], bigru_tokenize, is_bigrus=True)
    
    sentiment140_distilbert_train = preprocess_and_tokenize(sentiment140_dataset["train"], distilbert_tokenize, is_bigrus=False)
    sentiment140_distilbert_test = preprocess_and_tokenize(sentiment140_dataset["test"], distilbert_tokenize, is_bigrus=False)

    def save_tokenized_data(data, path):
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    # Save datasets
    save_tokenized_data(sentiment140_bigrus_train, "data/sentiment140/train_biGRU_tokenized.csv")
    save_tokenized_data(sentiment140_bigrus_test, "data/sentiment140/test_biGRU_tokenized.csv")
    save_tokenized_data(sentiment140_distilbert_train, "data/sentiment140/train_distilBERT_tokenized.csv")
    save_tokenized_data(sentiment140_distilbert_test, "data/sentiment140/test_distilBERT_tokenized.csv")

    print("Sentiment140 preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_sentiment140()