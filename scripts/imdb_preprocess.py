import pandas as pd
from datasets import load_dataset
from transformers import DistilBertTokenizer

# Initialize tokenizers
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load dataset
print("Loading IMDb dataset...")
imdb_dataset = load_dataset("imdb")

# Preprocessing for BiGRU (word-level tokenization with truncation to max length 512)
def bigru_tokenize(text, max_length=512):
    if isinstance(text, list):  # Handle batched input
        return {"input_ids": [distilbert_tokenizer.tokenize(t)[:max_length] for t in text]}  # Truncate sequences
    else:  # Handle single input
        return {"input_ids": distilbert_tokenizer.tokenize(text)[:max_length]}  # Truncate sequence

# Preprocessing for DistilBERT (with max_length parameter for truncation)
def distilbert_tokenize(text, max_length=512):
    encodings = distilbert_tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"]
    }

# Dynamic padding for BiGRU
def dynamic_padding_bigrus(batch):
    max_length = max(len(seq) for seq in batch["input_ids"])
    batch["input_ids"] = [seq + [0] * (max_length - len(seq)) for seq in batch["input_ids"]]
    return batch

# Tokenization function for the entire dataset
def preprocess_and_tokenize(dataset, tokenizer_function, is_bigrus=False, max_length=512):
    # Apply tokenization
    tokenized_data = dataset.map(lambda x: tokenizer_function(x["text"], max_length=max_length), batched=True)
    
    # For BiGRU, convert tokens to IDs
    if is_bigrus:
        tokenized_data = tokenized_data.map(lambda x: {"input_ids": [distilbert_tokenizer.convert_tokens_to_ids(tokens) for tokens in x["input_ids"]]})
    
    # Apply dynamic padding for BiGRU
    if is_bigrus:
        tokenized_data = tokenized_data.map(dynamic_padding_bigrus, batched=True)
    
    return tokenized_data

# Preprocess and save IMDb datasets for both models
def preprocess_imdb():
    imdb_bigrus_train = preprocess_and_tokenize(imdb_dataset["train"], bigru_tokenize, is_bigrus=True, max_length=512)
    imdb_bigrus_test = preprocess_and_tokenize(imdb_dataset["test"], bigru_tokenize, is_bigrus=True, max_length=512)
    
    imdb_distilbert_train = preprocess_and_tokenize(imdb_dataset["train"], distilbert_tokenize, is_bigrus=False, max_length=512)
    imdb_distilbert_test = preprocess_and_tokenize(imdb_dataset["test"], distilbert_tokenize, is_bigrus=False, max_length=512)
    
    # Save tokenized files
    def save_tokenized_data(data, path, include_attention_mask=False):
        df = pd.DataFrame({
            "input_ids": data["input_ids"],
            "labels": data["label"]
        })
    
        if include_attention_mask:
            df["attention_mask"] = data["attention_mask"]

        df.to_csv(path, index=False)
    
    # Save IMDb datasets
    save_tokenized_data(imdb_bigrus_train, "data/imdb/train_biGRU_tokenized.csv")
    save_tokenized_data(imdb_bigrus_test, "data/imdb/test_biGRU_tokenized.csv")
    save_tokenized_data(imdb_distilbert_train, "data/imdb/train_distilBERT_tokenized.csv", include_attention_mask=True)
    save_tokenized_data(imdb_distilbert_test, "data/imdb/test_distilBERT_tokenized.csv", include_attention_mask=True)

    print("IMDb preprocessing complete and datasets saved.")

if __name__ == "__main__":
    preprocess_imdb()
