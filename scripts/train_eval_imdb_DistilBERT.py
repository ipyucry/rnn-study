import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import subprocess
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler, autocast

# Load pre-trained tokenizer
TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Custom Dataset class
class TokenizedDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.input_ids = data["input_ids"].apply(eval).tolist()  # Convert string to list
        self.attention_mask = data["attention_mask"].apply(eval).tolist()  # Convert string to list again
        self.labels = data["labels"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Get GPU memory usage
def get_gpu_usage():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"])
        return int(output.decode("utf-8").strip().split("\n")[0])  # Returns GPU memory usage in MB
    except:
        return 0  # Return 0 if there's a problem

# Training function
def train(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask).logits
            predictions = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")  
    f1 = f1_score(y_true, y_pred, average="weighted")  

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy, precision, recall, f1

# Main function
def main():
    # Hyperparameters
    BATCH_SIZE, LEARNING_RATE, N_EPOCHS = 16, 2e-5, 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader = DataLoader(TokenizedDataset("data/imdb/train_distilBERT_tokenized.csv"), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TokenizedDataset("data/imdb/test_distilBERT_tokenized.csv"), batch_size=BATCH_SIZE)

    # Load pre-trained DistilBERT model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(DEVICE)
    optimizer, criterion, scaler = optim.AdamW(model.parameters(), lr=LEARNING_RATE), nn.CrossEntropyLoss(), GradScaler()

    # Training loop with per-epoch stats
    os.makedirs("results", exist_ok=True)
    results_file = "results/results_imdb_DistilBERT.csv"
    with open(results_file, "w") as f:
        f.write("Epoch,Loss,Accuracy,Precision,Recall,F1-score,Training Time (s),GPU Memory (MB)\n")

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE, scaler)
        accuracy, precision, recall, f1 = evaluate(model, test_loader, DEVICE)
        gpu_memory = get_gpu_usage()
        epoch_time = time.time() - start_time

        # Save results
        with open(results_file, "a") as f:
            f.write(f"{epoch+1},{train_loss:.4f},{accuracy:.4f},{precision:.4f},{recall:.4f},{f1:.4f},{epoch_time:.2f},{gpu_memory}\n")

        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Time={epoch_time:.2f}s, GPU={gpu_memory}MB")

    # Save model
    torch.save(model.state_dict(), "models/distilbert_imdb_model.pth")

if __name__ == "__main__":
    main()