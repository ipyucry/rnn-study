import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import subprocess  # For GPU memory tracking
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.cuda.amp import GradScaler, autocast

# Custom Dataset class
class TokenizedDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.input_ids = data["input_ids"].apply(eval).tolist()
        self.labels = data["labels"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# BiGRU Model
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(BiGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, hidden = self.gru(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden)

# Get GPU memory usage
def get_gpu_usage():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"])
        return int(output.decode("utf-8").strip().split("\n")[0])  # Returns GPU memory usage in MB
    except:
        return 0  # return 0 if there's a problem

# Training function
def train(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    epoch_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            predictions = model(inputs)
            loss = criterion(predictions, labels)
        
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
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs).argmax(dim=1)
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
    VOCAB_SIZE = 30522  # DistilBERT vocab size
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 128
    OUTPUT_DIM = 2
    N_LAYERS = 2
    DROPOUT = 0.5
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    N_EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader = DataLoader(TokenizedDataset("data/imdb/train_biGRU_tokenized.csv"), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TokenizedDataset("data/imdb/test_biGRU_tokenized.csv"), batch_size=BATCH_SIZE)

    # Initialize model, optimizer, loss function, and scaler
    model = BiGRU(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Training loop with per-epoch stats
    os.makedirs("results", exist_ok=True)
    results_file = "results/results_imdb_BiGRU.csv"
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
    torch.save(model.state_dict(), "models/bigru_imdb_model.pth")

if __name__ == "__main__":
    main()