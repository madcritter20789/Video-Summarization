import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import json
# ------------------------------
# Dataset for LSTM Training
# ------------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, summaries):
        self.embeddings = embeddings
        self.summaries = summaries

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.summaries[idx]


# ------------------------------
# LSTM Model for Summarization
# ------------------------------
class LSTMSummarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMSummarizer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last hidden state
        return out


# ------------------------------
# Training LSTM Model
# ------------------------------
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def train_lstm_model(combined_vector_path, summaries_path, model_save_path):
    # Load combined embeddings
    index = faiss.read_index(combined_vector_path)
    embeddings = np.array([index.reconstruct(i) for i in range(index.ntotal)])

    # Load ground-truth summaries (JSON format)
    with open(summaries_path, "r", encoding="utf-8") as f:
        summaries_data = json.load(f)

    # Extract only the "output" text from summaries
    summary_texts = [item["output"] for item in summaries_data]

    # Convert text summaries into numerical embeddings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    summary_embeddings = model.encode(summary_texts)  # Convert text to vectors

    # Convert to PyTorch tensors
    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(summary_embeddings, dtype=torch.float32)

    # Print shapes for debugging
    print(f"Input Shape (X): {X.shape}, Output Shape (y): {y.shape}")

    # Ensure dimensions match
    assert X.shape[0] == y.shape[0], "Mismatch: X and y must have the same number of samples"

    # (Continue with your DataLoader, Model training, etc.)


    # Dataset and DataLoader
    dataset = EmbeddingDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Model, loss, optimizer
    input_size = X.shape[1]
    output_size = y.shape[1]
    model = LSTMSummarizer(input_size=input_size, hidden_size=256, num_layers=2, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(50):  # Adjust the number of epochs as needed
        epoch_loss = 0
        for batch in dataloader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        print(f"Epoch [{epoch + 1}/50], Loss: {epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"LSTM model saved to {model_save_path}")


# ------------------------------
# Fine-tuning Transformer Model
# ------------------------------
class SummaryDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]



def fine_tune_transformer(combined_vector_path, summaries_path, model_save_path):
    # Check if the summaries file exists
    if not os.path.exists(summaries_path):
        raise FileNotFoundError(
            f"The summaries file '{summaries_path}' is missing. Please provide the file to proceed."
        )
    # Proceed with the rest of the function
    print(f"Using summaries file: {summaries_path}")

    # Load T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Load combined embeddings
    index = faiss.read_index(combined_vector_path)
    embeddings = np.array([index.reconstruct(i) for i in range(index.ntotal)])

    # Load summaries
    #summaries = np.load(summaries_path, allow_pickle=True)  # Assuming summaries are text
    # Load summaries from a JSON file
    with open(summaries_path, 'r') as file:
        summaries = json.load(file)

    print(f"Using summaries file: {summaries_path}")

    # Prepare dataset
    inputs = [tokenizer.encode(f"Summarize: {embedding}", return_tensors="pt", truncation=True, max_length=512) for
              embedding in embeddings]
    #targets = [tokenizer.encode(summary, return_tensors="pt", truncation=True, max_length=128) for summary in summaries]
    targets = [
        tokenizer.encode(summary['output'], return_tensors="pt", truncation=True, max_length=128)
        for summary in summaries
    ]
    dataset = SummaryDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Fine-tuning loop
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    for epoch in range(3):  # Adjust epochs as needed
        epoch_loss = 0
        for batch in dataloader:
            input_ids, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids.squeeze(1), labels=labels.squeeze(1))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    # Save the fine-tuned model
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Transformer model saved to {model_save_path}")


# ------------------------------
# Main Function
# ------------------------------
def train_model(combined_vector_path, summaries_path, lstm_model_path, transformer_model_path):
    print("Training LSTM model...")
    train_lstm_model(combined_vector_path, summaries_path, lstm_model_path)

    print("Fine-tuning Transformer model...")
    fine_tune_transformer(combined_vector_path, summaries_path, transformer_model_path)


if __name__ == "__main__":
    combined_vector_path = "embeddings/combined_vectors/sample.index"
    summaries_path = "results/summaries/ground_truth.npy"
    lstm_model_path = "models/lstm_model.pth"
    transformer_model_path = "models/transformer_model"

    train_model(combined_vector_path, summaries_path, lstm_model_path, transformer_model_path)
