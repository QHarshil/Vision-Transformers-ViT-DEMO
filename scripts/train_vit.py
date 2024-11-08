import os
import torch
from transformers import ViTForImageClassification, AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from preprocess import load_data

BATCH_SIZE = 16 # 16 images per batch
EPOCHS = 3
LEARNING_RATE = 2e-5
CHECKPOINT_DIR = "models/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Pre-trained ViT model with CIFAR-10 label configuration
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", 
    num_labels=10, 
    ignore_mismatched_sizes=True # Because CIFAR-10 has only 10 classes, but pre-trained model has 1000 for ImageNet
)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# Use NVIDIA GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)  # Confirmation of device
model.to(device)

# Load CIFAR-10 data from preprocess.py
train_loader, test_loader = load_data(batch_size=BATCH_SIZE)

# Training Loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)  # Calculate running average loss

            progress_bar.set_postfix(loss=avg_loss)

        # avg loss per epoch
        print(f"\nEpoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.4f}")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"vit_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
