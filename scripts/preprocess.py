import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Create the processed data directory if it doesn’t exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),                 # Resize to 224x224 for ViT
    transforms.ToTensor(),                         # turn into tensor 1D vector
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Download and apply transformations to CIFAR-10 dataset
def load_data(batch_size=16):
    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, transform=transform, download=True)

    # DataLoader for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = load_data()
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
