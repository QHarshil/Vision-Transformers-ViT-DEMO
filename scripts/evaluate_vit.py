import os
import torch
from transformers import ViTForImageClassification
from preprocess import load_data

# Configuration
CHECKPOINT_PATH = "models/checkpoints/vit_epoch_2.pt"
BATCH_SIZE = 16
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

# CIFAR-10 test data
_, test_loader = load_data(batch_size=BATCH_SIZE)

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", 
    num_labels=10, 
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))  # Load model weights
model.eval()  # Set model to evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluation function
def evaluate():
    correct = 0
    total = 0

    with torch.no_grad():  # no need for gradients when evaluating because we're not changing weights of vectors.
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(pixel_values=images)
            _, predicted = torch.max(outputs.logits, 1)

            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
    return accuracy

# Save the accuracy to text file
def save_evaluation_results(accuracy):
    with open(os.path.join(SAVE_DIR, "evaluation_results.txt"), "w") as f:
        f.write(f"Test Set Accuracy: {accuracy:.2f}%\n")
    print("Evaluation results saved to results/evaluation_results.txt")

if __name__ == "__main__":
    accuracy = evaluate()
    save_evaluation_results(accuracy)
