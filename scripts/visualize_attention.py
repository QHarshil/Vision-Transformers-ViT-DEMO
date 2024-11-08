import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTFeatureExtractor
from preprocess import load_data
from PIL import Image

CHECKPOINT_PATH = "models/checkpoints/vit_epoch_2.pt"
BATCH_SIZE = 1
SAVE_DIR = "results/visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", 
    num_labels=10, 
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

_, test_loader = load_data(batch_size=BATCH_SIZE)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

color_maps = ["jet", "viridis", "plasma", "inferno"]

# Function to save attention map overlay as an image file
def save_attention_map(image, attention_map, save_path="results/visualizations/attention_map.png"):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(attention_map, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Attention map saved to {save_path}")

# Function to visualize attention maps across different layers and color maps
def visualize_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass with attention outputs
        with torch.no_grad():
            outputs = model(pixel_values=images, output_attentions=True)
            attentions = outputs.attentions

        # Tensor to PIL image for display
        image = images.squeeze().permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

        layers_to_visualize = [0, len(attentions) // 2, -1]
        for layer_num, layer_idx in enumerate(layers_to_visualize):
            attention_map = attentions[layer_idx].mean(dim=1).squeeze().cpu().numpy()

            # Rescale attention map to input image size
            attention_map = np.mean(attention_map, axis=0)
            attention_map = np.resize(attention_map, (224, 224))

            # Visualize with different color maps
            for cmap in color_maps:
                plt.figure(figsize=(6, 6))
                plt.imshow(image)
                plt.imshow(attention_map, cmap=cmap, alpha=0.5)
                plt.axis("off")

                # Save attention map overlay
                save_path = os.path.join(SAVE_DIR, f"attention_map_layer{layer_num+1}_{cmap}_{idx+1}.png")
                plt.savefig(save_path, bbox_inches="tight")
                print(f"Saved {save_path}")

        if idx == 0:
            plt.show()

        #stop after 1 image
        break

if __name__ == "__main__":
    visualize_attention()
