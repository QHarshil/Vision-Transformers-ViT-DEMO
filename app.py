import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
import gradio as gr
from PIL import Image
import traceback
import torch.nn as nn

CHECKPOINT_PATH = "models/checkpoints/vit_epoch_2.pt"

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Update the model's classification head for 10 classes
model.classifier = nn.Linear(model.classifier.in_features, 10)

model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# CIFAR-10 labels
labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Prediction function
def predict(image):
    try:
        # Check if image is in RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize image for ViT
        image = image.resize((224, 224))

        # Preprocess the input image
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predicted class index
        predicted_class = outputs.logits.argmax(-1).item()
        
        print(f"Predicted class index: {predicted_class}")

        # Check if the predicted class index is within the labels range
        if predicted_class < len(labels):
            return labels[predicted_class]
        else:
            return "Unknown class index predicted"

    except Exception as e:
        error_message = traceback.format_exc()
        print("Error processing image:", error_message)
        return "Error processing image. Please try a different file."

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Vision Transformer Image Classifier",
    description="Upload an image and the Vision Transformer will classify it."
)

if __name__ == "__main__":
    interface.launch()
    share=True