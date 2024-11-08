
# Vision Transformer (ViT) CIFAR-10 Classifier

This project demonstrates the use of a Vision Transformer (ViT) model fine-tuned on the CIFAR-10 dataset for image classification. ViTs work by transforming images into patches, processing them through transformer layers, and generating attention maps that reveal which parts of the image the model "focuses on" for classification.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture and Training](#model-architecture-and-training)
4. [Evaluation](#evaluation)
5. [Attention Map Visualization](#attention-map-visualization)
6. [How to Use](#how-to-use)
7. [Project Structure](#project-structure)

## Introduction

The Vision Transformer (ViT) is a deep learning model designed for image classification. Unlike traditional Convolutional Neural Networks (CNNs), ViT applies the transformer architecture, which has shown success in NLP, to images. The model divides images into small patches, processes each patch as a sequence, and generates class predictions based on contextual relationships between patches.

In this project, ViT is fine-tuned on the CIFAR-10 dataset, and attention maps are visualized to illustrate how the model focuses on different areas of the image.

## Dataset

**CIFAR-10** is a popular image dataset for machine learning research. It contains 60,000 images across 10 classes, with 6,000 images per class. Each image is 32x32 pixels. The classes in CIFAR-10 include:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

This project fine-tunes the ViT model on CIFAR-10, training it to classify images into one of these 10 classes.

## Model Architecture and Training

### Image Patching and Embedding

1. **Image Splitting**: Each 32x32 image is resized to 224x224 (the input size ViT expects) and divided into 16x16 patches, resulting in a 14x14 grid of patches.
2. **Embedding**: Each patch is then flattened into a 1-dimensional vector, which is passed through a linear embedding layer to convert it to a fixed size (e.g., 768).
3. **Positional Encoding**: Since the model processes patches as a sequence, positional embeddings are added to each patch to retain spatial information.

### Transformer Layers

The model uses multiple transformer layers, each consisting of:

- **Multi-Head Self-Attention (MHSA)**: Each patch attends to other patches to learn contextual information. Multiple attention heads capture different aspects of the relationships.
- **Feed-Forward Network (FFN)**: After self-attention, each patch is passed through a feed-forward network to introduce non-linearity and enhance feature extraction.

### Classification Head

After passing through transformer layers, a classification token is appended to the patch sequence. This token aggregates the contextual information across patches, and the model outputs class logits from this token.

### Training Process

- **Fine-Tuning**: The model is loaded with ImageNet pre-trained weights, and the classification head is replaced with a new head for 10 classes. The model is then fine-tuned on CIFAR-10.
- **Optimization**: Cross-entropy loss is minimized using AdamW, and the model runs for 8 epochs.

## Evaluation

The model's accuracy on the CIFAR-10 test set is evaluated after training. Accuracy is calculated by comparing the model’s predictions with the true class labels for each test image.

### Test Set Accuracy

The model achieved a test set accuracy of **98.51%**, as recorded in the `results/evaluation_results.txt`.

## Attention Map Visualization

Attention maps are visualized to understand which parts of the image the model focuses on for classification. This provides insight into the interpretability of the Vision Transformer.

### How Attention Maps are Created

1. **Extracting Attention Weights**: During inference, the model generates attention weights for each layer and head, showing how much each patch attends to every other patch.
2. **Averaging Attention Maps**: The attention maps are averaged across all attention heads in the last layer to provide a holistic view of the model’s focus.
3. **Overlaying on Image**: The attention map is resized to the original image dimensions and overlayed to highlight the regions that influence the model's classification.

### Interpreting Attention Maps

- **Colors**: The color intensity in attention maps indicates the level of focus. Darker colors represent areas of high attention, while lighter areas are less focused.
- **Examples**: The attention maps for 12 different layers are saved in `results/visualizations`. Each image highlights how attention changes across layers, showing different focal points.

## How to Use

### Requirements

Install dependencies by running:

```bash
pip install -r requirements.txt
```

### Training the Model

To fine-tune the model on CIFAR-10, run:

```bash
python scripts/train_vit.py
```

This script will save checkpoints in `models/checkpoints`.

### Evaluating the Model

To evaluate the model on the CIFAR-10 test set, run:

```bash
python scripts/evaluate_vit.py
```

The accuracy will be saved in `results/evaluation_results.txt`.

### Visualizing Attention Maps

To visualize the attention maps, run:

```bash
python scripts/visualize_attention.py
```

The attention maps will be saved in `results/visualizations`.

### Gradio App

To generate gradio app link, run:

```bash
python app.py
```
