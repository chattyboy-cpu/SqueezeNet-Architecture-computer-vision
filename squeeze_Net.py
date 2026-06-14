# SqueezeNet Implementation in PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision import models
from torchsummary import summary
import wandb
import os
import shutil
import kagglehub

# installing the wandb library
# !pip install wandb    

try:
    wandb.login()
except Exception as e:
    print("WandB login failed. Continuing in offline mode.")
    os.environ["WANDB_MODE"] = "offline"

wandb.init(project="SqueezeNet-flowers-recognition_new",
           config = {
               "learning_rate": 0.001,
                "epochs": 50,
                "batch_size": 16,
                "architecture": "SqueezeNet",
                "Pretrained": True,
                "input_size": 224
               })
# Shortcut to config values
config = wandb.config

final_path = r"C:\Users\lamin\Downloads\computer vision\flowers-recognition"
if not os.path.exists(final_path):
    print("Dataset not found locally, downloading from Kaggle...")
    path = kagglehub.dataset_download("alxmamaev/flowers-recognition")
    shutil.copytree(path, final_path)
    print(f"Dataset downloaded and copied to: {final_path}")
else:
    print(f"Dataset found at: {final_path}. Skipping download.")

data_dir = os.path.join(final_path, 'flowers')
if not os.path.exists(data_dir):
    data_dir = final_path

print(f"Data directory set to: {data_dir}")

# ===========================
# STEP 1: Data Preparation
# ===========================
# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize((config.input_size, config.input_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load entire dataset using ImageFolder
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

# Split into training (80%) and validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

print(f"Classes found: {full_dataset.classes}")
print(f"Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

# ===========================
# STEP 2: Load Pretrained Model
# ===========================
from torchvision.models import SqueezeNet1_1_Weights

model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)

# Change the final conv layer to match 5 flower classes
model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=5, kernel_size=1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only final conv layer
for param in model.classifier[1].parameters():
    param.requires_grad = True

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Watch the model's weights and gradients
wandb.watch(model, log="all", log_freq=10)

# ===================
# STEP 3: Loss & Optimizer
# ===================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        running_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            batch_correct = (preds == labels).sum().item()
            train_correct += batch_correct
            train_total += labels.size(0)
            
            # Print every 10 batches
            if (i + 1) % 10 == 0:
                batch_acc = batch_correct / labels.size(0)
                print(f"[Batch {i+1}/{len(train_loader)}] Loss: {loss.item():.4f}, Batch Acc: {batch_acc:.4f}")

        train_acc = train_correct / train_total
        wandb.log({"epoch": epoch + 1, "train_loss": running_loss, "train_accuracy": train_acc})
        print(f"Epoch {epoch+1} Summary - Loss: {running_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        wandb.log({"epoch": epoch + 1, "val_accuracy": val_acc})
        print(f"Validation Accuracy: {val_acc:.4f}")

# ===================
# Train the model
# ===================
train_model(model, criterion, optimizer, train_loader, val_loader, epochs=config.epochs)