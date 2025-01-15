import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # For progress tracking
from preprocess import get_mnist_datasets, visualize_mnist
from model import MNISTModel
import os

# Function to save a checkpoint
def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

# Function to load a checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from: {checkpoint_path}, starting from epoch {epoch + 1}")
    return epoch

# Load datasets
train_dataset, val_dataset, test_dataset = get_mnist_datasets()

# Visualize some images (optional)
visualize_mnist(train_dataset, num_images=10)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TensorBoard writer
writer = SummaryWriter(log_dir="logs/run1")

# Add model graph to TensorBoard
dummy_input = torch.randn(1, 1, 28, 28).to(device)
writer.add_graph(model, dummy_input)

# Checkpoint directory
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "mnist_checkpoint.pth")

# Training settings
epochs = 5
save_every = 500  # Save checkpoint every 500 iterations
start_epoch = 0

# Load checkpoint if available
if os.path.exists(checkpoint_path):
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

# Training loop
for epoch in range(start_epoch, epochs):
    model.train()
    total_train_loss, total_train_correct = 0, 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total_train_correct += (preds == labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())

        # Save checkpoint every `save_every` iterations
        if (i + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = total_train_correct / len(train_dataset)

    # Validation phase
    model.eval()
    total_val_loss, total_val_correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total_val_correct += (preds == labels).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = total_val_correct / len(val_dataset)

    # Log metrics to TensorBoard
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Save the final model
torch.save(model.state_dict(), "models/mnist_model.pth")
print("Final model saved!")

# Close TensorBoard writer
writer.close()