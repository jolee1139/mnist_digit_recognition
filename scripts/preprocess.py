# scripts/preprocess.py
from torchvision import datasets, transforms
from torch.utils.data import random_split
from tqdm import tqdm  # Import tqdm for progress bar
import matplotlib.pyplot as plt
import os

# Define a custom download progress bar using TQDM
class TqdmDownload:
    def __init__(self):
        self.pbar = None

    def __call__(self, count, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading MNIST")
        self.pbar.update(count * block_size - self.pbar.n)

    def close(self):
        if self.pbar:
            self.pbar.close()

def get_mnist_datasets():
    # Define preprocessing transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Set up the download progress bar
    download_callback = TqdmDownload()

    # Check if the data folder exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print("Downloading the training dataset...")
    # Download the full training dataset
    full_train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    print("Training dataset downloaded successfully!")

    print("Splitting the training dataset into training and validation sets...")
    # Split into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")

    print("Downloading the test dataset...")
    # Download the test dataset
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    print("Test dataset downloaded successfully!")
    print(f"Test set size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

def visualize_mnist(dataset, num_images=10):
    """Visualize a few images from the dataset."""
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_images, figsize=(15, 2))
    for i in range(num_images):
        # Get an image and its label
        image, label = dataset[i]
        # De-normalize to [0, 1] for visualization
        image = image * 0.5 + 0.5
        # Squeeze the channel dimension
        image = image.squeeze()
        # Plot the image
        axes[i].imshow(image, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Label: {label}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test the function and show progress in the terminal
    train_dataset, val_dataset, test_dataset = get_mnist_datasets()
    print("Dataset preprocessing complete. Files are saved in the 'data/' folder.")
    
    # Visualize the first 10 images from the training dataset
    print("Visualizing the first 10 images in the training dataset...")
    visualize_mnist(train_dataset, num_images=10)
