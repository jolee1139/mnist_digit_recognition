import random
import matplotlib.pyplot as plt
import torch
from preprocess import get_mnist_datasets
from model import MNISTModel

# Load the test dataset
_, _, test_dataset = get_mnist_datasets()

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTModel().to(device)
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()

# Function to test the model with a random image
def test_random_image(dataset, model, device):
    # Pick a random index from the dataset
    idx = random.randint(0, len(dataset) - 1)
    image, label = dataset[idx]

    # Prepare the image for the model
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Output the results in the console
    print(f"True Label: {label}, Predicted Label: {predicted.item()}")

    # Display the image
    image_to_display = image.cpu().squeeze() * 0.5 + 0.5  # De-normalize and move to CPU
    plt.imshow(image_to_display, cmap="gray")
    plt.title(f"True: {label}, Predicted: {predicted.item()}")
    plt.axis("off")
    plt.show()

# Interactive loop for testing
def interactive_testing(dataset, model, device):
    while True:
        # Test a random image
        test_random_image(dataset, model, device)

        # Prompt the user
        user_input = input("Test another image? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Exiting interactive testing.")
            break

# Start the interactive testing loop
interactive_testing(test_dataset, model, device)
