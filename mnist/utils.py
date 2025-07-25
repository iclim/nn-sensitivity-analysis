import torch
from .dataset import get_mnist_dataloaders, get_single_sample
from .model import MNISTNet
import matplotlib.pyplot as plt


def load_model(model_path, device=None):
    """
    Load a trained MNIST model

    Args:
        model_path: Path to the saved model file
        device: Device to load on (auto-detected if None)

    Returns:
        tuple: (model, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, device


def predict_single_image(model, device, image_tensor):
    """Make prediction on a single image"""
    model.eval()
    with torch.no_grad():
        if len(image_tensor.shape) == 3:  # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = torch.exp(output)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities.max().item()

        return predicted_class, confidence, probabilities


def evaluate_model(model, device, data_loader):
    """Evaluate model on a dataset"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            predicted = outputs.argmax(dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def visualize_sample_predictions(model, device, num_samples=5):
    """Visualize model predictions on random samples"""
    _, test_loader = get_mnist_dataloaders(batch_size=1)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        data, true_label = next(iter(test_loader))
        predicted_class, confidence, _ = predict_single_image(model, device, data)

        axes[i].imshow(data.squeeze().numpy(), cmap="gray")
        axes[i].set_title(
            f"True: {true_label.item()}\nPred: {predicted_class}\nConf: {confidence:.3f}"
        )
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def prepare_for_sensitivity_analysis(model_path=None, sample_index=0):
    """
    Prepare model and sample for sensitivity analysis

    Returns:
        tuple: (model, device, sample_image, sample_label)
    """
    if model_path:
        model, device = load_model(model_path)
    else:
        # Return a randomly initialized model for testing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MNISTNet().to(device)
        model.eval()

    sample_image, sample_label = get_single_sample("test", sample_index)

    # Enable gradient computation for sensitivity analysis
    sample_image.requires_grad_(True)

    return model, device, sample_image, sample_label
