import torch
import torch.nn as nn
import torch.optim as optim
from .model import MNISTNet
import os


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, verbose=True):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if verbose and batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

    train_loss /= len(train_loader)
    accuracy = 100.0 * correct / len(train_loader.dataset)

    if verbose:
        print(
            f"Train set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)"
        )

    return train_loss, accuracy


def test_epoch(model, device, test_loader, criterion, verbose=True):
    """Test for one epoch"""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    if verbose:
        print(
            f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)"
        )

    return test_loss, accuracy


def train_model(
    num_epochs=5, batch_size=64, learning_rate=0.001, save_path=None, device=None
):
    """
    Complete training pipeline for MNIST

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_path: Path to save the trained model (optional)
        device: Device to train on (auto-detected if None)

    Returns:
        tuple: (trained_model, training_history)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on device: {device}")

    # Get data loaders
    train_loader, test_loader = get_mnist_dataloders(batch_size=batch_size)

    # Initialize model, optimizer, and loss function
    model = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # Training history
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        test_loss, test_acc = test_epoch(model, device, test_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print("-" * 50)

    print("Training completed!")

    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model, history
