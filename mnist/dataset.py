from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


def get_mnist_transforms():
    """Get standard MNIST transforms"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])


def get_mnist_dataloaders(batch_size=64, test_batch_size=1000, data_dir='./data'):
    """
    Get MNIST train and test dataloaders

    Args:
        batch_size: Training batch size
        test_batch_size: Test batch size
        data_dir: Directory to store/load MNIST data

    Returns:
        tuple: (train_loader, test_loader)
    """
    transform = get_mnist_transforms()

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Load datasets
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def get_single_sample(dataset_type='test', index=0, data_dir='./data'):
    """Get a single sample for analysis"""
    transform = get_mnist_transforms()

    if dataset_type == 'test':
        dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    else:
        dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)

    return dataset[index]