import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_feature_names():
    """Get wine dataset feature names"""
    return [
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od280/od315_of_diluted_wines",
        "proline",
    ]


def get_full_dataset_for_analysis():
    """
    Get the complete wine dataset for sensitivity analysis

    Returns:
        tuple: (X, y, scaler, feature_names)
    """
    wine_data = load_wine()
    X, y = wine_data.data, wine_data.target
    feature_names = wine_data.feature_names
    scaler = StandardScaler()

    return X, y, scaler, feature_names


def get_wine_dataloaders(batch_size=32, test_size=0.2, random_state=42):
    """
    Get wine dataset train and test dataloaders

    Args:
        batch_size: Training batch size
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility

    Returns:
        tuple: (train_loader, test_loader, scaler, feature_names)
    """
    # Load wine dataset
    X, y, scaler, feature_names = get_full_dataset_for_analysis()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler, feature_names
