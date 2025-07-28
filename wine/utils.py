import torch
from .model import WineNet
from .dataset import get_wine_dataloaders, get_full_dataset_for_analysis, get_feature_names
import pickle
import os


def load_model(model_path, scaler_path=None, device=None):
    """
    Load a trained wine model and its scaler

    Args:
        model_path: Path to the saved model file
        scaler_path: Path to the saved scaler (optional)
        device: Device to load on (auto-detected if None)

    Returns:
        tuple: (model, device, scaler, feature_names)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_names = get_feature_names()

    model = WineNet(input_size=len(feature_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load scaler if provided
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    return model, device, scaler, feature_names


def save_scaler(scaler, save_path):
    """Save the StandardScaler for later use"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)


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


def prepare_for_sensitivity_analysis(model_path=None, scaler_path=None):
    """
    Prepare model and data for sensitivity analysis

    Returns:
        tuple: (model, device, X_scaled, y, scaler, feature_names)
    """

    if model_path and os.path.exists(model_path):
        model, device, scaler, feature_names = load_model(model_path, scaler_path)

        # Scaler is required for proper model inference
        if scaler is None:
            if scaler_path:
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            else:
                raise ValueError("Model requires a scaler. Please provide scaler_path.")

        X, y, _, _ = get_full_dataset_for_analysis()
        X_scaled = scaler.fit_transform(X)

    else:
        # Train a new model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = WineNet().to(device)
        model.eval()
        # Get scaler from dataset
        X, y, scaler, feature_names = get_wine_dataloaders()
        X_scaled = scaler.transform(X)


    return model, device, X_scaled, y, scaler, feature_names
