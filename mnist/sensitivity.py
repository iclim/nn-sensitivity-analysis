import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from .utils import prepare_for_sensitivity_analysis
from .train import train_model


def compute_gradients(model, device, input_tensor, target_class=None):
    """
    Compute gradients of model output with respect to input pixels

    Args:
        model: Trained neural network
        device: Device (CPU/GPU)
        input_tensor: Input image with requires_grad=True
        target_class: Specific class to compute gradients for (if None, uses predicted class)

    Returns:
        tuple: (gradients, predicted_class, confidence)
    """
    model.eval()

    # Ensure we have the right shape and create a proper leaf tensor
    if len(input_tensor.shape) == 3:  # Single image [C, H, W]
        # Add batch dimension first, then clone to make it a leaf
        input_with_batch = input_tensor.unsqueeze(0)  # [1, C, H, W]
        input_leaf = input_with_batch.clone().detach().to(device)
    else:  # Already has batch dimension
        input_leaf = input_tensor.clone().detach().to(device)

    # Make it require gradients - this must be a leaf tensor
    input_leaf.requires_grad_(True)

    # Forward pass
    output = model(input_leaf)
    probabilities = torch.exp(output)

    # Choose which output to compute gradients for
    if target_class is None:
        # Use the predicted class
        target_class = output.argmax(dim=1).item()

    # Get the score for the target class
    target_score = output[0, target_class]
    confidence = probabilities[0, target_class].item()

    # Compute gradients
    target_score.backward()

    # Extract gradients - input_leaf.grad should definitely exist now
    if input_leaf.grad is not None:
        gradients = input_leaf.grad.clone()
    else:
        raise RuntimeError(
            "Gradients not computed. This shouldn't happen with a leaf tensor."
        )

    return gradients, target_class, confidence


def analyze_gradient_statistics(gradients, top_k=10):
    """
    Analyze statistical properties of gradients for MNIST

    Args:
        gradients: Gradient tensor (28x28 for MNIST)
        top_k: Number of top sensitive pixels to report

    Returns:
        dict: Statistics about the gradients
    """
    gradients_flat = gradients.flatten()
    abs_gradients_flat = torch.abs(gradients_flat)

    stats = {
        "mean_magnitude": abs_gradients_flat.mean().item(),
        "std_magnitude": abs_gradients_flat.std().item(),
        "max_magnitude": abs_gradients_flat.max().item(),
        "min_magnitude": abs_gradients_flat.min().item(),
        "total_pixels": gradients_flat.numel(),
        "top_pixels": [],
    }

    # Find most sensitive pixels
    _, top_indices = torch.topk(abs_gradients_flat, top_k)

    if len(gradients.shape) >= 2:  # 2D image (possibly with batch/channel dims)
        # Get the actual 2D dimensions
        img_shape = gradients.shape[-2:]  # Last 2 dimensions should be height, width
        h, w = img_shape

        for i, idx in enumerate(top_indices):
            row, col = (idx // w).item(), (idx % w).item()
            gradient_val = gradients_flat[idx].item()
            stats["top_pixels"].append(
                {
                    "rank": i + 1,
                    "position": (row, col),
                    "gradient": gradient_val,
                    "magnitude": abs(gradient_val),
                }
            )

    return stats


def visualize_sensitivity(
    original_image,
    gradients,
    predicted_class,
    true_class,
    confidence,
    save_path=None,
    title_suffix="",
):
    """
    Create comprehensive visualization of MNIST sensitivity analysis

    Args:
        original_image: Original 28x28 image tensor
        gradients: Gradient tensor of same shape
        predicted_class: Model's prediction
        true_class: Ground truth label
        confidence: Model confidence for predicted class
        save_path: Optional path to save the figure
        title_suffix: Additional text for the title

    Returns:
        matplotlib.figure.Figure: The created figure
    """

    # Convert to numpy and ensure proper 2D shape for MNIST
    def to_2d_numpy(tensor):
        """Convert tensor to 2D numpy array for MNIST (28x28)"""
        # Remove batch and channel dimensions, keep only spatial dimensions
        while len(tensor.shape) > 2:
            tensor = tensor.squeeze()
        return tensor.detach().cpu().numpy()

    original_np = to_2d_numpy(original_image)
    gradients_np = to_2d_numpy(gradients)

    # Verify we have the right shape for MNIST
    if original_np.shape != (28, 28) or gradients_np.shape != (28, 28):
        print(
            f"Warning: Unexpected shape. Original: {original_np.shape}, Gradients: {gradients_np.shape}"
        )
        # If shapes don't match, try to reshape
        if original_np.size == 784:
            original_np = original_np.reshape(28, 28)
        if gradients_np.size == 784:
            gradients_np = gradients_np.reshape(28, 28)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(original_np, cmap="gray")
    axes[0, 0].set_title(
        f"Original Image\nTrue: {true_class}, Pred: {predicted_class}\nConfidence: {confidence:.3f}"
    )
    axes[0, 0].axis("off")

    # Raw gradients
    grad_max = max(abs(gradients_np.min()), abs(gradients_np.max()))
    if grad_max > 0:  # Avoid division by zero
        im1 = axes[0, 1].imshow(
            gradients_np, cmap="RdBu_r", vmin=-grad_max, vmax=grad_max
        )
        plt.colorbar(im1, ax=axes[0, 1], shrink=0.6)
    else:
        im1 = axes[0, 1].imshow(gradients_np, cmap="RdBu_r")
    axes[0, 1].set_title("Raw Gradients\n(Red=Positive, Blue=Negative)")
    axes[0, 1].axis("off")

    # Absolute gradients (sensitivity magnitude)
    abs_gradients = np.abs(gradients_np)
    im2 = axes[0, 2].imshow(abs_gradients, cmap="hot")
    axes[0, 2].set_title("Gradient Magnitude\n(Sensitivity Strength)")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.6)

    # Overlay on original
    axes[1, 0].imshow(original_np, cmap="gray", alpha=0.7)
    if abs_gradients.max() > abs_gradients.min():  # Avoid division by zero
        norm_gradients = (abs_gradients - abs_gradients.min()) / (
            abs_gradients.max() - abs_gradients.min()
        )
    else:
        norm_gradients = abs_gradients
    axes[1, 0].imshow(norm_gradients, cmap="hot", alpha=0.5)
    axes[1, 0].set_title("Sensitivity Overlay")
    axes[1, 0].axis("off")

    # Gradient histogram
    axes[1, 1].hist(gradients_np.flatten(), bins=50, alpha=0.7, edgecolor="black")
    axes[1, 1].set_title("Gradient Distribution")
    axes[1, 1].set_xlabel("Gradient Value")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)

    # Top sensitive pixels visualization
    top_mask = np.zeros_like(abs_gradients)
    flat_abs = abs_gradients.flatten()
    if len(flat_abs) > 20:  # Make sure we have enough pixels
        top_indices = np.argpartition(flat_abs, -20)[-20:]  # Top 20 pixels
        top_coords = [
            (idx // abs_gradients.shape[1], idx % abs_gradients.shape[1])
            for idx in top_indices
        ]
        for row, col in top_coords:
            if (
                0 <= row < top_mask.shape[0] and 0 <= col < top_mask.shape[1]
            ):  # Bounds check
                top_mask[row, col] = 1

    axes[1, 2].imshow(original_np, cmap="gray", alpha=0.7)
    axes[1, 2].imshow(top_mask, cmap="Reds", alpha=0.8)
    axes[1, 2].set_title("Top 20 Most Sensitive Pixels")
    axes[1, 2].axis("off")

    # Overall title
    fig.suptitle(
        f"MNIST Sensitivity Analysis{title_suffix}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")

    return fig


def print_analysis_results(stats, predicted_class, true_class, confidence):
    """
    Print formatted results of the sensitivity analysis

    Args:
        stats: Dictionary of gradient statistics
        predicted_class: Model's prediction
        true_class: Ground truth label
        confidence: Model confidence
    """
    print("\n" + "=" * 60)
    print("MNIST SENSITIVITY ANALYSIS RESULTS")
    print("=" * 60)
    print(f"True Label: {true_class}")
    print(f"Predicted Label: {predicted_class}")
    print(f"Model Confidence: {confidence:.3f}")
    print(f"Prediction Correct: {'✓' if predicted_class == true_class else '✗'}")

    print(f"\nGRADIENT STATISTICS")
    print("-" * 30)
    print(f"Mean gradient magnitude: {stats['mean_magnitude']:.6f}")
    print(f"Std gradient magnitude:  {stats['std_magnitude']:.6f}")
    print(f"Max gradient magnitude:  {stats['max_magnitude']:.6f}")
    print(f"Min gradient magnitude:  {stats['min_magnitude']:.6f}")
    print(f"Total pixels analyzed:   {stats['total_pixels']}")

    print(f"\nTOP 10 MOST SENSITIVE PIXELS")
    print("-" * 40)
    print("Rank | Position  | Gradient    | Magnitude")
    print("-" * 40)
    for pixel in stats["top_pixels"]:
        print(
            f"{pixel['rank']:4d} | ({pixel['position'][0]:2d},{pixel['position'][1]:2d})     | "
            f"{pixel['gradient']:10.6f} | {pixel['magnitude']:9.6f}"
        )


def run_sensitivity_analysis(
    model_path=None, sample_index=0, target_class=None, save_dir=None, show_plots=True
):
    """
    Run complete sensitivity analysis on a single MNIST sample

    Args:
        model_path: Path to trained model (if None, trains a new one)
        sample_index: Which test sample to analyze
        target_class: Which class to compute gradients for (if None, uses prediction)
        save_dir: Directory to save results
        show_plots: Whether to display plots

    Returns:
        dict: Complete analysis results
    """
    # Prepare model and data
    if model_path and os.path.exists(model_path):
        model, device, sample_image, sample_label = prepare_for_sensitivity_analysis(
            model_path, sample_index
        )
    else:
        print("Training new MNIST model...")
        save_path = "mnist/saved_models/mnist_model.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model, _ = train_model(num_epochs=3, save_path=save_path)
        model, device, sample_image, sample_label = prepare_for_sensitivity_analysis(
            save_path, sample_index
        )

    # Compute gradients
    gradients, predicted_class, confidence = compute_gradients(
        model, device, sample_image, target_class
    )

    # Analyze statistics
    stats = analyze_gradient_statistics(gradients)

    # Print results
    print_analysis_results(stats, predicted_class, sample_label, confidence)

    # Create visualization
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"mnist_sensitivity_sample_{sample_index}.png"
        )

    title_suffix = f" - Sample {sample_index}"
    fig = visualize_sensitivity(
        sample_image,
        gradients,
        predicted_class,
        sample_label,
        confidence,
        save_path,
        title_suffix,
    )

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # Return complete results
    results = {
        "sample_index": sample_index,
        "true_label": sample_label,
        "predicted_label": predicted_class,
        "confidence": confidence,
        "correct_prediction": predicted_class == sample_label,
        "gradients": gradients,
        "statistics": stats,
        "model": model,
        "device": device,
    }

    return results
