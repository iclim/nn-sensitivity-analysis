import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
import os

from .utils import prepare_for_sensitivity_analysis, save_scaler
from .train import train_model


def compute_feature_gradients(model, device, X_data, y_data, target_class=None):
    """
    Compute gradients for all samples in the dataset

    Args:
        model: Trained neural network
        device: Device (CPU/GPU)
        X_data: Input features (numpy array or tensor)
        y_data: True labels (numpy array or tensor)
        target_class: Specific class to compute gradients for (if None, uses predicted class)

    Returns:
        dict: Contains gradients, predictions, and metadata
    """
    model.eval()

    # Convert to tensor if needed
    if isinstance(X_data, np.ndarray):
        X_tensor = torch.FloatTensor(X_data).to(device)
    else:
        X_tensor = X_data.to(device)

    if isinstance(y_data, np.ndarray):
        y_tensor = torch.LongTensor(y_data)
    else:
        y_tensor = y_data

    # Enable gradients
    X_tensor.requires_grad_(True)

    # Forward pass
    output = model(X_tensor)
    probabilities = torch.exp(output)
    predicted_classes = output.argmax(dim=1)

    # Store results
    all_gradients = []
    sample_info = []

    num_samples = X_tensor.shape[0]
    print(f"Computing gradients for {num_samples} samples...")

    for i in range(num_samples):
        # Clear previous gradients
        if X_tensor.grad is not None:
            X_tensor.grad.zero_()

        # Choose target class
        if target_class is not None:
            current_target = target_class
        else:
            current_target = predicted_classes[i].item()

        # Get score for target class
        target_score = output[i, current_target]

        # Compute gradients for this sample
        target_score.backward(retain_graph=True)

        # Store gradient for this sample
        sample_gradient = X_tensor.grad[i].clone().detach().cpu().numpy()
        all_gradients.append(sample_gradient)

        # Store metadata
        sample_info.append(
            {
                "sample_idx": i,
                "true_class": y_tensor[i].item(),
                "predicted_class": predicted_classes[i].item(),
                "target_class": current_target,
                "confidence": probabilities[i, current_target].item(),
                "correct_prediction": predicted_classes[i].item() == y_tensor[i].item(),
                "gradient": sample_gradient,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples")

    return {
        "gradients": np.array(all_gradients),
        "sample_info": sample_info,
        "feature_count": X_tensor.shape[1],
    }


def analyze_feature_importance(gradient_results, feature_names):
    """
    Analyze feature importance across the entire dataset

    Args:
        gradient_results: Results from compute_feature_gradients
        feature_names: List of feature names

    Returns:
        dict: Feature importance statistics
    """
    gradients = gradient_results["gradients"]
    sample_info = gradient_results["sample_info"]

    # Calculate statistics for each feature
    feature_stats = {}

    for i, feature_name in enumerate(feature_names):
        feature_gradients = gradients[:, i]
        abs_gradients = np.abs(feature_gradients)

        feature_stats[feature_name] = {
            "mean_magnitude": np.mean(abs_gradients),
            "std_magnitude": np.std(abs_gradients),
            "max_magnitude": np.max(abs_gradients),
            "min_magnitude": np.min(abs_gradients),
            "mean_gradient": np.mean(feature_gradients),
            "median_magnitude": np.median(abs_gradients),
            "percentile_95": np.percentile(abs_gradients, 95),
            "feature_index": i,
        }

    # Calculate per-class statistics
    class_stats = defaultdict(lambda: defaultdict(list))

    for info in sample_info:
        true_class = info["true_class"]
        gradient = info["gradient"]

        for i, feature_name in enumerate(feature_names):
            class_stats[true_class][feature_name].append(abs(gradient[i]))

    # Aggregate per-class statistics
    class_feature_stats = {}
    for class_id in class_stats:
        class_feature_stats[class_id] = {}
        for feature_name in feature_names:
            if feature_name in class_stats[class_id]:
                values = class_stats[class_id][feature_name]
                class_feature_stats[class_id][feature_name] = {
                    "mean_magnitude": np.mean(values),
                    "std_magnitude": np.std(values),
                    "count": len(values),
                }

    return {
        "overall_feature_stats": feature_stats,
        "class_feature_stats": class_feature_stats,
        "total_samples": len(sample_info),
    }


def visualize_feature_importance(importance_stats, feature_names, save_path=None):
    """
    Create comprehensive visualizations of feature importance

    Args:
        importance_stats: Results from analyze_feature_importance
        feature_names: List of feature names
        save_path: Optional path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    feature_stats = importance_stats["overall_feature_stats"]

    # Create figure with subplots - make it larger for better text spacing
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # Prepare data for plotting
    features = list(feature_names)
    mean_magnitudes = [feature_stats[f]["mean_magnitude"] for f in features]
    std_magnitudes = [feature_stats[f]["std_magnitude"] for f in features]
    max_magnitudes = [feature_stats[f]["max_magnitude"] for f in features]
    mean_gradients = [feature_stats[f]["mean_gradient"] for f in features]

    # Sort features by mean magnitude for better visualization
    sorted_indices = np.argsort(mean_magnitudes)[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_mean_mag = [mean_magnitudes[i] for i in sorted_indices]
    sorted_std_mag = [std_magnitudes[i] for i in sorted_indices]
    sorted_max_mag = [max_magnitudes[i] for i in sorted_indices]
    sorted_mean_grad = [mean_gradients[i] for i in sorted_indices]

    # 1. Mean gradient magnitude (most important plot)
    axes[0, 0].barh(
        range(len(sorted_features)), sorted_mean_mag, color="skyblue", edgecolor="navy"
    )
    axes[0, 0].set_yticks(range(len(sorted_features)))
    axes[0, 0].set_yticklabels(
        [f.replace("_", " ").title() for f in sorted_features], fontsize=9
    )
    axes[0, 0].set_xlabel("Mean Gradient Magnitude", fontsize=11)
    axes[0, 0].set_title(
        "Feature Sensitivity Rankings\n(Higher = More Important)",
        fontweight="bold",
        fontsize=12,
    )
    axes[0, 0].grid(axis="x", alpha=0.3)

    # Add values on bars
    for i, v in enumerate(sorted_mean_mag):
        axes[0, 0].text(
            v + max(sorted_mean_mag) * 0.01, i, f"{v:.4f}", va="center", fontsize=7
        )

    # 2. Standard deviation of gradients (variability)
    axes[0, 1].barh(
        range(len(sorted_features)),
        sorted_std_mag,
        color="lightcoral",
        edgecolor="darkred",
    )
    axes[0, 1].set_yticks(range(len(sorted_features)))
    axes[0, 1].set_yticklabels(
        [f.replace("_", " ").title() for f in sorted_features], fontsize=9
    )
    axes[0, 1].set_xlabel("Standard Deviation of Gradient Magnitude", fontsize=11)
    axes[0, 1].set_title(
        "Feature Sensitivity Variability\n(Higher = More Variable Impact)",
        fontweight="bold",
        fontsize=12,
    )
    axes[0, 1].grid(axis="x", alpha=0.3)

    # 3. Mean gradient (positive/negative influence)
    colors = ["green" if x > 0 else "red" for x in sorted_mean_grad]
    axes[1, 0].barh(
        range(len(sorted_features)), sorted_mean_grad, color=colors, alpha=0.7
    )
    axes[1, 0].set_yticks(range(len(sorted_features)))
    axes[1, 0].set_yticklabels(
        [f.replace("_", " ").title() for f in sorted_features], fontsize=9
    )
    axes[1, 0].set_xlabel("Mean Gradient Value", fontsize=11)
    axes[1, 0].set_title(
        "Feature Influence Direction\n(Green=Positive, Red=Negative)",
        fontweight="bold",
        fontsize=12,
    )
    axes[1, 0].axvline(x=0, color="black", linestyle="-", alpha=0.5)
    axes[1, 0].grid(axis="x", alpha=0.3)

    # 4. Max gradient magnitude (peak sensitivity)
    axes[1, 1].barh(
        range(len(sorted_features)), sorted_max_mag, color="gold", edgecolor="orange"
    )
    axes[1, 1].set_yticks(range(len(sorted_features)))
    axes[1, 1].set_yticklabels(
        [f.replace("_", " ").title() for f in sorted_features], fontsize=9
    )
    axes[1, 1].set_xlabel("Maximum Gradient Magnitude", fontsize=11)
    axes[1, 1].set_title(
        "Peak Feature Sensitivity\n(Highest Observed Impact)",
        fontweight="bold",
        fontsize=12,
    )
    axes[1, 1].grid(axis="x", alpha=0.3)

    plt.suptitle(
        "Wine Quality Neural Network - Feature Sensitivity Analysis",
        fontsize=18,
        fontweight="bold",
        y=0.96,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Feature importance visualization saved to {save_path}")

    return fig


def create_class_comparison_plot(importance_stats, feature_names, save_path=None):
    """
    Create visualization comparing feature importance across wine classes
    """
    class_stats = importance_stats["class_feature_stats"]

    # Prepare data for heatmap
    classes = sorted(class_stats.keys())
    class_names = [f"Class {c}" for c in classes]

    # Create matrix of mean magnitudes
    magnitude_matrix = []
    for class_id in classes:
        class_magnitudes = []
        for feature in feature_names:
            if feature in class_stats[class_id]:
                class_magnitudes.append(
                    class_stats[class_id][feature]["mean_magnitude"]
                )
            else:
                class_magnitudes.append(0)
        magnitude_matrix.append(class_magnitudes)

    magnitude_matrix = np.array(magnitude_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        magnitude_matrix,
        xticklabels=[f.replace("_", " ").title() for f in feature_names],
        yticklabels=class_names,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Mean Gradient Magnitude"},
    )

    plt.title(
        "Feature Sensitivity by Wine Class\n(Darker = More Important for Classification)",
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Features", fontweight="bold")
    plt.ylabel("Wine Classes", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        class_save_path = save_path.replace(".png", "_by_class.png")
        plt.savefig(class_save_path, dpi=300, bbox_inches="tight")
        print(f"Class comparison visualization saved to {class_save_path}")

    return fig


def print_sensitivity_results(importance_stats, feature_names, top_k=10):
    """
    Print formatted results of the sensitivity analysis
    """
    feature_stats = importance_stats["overall_feature_stats"]
    total_samples = importance_stats["total_samples"]

    print("\n" + "=" * 80)
    print("WINE QUALITY FEATURE SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Total samples analyzed: {total_samples}")
    print(f"Total features analyzed: {len(feature_names)}")

    # Sort features by mean magnitude
    sorted_features = sorted(
        feature_names, key=lambda f: feature_stats[f]["mean_magnitude"], reverse=True
    )

    print(f"\nTOP {top_k} MOST SENSITIVE FEATURES")
    print("-" * 80)
    print(
        f"{'Rank':<4} {'Feature Name':<25} {'Mean Mag':<12} {'Std Mag':<12} {'Max Mag':<12}"
    )
    print("-" * 80)

    for i, feature in enumerate(sorted_features[:top_k]):
        stats = feature_stats[feature]
        print(
            f"{i + 1:<4} {feature.replace('_', ' ').title():<25} "
            f"{stats['mean_magnitude']:<12.6f} {stats['std_magnitude']:<12.6f} "
            f"{stats['max_magnitude']:<12.6f}"
        )

    print(f"\nFEATURE SENSITIVITY SUMMARY")
    print("-" * 50)
    all_magnitudes = [feature_stats[f]["mean_magnitude"] for f in feature_names]
    print(f"Average feature sensitivity: {np.mean(all_magnitudes):.6f}")
    print(f"Most sensitive feature: {sorted_features[0].replace('_', ' ').title()}")
    print(f"Least sensitive feature: {sorted_features[-1].replace('_', ' ').title()}")

    # Top 3 vs bottom 3 comparison
    top_3_avg = np.mean(
        [feature_stats[f]["mean_magnitude"] for f in sorted_features[:3]]
    )
    bottom_3_avg = np.mean(
        [feature_stats[f]["mean_magnitude"] for f in sorted_features[-3:]]
    )
    sensitivity_ratio = top_3_avg / bottom_3_avg if bottom_3_avg > 0 else float("inf")

    print(f"Top 3 features avg sensitivity: {top_3_avg:.6f}")
    print(f"Bottom 3 features avg sensitivity: {bottom_3_avg:.6f}")
    print(f"Sensitivity ratio (top/bottom): {sensitivity_ratio:.2f}x")


def run_sensitivity_analysis(
    model_path=None, scaler_path=None, save_dir=None, show_plots=True, target_class=None
):
    """
    Run complete sensitivity analysis on the wine quality dataset

    Args:
        model_path: Path to trained model (if None, trains a new one)
        scaler_path: Path to saved scaler
        save_dir: Directory to save results
        show_plots: Whether to display plots
        target_class: Which class to compute gradients for (if None, uses predictions)

    Returns:
        dict: Complete analysis results
    """

    print("Preparing Wine Quality Sensitivity Analysis")
    print("=" * 50)

    # Prepare model and data
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model, device, X_scaled, y, scaler, feature_names = (
            prepare_for_sensitivity_analysis(model_path, scaler_path)
        )
    else:
        print("Training new wine quality model...")
        save_path = "wine/saved_models/wine_model.pth"
        scaler_save_path = "wine/saved_models/wine_scaler.pkl"

        model, history, scaler, feature_names = train_model(
            num_epochs=50, save_path=save_path
        )

        # Save the scaler
        save_scaler(scaler, scaler_save_path)

        # Get data for analysis
        model, device, X_scaled, y, scaler, feature_names = (
            prepare_for_sensitivity_analysis(save_path, scaler_save_path)
        )

    print(f"Dataset: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    print(f"Features: {', '.join(feature_names)}")

    # Compute gradients for all samples
    print("\nComputing feature gradients...")
    gradient_results = compute_feature_gradients(
        model, device, X_scaled, y, target_class
    )

    # Analyze feature importance
    print("Analyzing feature importance...")
    importance_stats = analyze_feature_importance(gradient_results, feature_names)

    # Print results
    print_sensitivity_results(importance_stats, feature_names)

    # Create visualizations
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "wine_feature_sensitivity.png")

    print("\nCreating visualizations...")
    fig1 = visualize_feature_importance(importance_stats, feature_names, save_path)
    fig2 = create_class_comparison_plot(importance_stats, feature_names, save_path)

    if show_plots:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

    # Calculate model performance stats
    sample_info = gradient_results["sample_info"]
    correct_predictions = sum(1 for info in sample_info if info["correct_prediction"])
    accuracy = correct_predictions / len(sample_info)
    avg_confidence = np.mean([info["confidence"] for info in sample_info])

    print(f"\nMODEL PERFORMANCE SUMMARY")
    print("-" * 40)
    print(f"Overall accuracy: {accuracy:.2%}")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Total samples: {len(sample_info)}")

    # Return complete results
    results = {
        "gradient_results": gradient_results,
        "importance_stats": importance_stats,
        "feature_names": feature_names,
        "model_accuracy": accuracy,
        "average_confidence": avg_confidence,
        "total_samples": len(sample_info),
        "model": model,
        "device": device,
        "scaler": scaler,
    }

    return results


def compare_feature_distributions(results, feature_names, save_dir=None):
    """
    Create additional analysis comparing gradient distributions across features
    """
    gradient_results = results["gradient_results"]
    gradients = gradient_results["gradients"]

    # Create violin plot of gradient distributions
    fig, ax = plt.subplots(figsize=(16, 8))

    # Prepare data for violin plot
    gradient_data = []
    feature_labels = []

    for i, feature in enumerate(feature_names):
        feature_gradients = np.abs(gradients[:, i])
        gradient_data.extend(feature_gradients)
        feature_labels.extend(
            [feature.replace("_", " ").title()] * len(feature_gradients)
        )

    # Create DataFrame for seaborn
    df = pd.DataFrame({"Feature": feature_labels, "Gradient_Magnitude": gradient_data})

    # Create violin plot
    sns.violinplot(data=df, x="Feature", y="Gradient_Magnitude", ax=ax)
    plt.xticks(rotation=45, ha="right")
    plt.title(
        "Distribution of Gradient Magnitudes by Feature\n(Shows variability in feature sensitivity)",
        fontweight="bold",
        pad=20,
    )
    plt.ylabel("Gradient Magnitude", fontweight="bold")
    plt.xlabel("Features", fontweight="bold")
    plt.tight_layout()

    if save_dir:
        dist_save_path = os.path.join(save_dir, "wine_gradient_distributions.png")
        plt.savefig(dist_save_path, dpi=300, bbox_inches="tight")
        print(f"Gradient distribution plot saved to {dist_save_path}")

    return fig


def create_feature_correlation_analysis(results, X_data, feature_names, save_dir=None):
    """
    Analyze correlation between feature sensitivity and feature values
    """
    gradient_results = results["gradient_results"]
    gradients = gradient_results["gradients"]

    # Calculate correlation between feature values and their gradients
    correlations = []
    for i, feature in enumerate(feature_names):
        feature_values = X_data[:, i]
        feature_gradients = np.abs(gradients[:, i])
        correlation = np.corrcoef(feature_values, feature_gradients)[0, 1]
        correlations.append(correlation)

    # Create correlation plot
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ["red" if x < 0 else "blue" for x in correlations]
    bars = ax.barh(feature_names, correlations, color=colors, alpha=0.7)

    ax.set_xlabel(
        "Correlation (Feature Value vs Gradient Magnitude)", fontweight="bold"
    )
    ax.set_title(
        "Feature Value-Sensitivity Correlation\n(Blue=Positive, Red=Negative)",
        fontweight="bold",
        pad=20,
    )
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    # Add correlation values on bars
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        ax.text(
            corr + 0.01 if corr >= 0 else corr - 0.01,
            i,
            f"{corr:.3f}",
            va="center",
            ha="left" if corr >= 0 else "right",
            fontsize=9,
        )

    plt.tight_layout()

    if save_dir:
        corr_save_path = os.path.join(save_dir, "wine_feature_correlation.png")
        plt.savefig(corr_save_path, dpi=300, bbox_inches="tight")
        print(f"Feature correlation analysis saved to {corr_save_path}")

    return fig
