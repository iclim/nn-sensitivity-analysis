import argparse
import sys
from mnist.sensitivity import run_sensitivity_analysis as mnist_sensitivity


def run_mnist_analysis(args):
    """Run sensitivity analysis on MNIST dataset"""

    print("Running MNIST Sensitivity Analysis")
    print("=" * 50)

    if args.multiple_samples == 1:
        # Single sample analysis
        results = mnist_sensitivity(
            model_path=args.model_path,
            sample_index=args.sample_index,
            target_class=args.target_class,
            save_dir=f"{args.save_dir}/mnist",
            show_plots=not args.no_display,
        )
        return results

    else:
        # Multiple sample analysis
        print(f"Analyzing {args.multiple_samples} samples...")
        all_results = []

        for i in range(args.multiple_samples):
            print(f"\n{'=' * 20} Sample {i} {'=' * 20}")
            results = mnist_sensitivity(
                model_path=args.model_path,
                sample_index=i,
                target_class=args.target_class,
                save_dir=f"{args.save_dir}/mnist",
                show_plots=not args.no_display,
            )
            all_results.append(results)

        # Print summary
        print_multi_sample_summary(all_results)
        return all_results


def print_multi_sample_summary(results_list):
    """Print summary statistics for multiple sample analysis"""
    print("\n" + "=" * 60)
    print("MULTI-SAMPLE ANALYSIS SUMMARY")
    print("=" * 60)

    total_samples = len(results_list)
    correct_predictions = sum(1 for r in results_list if r["correct_prediction"])
    accuracy = correct_predictions / total_samples

    print(f"Total samples analyzed: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")

    # Aggregate statistics
    all_mean_magnitudes = [r["statistics"]["mean_magnitude"] for r in results_list]
    all_max_magnitudes = [r["statistics"]["max_magnitude"] for r in results_list]
    all_confidences = [r["confidence"] for r in results_list]

    print(f"\nAGGREGATE SENSITIVITY STATISTICS")
    print("-" * 40)
    print(
        f"Average mean gradient magnitude: {sum(all_mean_magnitudes) / len(all_mean_magnitudes):.6f}"
    )
    print(
        f"Average max gradient magnitude:  {sum(all_max_magnitudes) / len(all_max_magnitudes):.6f}"
    )
    print(
        f"Average model confidence:        {sum(all_confidences) / len(all_confidences):.3f}"
    )

    # Show per-class breakdown
    class_results = {}
    for result in results_list:
        true_label = result["true_label"]
        if true_label not in class_results:
            class_results[true_label] = {
                "count": 0,
                "correct": 0,
                "mean_confidence": 0,
                "mean_sensitivity": 0,
            }

        class_results[true_label]["count"] += 1
        if result["correct_prediction"]:
            class_results[true_label]["correct"] += 1
        class_results[true_label]["mean_confidence"] += result["confidence"]
        class_results[true_label]["mean_sensitivity"] += result["statistics"][
            "mean_magnitude"
        ]

    print(f"\nPER-CLASS BREAKDOWN")
    print("-" * 50)
    print("Class | Count | Accuracy | Avg Confidence | Avg Sensitivity")
    print("-" * 50)

    for digit in sorted(class_results.keys()):
        stats = class_results[digit]
        count = stats["count"]
        accuracy = stats["correct"] / count if count > 0 else 0
        avg_conf = stats["mean_confidence"] / count if count > 0 else 0
        avg_sens = stats["mean_sensitivity"] / count if count > 0 else 0

        print(
            f"  {digit}   |   {count}   |  {accuracy:.2%}   |    {avg_conf:.3f}     |   {avg_sens:.6f}"
        )


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Run derivative-based sensitivity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dataset mnist
  python main.py --dataset mnist --model-path mnist/saved_models/model.pth
  python main.py --dataset mnist --sample-index 42 --target-class 3
  python main.py --dataset mnist --multiple-samples 10 --save-dir results/
        """,
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        choices=["mnist"],
        default="mnist",
        help="Dataset to analyze (default: mnist)",
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (if not provided, will train new model)",
    )

    # Sample selection
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of sample to analyze (default: 0)",
    )
    parser.add_argument(
        "--multiple-samples",
        type=int,
        default=1,
        help="Number of samples to analyze (default: 1)",
    )

    # Analysis configuration
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Target class for gradient computation (if not provided, uses prediction)",
    )

    # Output configuration
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display plots (useful for batch processing)",
    )

    args = parser.parse_args()

    # Main execution
    print("Derivative-Based Sensitivity Analysis")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Model path: {args.model_path or 'Will train new model'}")
    print(f"Samples to analyze: {args.multiple_samples}")
    if args.target_class is not None:
        print(f"Target class: {args.target_class}")
    print(f"Save directory: {args.save_dir}")
    print()

    # Run analysis based on dataset
    try:
        if args.dataset == "mnist":
            results = run_mnist_analysis(args)
            if results:
                print(f"\n✓ Analysis completed successfully!")
                if args.save_dir:
                    output_dir = f"{args.save_dir} / {args.dataset}"
                    print(f"  Results saved to: {output_dir}")
            else:
                print(f"\n✗ Analysis failed!")
                sys.exit(1)
        else:
            print(f"Dataset '{args.dataset}' not yet implemented")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
