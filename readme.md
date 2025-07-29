# Neural Network Sensitivity Analysis

A Neural Network is a great tool for learning complex patterns in data, but lack the interpretability
in which many other machine learning models come equipt. For example, a random forest model allows the user
to view the decision trees learned in the training process, which gives some insight into feature importance
and the way the model makes decision. In this analysis, we demonstrate a derivative-based local sensitivity
analysis on Neural Networks trained on 
- the [mnist](https://archive.ics.uci.edu/dataset/683/mnist+database+of+handwritten+digits) dataset for 
handwritten digit recognition
- a [wine](https://archive.ics.uci.edu/dataset/109/wine) dataset for classifying Italian wines into
three different cultivars by chemical composition

## Introduction

A trained neural network makes predictions by learning a function $f$ that maps input feature vectors $x$
to predictions $f(x)$. Since neural networks are differentiable functions, we can analyze how 
sensitive the model's predictions are to changes in individual input features.

This sensitivity analysis examines the partial derivatives:

$$\dfrac{\partial f(x)}{\partial x_i}$$

where $x_i$ represents a specific input feature. These gradients quantify how much the model's output changes in response to small perturbations in each feature, revealing which inputs most strongly influence the network's decisions.

### Why This Matters

Understanding feature sensitivity helps us:
- **Interpret model decisions**: See which features the network considers most important
- **Validate model behavior**: Ensure the network focuses on meaningful patterns rather than spurious correlations  
- **Debug poor performance**: Identify if the model is relying on unexpected or irrelevant features
- **Build trust**: Provide explainable insights into the "black box" neural network

### Implementation

The gradients are computed efficiently using backpropagation through the trained network. For each input
sample, we calculate how the predicted class score changes with respect to each input feature, providing 
both local (per-sample) and global (dataset-wide) insights into model behavior.

## Usage

### Dependencies

- Matplotlib
- Numpy
- Pandas
- PyTorch
- Scikit-Learn
- Seaborn

### Run

Each demonstration can be called from `main.py` with the following optional arguments:
- `--dataset <wine|mnist>` 
- `--model-path` to import previously trained model parameters
- `--scaler-path` to import previously scaler used to normalize the wine dataset
- `--sample-index` (`mnist` only) to pick a certain sample by index
- `--multiple-samples` (`mnist` only) to specify how many sample to analyze starting from
index 0
- `--target-class` specify target class for gradient computation and if not provided will use
prediction as target
- `--save-dir` location to write any results
- `--no-display` to supress plots
