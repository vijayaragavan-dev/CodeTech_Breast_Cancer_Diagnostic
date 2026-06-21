# Breast Cancer Diagnostic System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A professional Machine Learning project that predicts whether a breast tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** using the Breast Cancer Wisconsin Dataset and Logistic Regression algorithm.

---

## Project Overview

### What is Breast Cancer?

Breast cancer is a disease in which cells in the breast grow uncontrollably. It is the most commonly diagnosed cancer among women worldwide. According to the World Health Organization, early detection and diagnosis significantly improve survival rates.

### Why Early Diagnosis Matters?

When breast cancer is detected early, the five-year survival rate exceeds **90%**. Late-stage diagnosis dramatically reduces survival chances. This makes early, accurate diagnosis one of the most critical factors in breast cancer treatment.

### How Machine Learning Helps?

Machine Learning models can analyze tumor measurements (size, texture, shape, etc.) and learn patterns that distinguish cancerous tumors from non-cancerous ones. These models provide:

- **Fast** predictions (milliseconds per sample)
- **Consistent** results (no human fatigue)
- **High accuracy** (often exceeding 95%)

---

## Problem Statement

Develop a Machine Learning model that, given 30 measurements of a breast tumor (computed from digitized images of fine needle aspirates), can accurately classify the tumor as **Malignant** or **Benign**.

---

## Project Objectives

- Load and explore the Breast Cancer Wisconsin Dataset
- Perform data analysis and visualization
- Train a Logistic Regression model
- Evaluate model performance using multiple metrics
- Save the trained model for future use
- Document the entire process for beginners

---

## Dataset Information

| Attribute | Value |
|-----------|-------|
| **Dataset Source** | UCI Machine Learning Repository (via Scikit-Learn) |
| **Total Records** | 569 tumor samples |
| **Total Features** | 30 numeric measurements |
| **Target Classes** | 0 = Malignant (Cancerous), 1 = Benign (Non-Cancerous) |
| **Feature Types** | Mean, Standard Error, and Worst values of 10 cell nucleus characteristics |
| **Missing Values** | None |

### Feature Details

The 30 features include measurements of cell nuclei such as:
- Radius, Texture, Perimeter, Area
- Smoothness, Compactness, Concavity
- Concave Points, Symmetry, Fractal Dimension

Each is recorded as three values: **mean**, **standard error**, and **worst** (largest).

---

## Technologies Used

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.8+ | Programming language |
| Pandas | 1.3+ | Data manipulation and analysis |
| NumPy | 1.21+ | Numerical computations |
| Matplotlib | 3.4+ | Data visualization |
| Seaborn | 0.11+ | Statistical data visualization |
| Scikit-Learn | 1.0+ | Machine Learning algorithms |
| Pickle | (built-in) | Model persistence |

---

## Machine Learning Workflow

### 1. Data Collection

The dataset is loaded directly from Scikit-Learn's built-in `load_breast_cancer()` function. This dataset originates from the UCI Machine Learning Repository and contains 569 samples with 30 features.

### 2. Data Exploration

We examine the dataset structure — checking the first few rows, understanding column names, data types, and statistical summaries. This helps us get familiar with the data.

### 3. Data Cleaning

The dataset has **zero missing values**, which is ideal. In real-world projects, this is where we would handle null values, outliers, and data inconsistencies.

### 4. Visualization

We generate three key visualizations:
- **Target Distribution Plot** — Shows the balance between Malignant and Benign samples
- **Correlation Heatmap** — Shows relationships between all 30 features
- **Confusion Matrix Heatmap** — Visualizes prediction performance

### 5. Feature Selection

All 30 features are used since Logistic Regression handles multi-dimensional data well. Features are scaled using `StandardScaler` to normalize them (mean = 0, standard deviation = 1).

### 6. Data Splitting

The dataset is split into:
- **80% Training data** — Used to teach the model
- **20% Testing data** — Used to evaluate performance

The split is **stratified**, meaning both sets maintain the same ratio of Malignant to Benign samples.

### 7. Model Training

A **Logistic Regression** model is trained on the training data. The algorithm learns the relationship between the 30 features and the target diagnosis.

### 8. Prediction

The trained model makes predictions on the unseen test data.

### 9. Evaluation

We calculate:
- **Accuracy** — Overall correctness
- **Precision** — How many predicted cancers were actually cancers
- **Recall** — How many actual cancers were detected
- **F1 Score** — Harmonic mean of Precision and Recall
- **Confusion Matrix** — Detailed breakdown of correct/incorrect predictions
- **Classification Report** — Complete performance summary

### 10. Model Saving

The trained model is saved using **Pickle** as `cancer_model.pkl` for future use without retraining.

---

## Algorithm Used

### Logistic Regression

**What it is:** Despite its name, Logistic Regression is a **classification** algorithm (not regression). It predicts the probability that a sample belongs to a particular class.

**How it works (simplified):**

1. It takes the 30 tumor measurements as input
2. It computes a weighted sum of these measurements
3. It passes this sum through a **sigmoid function**, which outputs a probability between 0 and 1
4. If the probability > 0.5, it predicts **Malignant** (or **Benign**, depending on configuration)

**Why Logistic Regression?**
- Simple and interpretable
- Works well with high-dimensional data
- Fast to train and predict
- Provides probability scores (not just labels)
- Excellent baseline for binary classification

---

## Project Structure

```
CodeTech_Breast_Cancer_Diagnostic/
│
├── breast_cancer.py          # Main Python script
├── cancer_model.pkl          # Saved trained model
├── requirements.txt          # Required Python packages
├── README.md                 # Project documentation (this file)
├── dataset_info.txt          # Dataset details
│
├── screenshots/              # Generated visualizations
│   ├── distribution.png      # Target distribution plot
│   ├── heatmap.png           # Feature correlation heatmap
│   └── confusion_matrix.png  # Confusion matrix heatmap
│
└── outputs/                  # Generated outputs
    └── model_metrics.txt     # Evaluation metrics report
```

---

## Installation Guide

### Step 1: Install Python

1. Go to [python.org](https://python.org/downloads)
2. Download Python 3.8 or higher (click the yellow "Download" button)
3. Run the installer
4. **Important:** Check the box that says **"Add Python to PATH"**
5. Click "Install Now"
6. Open Command Prompt (Windows) or Terminal (Mac/Linux) and type:
   ```
   python --version
   ```
   You should see `Python 3.8.x` or higher.

### Step 2: Download the Project

Click the green **"Code"** button on this GitHub page, then click **"Download ZIP"**. Extract the ZIP file to your Desktop.

### Step 3: Open VS Code

1. Download and install [VS Code](https://code.visualstudio.com/download)
2. Open VS Code
3. Click **File > Open Folder**
4. Select the extracted project folder (`CodeTech_Breast_Cancer_Diagnostic`)

### Step 4: Open Terminal

In VS Code, click **Terminal > New Terminal** (or press `` Ctrl + ` ``).

### Step 5: Install Dependencies

In the terminal, type the following command and press Enter:

```
pip install -r requirements.txt
```

This installs all required Python packages (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn).

### Step 6: Run the Project

In the terminal, type:

```
python breast_cancer.py
```

---

## How to Run the Project

1. Open your terminal (Command Prompt, PowerShell, or VS Code terminal)
2. Navigate to the project folder:
   ```
   cd Desktop/CodeTech_Breast_Cancer_Diagnostic
   ```
3. Install dependencies (only once):
   ```
   pip install -r requirements.txt
   ```
4. Run the script:
   ```
   python breast_cancer.py
   ```

That's it! The script will:
- Load the dataset
- Display analysis and statistics
- Show three graphs (one at a time — close each to see the next)
- Print evaluation metrics
- Save the trained model

---

## Expected Output

### Graphs

When you run the script, three visualization windows will appear (one at a time):

1. **Distribution Plot** — A bar chart showing how many Malignant vs. Benign samples are in the dataset
2. **Correlation Heatmap** — A colorful grid showing relationships between features
3. **Confusion Matrix** — A 2x2 grid showing correct and incorrect predictions

Close each graph window to proceed to the next one.

### Console Output

The terminal will display:
- Dataset information (shape, columns, statistics)
- Missing value analysis
- Training progress
- **Accuracy score** (typically 97-98%)
- Precision, Recall, and F1 Score
- Classification report
- Confirmation messages

### Saved Model

After execution, `cancer_model.pkl` is created in the project folder. This file contains the trained model and can be loaded in other Python scripts without retraining.

---

## Evaluation Metrics

### Accuracy

**Simple explanation:** Out of all predictions made, how many were correct?

**Formula:** (Correct Predictions) / (Total Predictions)

**Example:** If the model makes 100 predictions and 97 are correct, accuracy is 97%.

### Precision

**Simple explanation:** When the model says "Cancer," how often is it actually cancer?

**Formula:** (True Positives) / (True Positives + False Positives)

**Why it matters:** Low precision means many false alarms — telling healthy patients they might have cancer.

### Recall (Sensitivity)

**Simple explanation:** Out of all actual cancer cases, how many did the model catch?

**Formula:** (True Positives) / (True Positives + False Negatives)

**Why it matters:** Low recall means missing actual cancer cases — which could be life-threatening.

### F1 Score

**Simple explanation:** A balanced score that combines both Precision and Recall. It is the harmonic mean of the two.

**Formula:** 2 × (Precision × Recall) / (Precision + Recall)

**Why it matters:** Provides a single number that balances both concerns. High F1 means the model is both precise and catches most cases.

---

## Screenshots

### Target Distribution Plot

![Distribution Plot](screenshots/distribution.png)

*Figure 1: Distribution of Malignant vs. Benign samples in the dataset.*

### Correlation Heatmap

![Heatmap](screenshots/heatmap.png)

*Figure 2: Correlation between all 30 features in the dataset.*

### Confusion Matrix

![Confusion Matrix](screenshots/confusion_matrix.png)

*Figure 3: Confusion matrix showing model prediction performance.*

---

## Future Improvements

### Advanced Algorithms

- **Random Forest** — Ensemble method that typically achieves higher accuracy
- **XGBoost** — Gradient boosting for even better performance
- **Support Vector Machines (SVM)** — Effective for high-dimensional medical data

### Deployment

- **Flask API** — Deploy the model as a REST API for integration with web applications
- **Streamlit Dashboard** — Create an interactive web interface for non-technical users
- **Real-time Prediction Interface** — Build a UI where doctors can input measurements and get instant predictions

### Model Enhancements

- Hyperparameter tuning with GridSearchCV
- Cross-validation for more robust evaluation
- Feature importance analysis and dimensionality reduction
- Neural network implementation using TensorFlow/PyTorch

---

## Learning Outcomes

Through this project, I learned:

- **Data Manipulation** — Using Pandas to load, explore, and analyze datasets
- **Data Visualization** — Creating professional plots with Matplotlib and Seaborn
- **Machine Learning Workflow** — Understanding the end-to-end process from data to deployment
- **Classification Algorithms** — How Logistic Regression works and when to use it
- **Model Evaluation** — Interpreting accuracy, precision, recall, F1 score, and confusion matrices
- **Feature Scaling** — Why normalization is important for ML algorithms
- **Model Persistence** — Saving and loading trained models with Pickle
- **Project Organization** — Structuring ML projects for readability and reproducibility
- **Documentation** — Writing clear, beginner-friendly documentation

---

## Author

**Vijayaragavan U**

B.E Computer Science and Engineering  
Saranathan College of Engineering  
Tiruchirappalli, Tamil Nadu, India

---

*Submitted as part of the Machine Learning Internship at CodeTech IT Solutions*

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.
