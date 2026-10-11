# Fetal Health Classification Using TensorFlow and Keras

A deep learning project that classifies fetal health status from cardiotocography (CTG) measurements using an Artificial Neural Network (ANN) implemented with TensorFlow/Keras.

> **Disclaimer:** This is an educational machine learning project, not a validated medical diagnostic tool. Predictions must not be used for clinical decisions.

## Project Overview

The model predicts one of three fetal health categories from 21 numerical CTG features:

| Original label | Encoded label | Category |
| --- | --- | --- |
| 1 | 0 | Normal |
| 2 | 1 | Suspect |
| 3 | 2 | Pathological |

## Dataset

- **File:** `fetal_health.csv` (provide separately; not included in the notebook)
- **Initial shape:** 2,126 rows × 22 columns
- **Inputs:** 21 numerical features, including baseline fetal heart rate, accelerations, fetal movement, uterine contractions, decelerations, and variability-related measurements
- **Target:** `fetal_health`
- **Missing values:** None reported in the notebook
- **Duplicate rows:** 13 removed, leaving 2,113 records

Class distribution after preprocessing:

| Class | Count |
| --- | ---: |
| Normal (0) | 1,646 |
| Suspect (1) | 292 |
| Pathological (2) | 175 |

The classes are imbalanced, so accuracy alone does not fully describe model performance.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

## Workflow

1. Load `fetal_health.csv` using Pandas.
2. Inspect dataset dimensions, data types, missing values, and duplicates.
3. Remove duplicate rows and encode target labels from `1, 2, 3` to `0, 1, 2`.
4. Separate 21 input features from the target column.
5. Split the dataset into training (80%) and testing (20%) sets using `random_state=42`.
6. Train a feed-forward ANN for 150 epochs.
7. Evaluate test-set accuracy and save the trained model.

## ANN Architecture

| Layer | Units | Activation |
| --- | ---: | --- |
| Input | 21 features | — |
| Dense hidden layer 1 | 1,024 | ReLU |
| Dense hidden layer 2 | 512 | ReLU |
| Output | 3 | Softmax |

**Training configuration:** Adam optimizer, sparse categorical cross-entropy loss, accuracy metric, 150 epochs, batch size 32.

## Results

The notebook reports:

**Test accuracy: 92.9%** (`0.929` as printed by the evaluation cell).

This is the result of the recorded notebook run, not a guarantee of performance on new data. The notebook does not report class-wise precision, recall, F1-score, or a confusion matrix.

## Installation and Usage

1. Clone your GitHub repository and open its folder.
2. Create and activate a Python virtual environment (recommended).
3. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn tensorflow jupyter
   ```

4. Place `fetal_health.csv` in the project folder.
5. Open the notebook:

   ```bash
   jupyter notebook tf_clsftn.ipynb
   ```

6. Update the dataset path in the first cell to a portable relative path:

   ```python
   df = pd.read_csv('fetal_health.csv')
   ```

7. Run the notebook cells in order through training and evaluation.

## Saving the Model

The notebook saves and reloads the model as `model.h5`:

```python
model.save('model.h5')

from tensorflow.keras.models import load_model
model = load_model('model.h5')
```

For new projects, the native Keras format (`model.keras`) is also an option.

## Current Limitations

- The notebook's two example prediction cells contain invalid input shapes: one uses an empty array and another uses only four features, while the trained model expects **21 features**. These cells need correction before demonstrating inference.
- No feature scaling is applied in the recorded training pipeline.
- The train/test split is not stratified, despite the imbalanced target classes.
- There is no independent clinical validation or external-dataset evaluation.
- A held-out test accuracy is reported, but class-specific performance and error analysis are not shown.

## Future Improvements

- Add standardization fitted on training data only.
- Use a stratified split and investigate class imbalance.
- Report a classification report, confusion matrix, and per-class recall.
- Compare against baseline machine learning models.
- Implement an inference function that validates the complete 21-feature input.
- Evaluate on independent data before considering any real-world use.

## Suggested Repository Structure

```text
Fetal-Health-Classification-TensorFlow/
├── README.md
├── tf_clsftn.ipynb
├── fetal_health.csv          # Add only if redistribution is permitted
└── model.h5                  # Optional generated model artifact
```

## Repository Description

Fetal health classification using a TensorFlow/Keras artificial neural network trained on CTG features, achieving 92.9% test accuracy in the notebook evaluation.
