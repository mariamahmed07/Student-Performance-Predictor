
# Student Performance Predictor

A machine learning project that predicts students' final mathematics grades using student demographic, family, social, and school-related features.

## Project Overview

This project uses a Random Forest Regressor to estimate students' final mathematics grades (`G3`) based on the UCI Student Performance dataset.

Previous exam grades (`G1` and `G2`) are excluded to evaluate predictions using other available student features.

## Features

- Data loading and preprocessing with Pandas
- Categorical feature encoding using OneHotEncoder
- Machine learning pipeline using Scikit-learn
- Model evaluation using MAE, RMSE, and R²
- Actual-versus-predicted visualization
- Export of predictions to a CSV file

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Dataset

The project uses `student-mat.csv`, containing student information and final mathematics grades.

Dataset source: [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

## Model Evaluation

The current test run produced the following results:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 2.97 |
| Root Mean Squared Error (RMSE) | 3.75 |
| R² Score | 0.313 |

These metrics are based on one 80/20 train-test split with a fixed random seed. Results may vary with different data splits or model configurations.

## Generated Files

- `prediction_results.csv` — Actual and predicted grades for the test set.
- `actual_vs_predicted.png` — Scatter plot comparing actual and predicted grades.

## How to Run

1. Clone or download this repository.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure `student-mat.csv` is in the project directory.
4. Run:

   ```bash
   python main.py
   ```

## Limitations

- The dataset contains 395 students and may not represent other student populations.
- Predictive performance is currently limited.
- Predictions are estimates and should not be used as definitive judgments about individual students.

## Future Improvements

- Compare multiple regression algorithms.
- Tune model hyperparameters.
- Use cross-validation for more robust evaluation.
- Analyze feature importance.
- Improve prediction performance.

## Disclaimer

This project is for educational purposes and is not intended to determine a student's ability or future academic success.
