
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_FILE = "student-mat.csv"
TARGET = "G3"

# Exclude previous exam grades to predict performance from other features.
EXCLUDED_COLUMNS = ["G1", "G2", TARGET]


def main():
    print("=" * 55)
    print("STUDENT PERFORMANCE PREDICTOR")
    print("=" * 55)

    try:
        data = pd.read_csv(DATA_FILE, sep=";")
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} was not found.")
        return

    X = data.drop(columns=EXCLUDED_COLUMNS)
    y = data[TARGET]

    categorical_columns = X.select_dtypes(
        include=["str", "object", "category"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            ("numerical", "passthrough", numerical_columns),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"\nTotal students: {len(data)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    print("\nTraining the model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nMODEL EVALUATION")
    print("-" * 30)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.3f}")

    results = pd.DataFrame(
        {
            "Actual Grade": y_test.to_numpy(),
            "Predicted Grade": predictions.round(2),
        }
    )

    results.to_csv("prediction_results.csv", index=False)
    print("\nPrediction results saved to prediction_results.csv")

    print("\nSample predictions:")
    print(results.head(10).to_string(index=False))

    print("\nNote: Predictions are estimates, not guaranteed grades.")

    # Plot actual grades against predicted grades.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.plot([0, 20], [0, 20], linestyle="--")

    plt.xlabel("Actual Grade")
    plt.ylabel("Predicted Grade")
    plt.title("Actual vs Predicted Student Grades")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("actual_vs_predicted.png")
    print("\nChart saved to actual_vs_predicted.png")
    plt.show()


if __name__ == "__main__":
    main()
