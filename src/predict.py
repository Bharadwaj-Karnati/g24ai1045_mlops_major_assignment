import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def main():
    model = joblib.load("model.joblib")

    data = fetch_california_housing()
    X, y = data.data, data.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    print(f"{'Predicted':>12} | {'True':>8}")
    print("-" * 25)
    for pred, true in zip(y_pred[:10], y_test[:10]):
        print(f"{pred:12.4f} | {true:8.4f}")

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("\nModel Evaluation Metrics:")
    print(f"R2 score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")

if __name__ == "__main__":
    main()
