import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load trained model
model = joblib.load("model.joblib")
coef = model.coef_
intercept = model.intercept_

# Save raw parameters
joblib.dump({'coef': coef, 'intercept': intercept}, "unquant_params.joblib")

# Quantize
coef_min, coef_max = coef.min(), coef.max()
coef_q = np.round((coef - coef_min) / (coef_max - coef_min) * 65535).astype(np.uint16)

joblib.dump({
    "coef": coef_q,
    "coef_min": coef_min,
    "coef_max": coef_max,
    "intercept": intercept
}, "quant_params.joblib")

# Dequantize
coef_dq = coef_q.astype(np.float32) / 65535 * (coef_max - coef_min) + coef_min

# Predict
data = fetch_california_housing()
X = data.data
y = data.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred_quant = X_test @ coef_dq + intercept
r2 = r2_score(y_test, y_pred_quant)
mse = mean_squared_error(y_test, y_pred_quant)

print(f"R2 score with quantized params: {r2:.4f}")
print(f"MSE with quantized params: {mse:.4f}")
