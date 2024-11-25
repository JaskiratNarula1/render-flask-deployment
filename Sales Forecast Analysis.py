import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate sample sales data
data = {
    "Month": pd.date_range(start="2023-01-01", periods=24, freq="M"),
    "Sales": [
        200 + np.random.randint(-20, 20) + i * 10 for i in range(24)
    ],
}
sales_data = pd.DataFrame(data)
sales_data.set_index("Month", inplace=True)

# Plot the original data
plt.figure(figsize=(10, 6))
plt.plot(sales_data.index, sales_data["Sales"], label="Actual Sales", marker="o")
plt.title("Sales Data")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()

# Prepare data for training
X = np.arange(len(sales_data)).reshape(-1, 1)  # Feature: Time (months)
y = sales_data["Sales"].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Error metrics
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.plot(X, y, label="Actual Sales", marker="o")
plt.plot(X_train, y_pred_train, label="Train Predictions", linestyle="--")
plt.plot(X_test, y_pred_test, label="Test Predictions", linestyle="--")
plt.title("Sales Predictions")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()

# Future predictions
future_months = 12
X_future = np.arange(len(sales_data), len(sales_data) + future_months).reshape(-1, 1)
y_future = model.predict(X_future)

# Plot future predictions
plt.figure(figsize=(10, 6))
plt.plot(sales_data.index, sales_data["Sales"], label="Actual Sales", marker="o")
plt.plot(
    pd.date_range(start=sales_data.index[-1], periods=future_months + 1, freq="M")[1:],
    y_future,
    label="Future Predictions",
    linestyle="--",
)
plt.title("Future Sales Predictions")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()
