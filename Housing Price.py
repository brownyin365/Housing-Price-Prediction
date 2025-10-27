# ============================================================
# 📦 Import Libraries
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 📂 Load and Inspect Data
# ============================================================

data = pd.read_csv('house_price_regression_dataset.csv')

print("📊 Dataset Preview:")
print(data.head(), "\n")

# ============================================================
# ⚙️ Prepare Features and Target
# ============================================================

X = data.drop('House_Price', axis=1)
y = data['House_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 🧠 Train Model
# ============================================================

model = LinearRegression()
model.fit(X_train, y_train)

# ============================================================
# 🔮 Make Predictions
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# 📏 Evaluate Model
# ============================================================

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📈 Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.4f}\n")

# ============================================================
# 🔍 Display Sample Predictions
# ============================================================

comparison = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred[:10]
})
print("🔍 Sample Predictions (Actual vs Predicted):")
print(comparison.to_string(index=False))

# ============================================================
# 📊 Graphical Visualizations
# ============================================================

# 1️⃣ Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("🔗 Feature Correlation Heatmap")
plt.show()

# 2️⃣ Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='dodgerblue', alpha=0.7, edgecolor='k')
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("🏠 Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# 3️⃣ Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.title("📉 Distribution of Residuals (Errors)")
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.show()

# 4️⃣ Feature Importance (Coefficient Magnitude)
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
plt.title("📊 Feature Importance (Linear Regression Coefficients)")
plt.show()









# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # ============================================================
# # Load and Inspect Data
# # ============================================================

# # Load the dataset
# data = pd.read_csv('house_price_regression_dataset.csv')

# # Display the first few rows to understand the data
# print("📊 Dataset Preview:")
# print(data.head(), "\n")

# # ============================================================
# # Prepare Features and Target
# # ============================================================

# # Separate features (X) and target variable (y)
# X = data.drop('House_Price', axis=1)
# y = data['House_Price']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # ============================================================
# # Train Model
# # ============================================================

# # Initialize and train the Linear Regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # ============================================================
# # Make Predictions
# # ============================================================

# y_pred = model.predict(X_test)

# # ============================================================
# # Evaluate Model
# # ============================================================

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("📈 Model Evaluation:")
# print(f"Mean Squared Error (MSE): {mse:.2f}")
# print(f"R-squared (R²): {r2:.4f}\n")

# # ============================================================
# # Display Sample Predictions
# # ============================================================

# print("🔍 Sample Predictions (Actual vs Predicted):")
# comparison = pd.DataFrame({
#     'Actual': y_test.values[:5],
#     'Predicted': y_pred[:5]
# })
# print(comparison.to_string(index=False))
