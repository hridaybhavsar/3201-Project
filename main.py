import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("Loading dataset...")
# 1. Load the Dataset
df = pd.read_csv('amazon_sales_dataset.csv')

# --- TASK 1: VISUALIZATIONS ---
print("Generating and saving visualizations...")
plt.figure(figsize=(12, 5))

# Plot 1: Distribution of Outcome Variable (Total Revenue)
plt.subplot(1, 2, 1)
sns.histplot(df['total_revenue'], bins=50, kde=True, color='blue')
plt.title('Distribution of Total Revenue')
plt.xlabel('Total Revenue')
plt.ylabel('Frequency')

# Plot 2: Input vs. Outcome (Price vs. Total Revenue)
plt.subplot(1, 2, 2)
sns.scatterplot(x=df['price'], y=df['total_revenue'], alpha=0.5, color='orange')
plt.title('Price vs. Total Revenue')
plt.xlabel('Price')
plt.ylabel('Total Revenue')

plt.tight_layout()
# Save the plots to your folder instead of just displaying them
plt.savefig('task1_visualizations.png')
print("Saved 'task1_visualizations.png' to your folder.")
# -----------------------------------------------

# --- TASK 2: MACHINE LEARNING IMPLEMENTATION ---
print("Preprocessing data and training model...")
# Drop rows with missing values in our target or critical features
features = ['price', 'discount_percent', 'rating', 'product_category', 'customer_region']
target = 'total_revenue'
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# Define categorical and numerical columns
numeric_features = ['price', 'discount_percent', 'rating']
categorical_features = ['product_category', 'customer_region']

# Create a preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Model Implementation Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)) # n_jobs=-1 uses all CPU cores
])

# Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluation
print("Evaluating model...")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2):            {r2:.4f}")