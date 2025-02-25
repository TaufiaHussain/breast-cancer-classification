import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib


# Load the dataset
df = pd.read_csv("data/data.csv")

# Drop unnecessary columns
df.drop(columns=['id', 'Unnamed: 32'], inplace=True, errors='ignore')

# Convert 'diagnosis' column: M → 1 (Malignant), B → 0 (Benign)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Check for duplicate rows and remove them
df.drop_duplicates(inplace=True)

# Display basic dataset information
print("Updated Dataset Info:")
print(df.info())

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Separate features (X) and target variable (y)
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Print scaled dataset shape
print(f"\nShape of X_scaled: {X_scaled.shape}")

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Print dataset shapes
print(f"Training Set Shape: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Testing Set Shape: X_test = {X_test.shape}, y_test = {y_test.shape}")

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nBaseline Logistic Regression Accuracy: {accuracy * 100:.2f}%")

# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization strength
    'solver': ['lbfgs', 'liblinear']  # Optimization algorithm
}

# Perform Grid Search
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Predict using the best model
y_pred_best = best_model.predict(X_test)

# Accuracy of the best model
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"\nTuned Logistic Regression Accuracy: {best_accuracy * 100:.2f}%")

# Print best parameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Save the best trained model
joblib.dump(best_model, "best_breast_cancer_model.pkl")

print("\nBest trained model saved as 'best_breast_cancer_model.pkl'")

