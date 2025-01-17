import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

def load_data(train_path, test_path):
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    return train_data, test_data

def preprocess_data(train_data, selected_features):
    X = train_data[selected_features]
    y = train_data["SalePrice"]

    # Handle missing values
    X.loc[:, "GarageArea"] = X.groupby("YearBuilt")["GarageArea"].transform("median").astype(int)
    X = X.fillna(X.median())

    # Feature scaling
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=selected_features)

    return X, y, scaler

def feature_selection(X_train, y_train, selected_features):
    linear_model = LinearRegression()
    rfe_selector = RFE(estimator=linear_model, n_features_to_select=3, step=1)
    X_train_selected = rfe_selector.fit_transform(X_train, y_train)
    selected_features_rfe = [selected_features[i] for i in range(len(selected_features)) if rfe_selector.support_[i]]
    print(f"Selected features by RFE: {selected_features_rfe}")
    return X_train_selected, rfe_selector

def train_models(X_train, y_train, X_train_selected):
    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train_selected, y_train)

    # Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Ridge Regression with GridSearchCV
    ridge = Ridge()
    param_grid = {"alpha": [0.1, 1.0, 10.0]}
    ridge_grid = GridSearchCV(ridge, param_grid, cv=5)
    ridge_grid.fit(X_train, y_train)
    best_ridge = ridge_grid.best_estimator_

    return linear_model, rf_model, best_ridge

def evaluate_models(models, X_valid, X_valid_selected, y_valid):
    linear_model, rf_model, ridge_model = models

    # Predictions
    linear_pred = linear_model.predict(X_valid_selected)
    rf_pred = rf_model.predict(X_valid)
    ridge_pred = ridge_model.predict(X_valid)

    # Evaluation
    results = {
        "Linear Regression": {
            "MSE": mean_squared_error(y_valid, linear_pred),
            "R2": r2_score(y_valid, linear_pred)
        },
        "Random Forest": {
            "MSE": mean_squared_error(y_valid, rf_pred),
            "R2": r2_score(y_valid, rf_pred)
        },
        "Ridge Regression": {
            "MSE": mean_squared_error(y_valid, ridge_pred),
            "R2": r2_score(y_valid, ridge_pred)
        }
    }

    return results, (linear_pred, rf_pred, ridge_pred)

def visualize_predictions(y_valid, predictions, model_names):
    for pred, name in zip(predictions, model_names):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_valid, pred, alpha=0.7, label=f"{name} Predictions")
        plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], color="red", linestyle="--", label="Perfect Fit")
        plt.xlabel("Actual SalePrice")
        plt.ylabel("Predicted SalePrice")
        plt.title(f"{name}: Actual vs Predicted SalePrice")
        plt.legend()
        plt.grid()
        plt.show()

def save_predictions(test_data, selected_features, rf_model, scaler, output_path):
    test_data[selected_features] = test_data[selected_features].fillna(test_data[selected_features].median())
    test_data_scaled = pd.DataFrame(scaler.transform(test_data[selected_features]), columns=selected_features)
    test_preds = rf_model.predict(test_data_scaled)
    submission = pd.DataFrame({"Id": test_data["Id"], "SalePrice": test_preds})
    submission.to_csv(output_path, index=False)
    print(f"Test predictions saved to {output_path}")

def main():
    # File paths
    train_path = "/kaggle/input/home-data-for-ml-course/train.csv"
    test_path = "/kaggle/input/home-data-for-ml-course/test.csv"
    output_path = "submission.csv"

    # Features
    selected_features = ["OverallQual", "GrLivArea", "GarageArea", "YearBuilt"]

    # Load data
    train_data, test_data = load_data(train_path, test_path)

    # Preprocess data
    X, y, scaler = preprocess_data(train_data, selected_features)

    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature selection
    X_train_selected, rfe_selector = feature_selection(X_train, y_train, selected_features)
    X_valid_selected = rfe_selector.transform(X_valid)

    # Train models
    models = train_models(X_train, y_train, X_train_selected)

    # Evaluate models
    results, predictions = evaluate_models(models, X_valid, X_valid_selected, y_valid)
    for model, metrics in results.items():
        print(f"{model} - MSE: {metrics['MSE']}, R2: {metrics['R2']}")

    # Visualize predictions
    visualize_predictions(y_valid, predictions, ["Linear Regression", "Random Forest", "Ridge Regression"])

    # Save test predictions
    save_predictions(test_data, selected_features, models[1], scaler, output_path)

if __name__ == "__main__":
    main()
