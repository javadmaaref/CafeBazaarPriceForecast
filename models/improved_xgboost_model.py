# models/improved_xgboost_model.py

import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

class ImprovedXGBoostModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def prepare_data(self, X, y):
        # Select only numeric columns for scaling
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = self.scaler.fit_transform(X[numeric_columns])
        y = np.log1p(y)  # Log transform the target variable
        return X, y

    def train(self, X, y):
        X, y = self.prepare_data(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500, 600, 700],
            'max_depth': [3, 5, 7, 9, 12, 15],
            'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }

        self.model = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            param_distributions=param_dist,
            n_iter=10,
            cv=5,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred_train = self.model.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        r2_train = r2_score(y_train, y_pred_train)

        print("\n")
        print(f"Mean Squared Error (Train): {mse_train}")
        print(f"Root Mean Squared Error (Train): {rmse_train}")
        print(f"R2 Score (Train): {r2_train}")

        print("\n")
        print("Best parameters:", self.model.best_params_)

        y_pred = self.model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("\n")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"R2 Score: {r2}")


        return self.model

    def predict(self, X):
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = self.scaler.transform(X[numeric_columns])
        return np.expm1(self.model.predict(X))  # Inverse log transform