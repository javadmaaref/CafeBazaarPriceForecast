# main.py

from config import DATA_DIR, CSV_FILES
from data.csv_reader import CSVReader
from data.data_cleaning import find_duplicate_apps, clean_data, prepare_features
from models.improved_xgboost_model import ImprovedXGBoostModel
from models.setup_database import setup_database, get_session
from models.database_models import App
import pandas as pd
from sklearn.utils import resample

def main():
    # Set up database
    engine = setup_database()
    session = get_session(engine)

    # Read CSV files
    csv_reader = CSVReader(data_dir=DATA_DIR, csv_files=CSV_FILES)
    df = csv_reader.read_csv_files()

    if df.empty:
        print("No data available for analysis.")
        return

    print("\nRaw data summary:")
    print(df.describe())

    # Find and remove duplicates
    unique_apps_df = find_duplicate_apps(df)

    # Clean the data
    df_cleaned = clean_data(unique_apps_df)

    # Remove apps with price=0
    df_paid = df_cleaned[df_cleaned['Price'] > 0].copy()
    # print(f"\nNumber of apps before removing free apps: {len(df_cleaned)}")
    # print(f"Number of apps after removing free apps: {len(df_paid)}")
    # print(f"Percentage of paid apps: {len(df_paid) / len(df_cleaned) * 100:.2f}%")
    #
    # print("\nColumn dtypes after cleaning:")
    # print(df_paid.dtypes)
    #
    # print("\nNull values after cleaning:")
    # print(df_paid.isnull().sum())

    print("\nSample data after cleaning:")
    print(df_paid.head(20))

    # Summary statistics for paid apps
    print("\nPaid apps data summary:")
    print(df_paid.describe(include='all'))

    # Save cleaned data to database
    for _, row in df_paid.iterrows():
        app = App(
            category=row['Category'],
            rating=row['Rating'],
            reviews=row['Reviews'],
            size=row['Size'],
            installs=row['Installs'],
            price=row['Price']
        )
        session.add(app)

    session.commit()
    print("Cleaned data for paid apps saved to database.")

    # Features and target
    X = prepare_features(df_paid)
    y = df_paid['Price_Log']

    # Bootstrap resampling
    n_samples = 200000
    X_resampled, y_resampled = resample(X, y, n_samples=n_samples, random_state=42)

    # Final database state
    print("\nFinal database state:")
    print(X.head(20))

    # print("\nFeature information:")
    # print(X.info())
    #
    # print("\nFeature columns:")
    # print(X.columns)

    # Train for price
    xgb_model = ImprovedXGBoostModel()
    trained_model = xgb_model.train(X_resampled, y_resampled)

    # Feature importance
    feature_importance = trained_model.best_estimator_.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    print("\n")
    print("Top 10 most important features for price prediction:")
    print(feature_importance_df.head(10))


if __name__ == "__main__":
    main()
