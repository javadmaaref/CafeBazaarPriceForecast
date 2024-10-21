# CafeBazaarPriceForecast

This project analyzes data from CafeBazaar, focusing on paid apps and their pricing. It includes data cleaning, preprocessing, database integration, and a machine learning model for price prediction.

## Features

- Data cleaning and preprocessing of CafeBazaar app data
- Database integration with PostgreSQL for storing cleaned data
- XGBoost model for app price prediction
- Feature importance analysis to understand key pricing factors
- Handling of Persian text and numeric data
- Bootstrap resampling for better model training

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/javadmaaref/CafeBazaarPriceForecast.git
   cd CafeBazaarPriceForecast
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   # On macOS/Linux
   source .venv/bin/activate  
   # On Windows
   .venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the PostgreSQL database:
   - Install PostgreSQL if you haven't already.
   - Create a new database named `cafebazaar`.
   - Update the `DATABASE_URI` in `config.py` with your PostgreSQL credentials.

5. Prepare your data:
   - Place your CafeBazaar CSV files in the directory specified by `DATA_DIR` in `config.py`.
   - Update the `CSV_FILES` list in `config.py` with your CSV filenames.

6. Run the main script:
   ```bash
   python main.py
   ```
## Configuration Example

Your `config.py` file should be structured like this:

```python
import os

# Database configuration
DATABASE_URI = 'postgresql://username:password@localhost/database_name'

# Data configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), r'path\to\data\directory')
CSV_FILES = ['file1.csv', 'file2.csv', 'file3.csv']  # Replace with your actual filenames
```

## Project Structure

- `main.py`: Main script to run the entire analysis pipeline
- `config.py`: Configuration file for database URI and data paths
- `models/`:
  - `setup_database.py`: Functions to set up and connect to the database
  - `database_models.py`: SQLAlchemy models for the database tables
  - `improved_xgboost_model.py`: XGBoost model for price prediction
- `data/`:
  - `csv_reader.py`: Class for reading and combining multiple CSV files
  - `data_cleaning.py`: Functions for cleaning and preprocessing the data
- `utils/`: (Create this directory if you have any utility functions)

## Data Cleaning and Preprocessing

The project includes extensive data cleaning and preprocessing steps:
- Converting Persian digits to English digits
- Handling installation numbers with Persian words (e.g., "هزار" for thousand)
- Cleaning and converting price, rating, and review data
- Creating log-transformed features for better model performance
- One-hot encoding of app categories
- Normalization of app names and developer names to better handle duplicates.

## Machine Learning Model

The project uses an XGBoost regressor for price prediction:
- Hyperparameter tuning using RandomizedSearchCV
- Feature scaling with StandardScaler
- Log transformation of the target variable (price)
- Bootstrap resampling to enhance training data size
- Evaluation metrics including MSE, RMSE, and R² score

## Bootstrap Resampling

To improve model performance and reduce overfitting, bootstrap resampling is applied. This technique allows the model to be trained on a larger dataset by generating synthetic samples from the original dataset, which helps to provide more robust training. In this project, 200,000 samples were generated for training, leading to improved evaluation metrics.

## Results and Insights

After running the analysis on the CafeBazaar dataset, we gained several insights:

1. **Data Overview:**
   - The initial dataset contained 178,490 apps.
   - After filtering for paid apps, we were left with 29,514 apps, which is 16.54% of the total.

2. **Paid Apps Statistics:**
   - Average Rating: 2.42 (on a scale of 0-5)
   - Average Price: 5,479.54 (in Tomans)
   - Average Number of Installs: 188.39

3. **Price Prediction Model:**
   - The XGBoost model achieved an R² score of 0.6372 on the test set, indicating that about 63.72% of the variance in app prices can be explained by the model.
   - Root Mean Squared Error (RMSE) on the test set: 0.0663, which suggests that on average, our predictions deviate by about 0.0663 log units from the actual log-transformed prices.

4. **Feature Importance:**
   The top 10 most important features for predicting app prices were:
   1. Personalization Category (شخصی‌سازی): 7.55%
   2. Transportation Category (رفت و آمد): 5.58%
   3. Cooking and Restaurant Category (آشپزی و رستوران): 5.51%
   4. Travel Category (سفر): 4.23%
   5. Strategy Category (استراتژی): 3.70%
   6. Sports Category (ورزشی): 3.57%
   7. Medical Category (پزشکی): 3.49%
   8. Words and Knowledge Category (کلمات و دانستنی‌ها): 3.48%
   9. Action Category (اکشن): 3.33%
   10. Weather Category (آب و هوا): 3.24%

5. **Data Cleaning Improvements:**
   - The data cleaning process now includes normalization of app names and developer names to better handle duplicates.
   - Persian digits are converted to English digits for consistent numeric processing.
   - Special handling for installation numbers with Persian words (e.g., "هزار" for thousand) has been implemented.

## Feature Importance

After training the model, the script outputs the top 10 most important features for price prediction, helping to understand the key factors influencing app pricing on CafeBazaar.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset obtained from [arshin1989/CafeBazaar](https://github.com/arshin1989/CafeBazaar).
- The open-source community for the amazing tools and libraries used in this project.

