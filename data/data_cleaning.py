# data/data_cleaning.py

import os
import pandas as pd
import numpy as np

def normalize_text(text):
    # Normalize Persian text by replacing the zero-width space with a regular space
    if isinstance(text, str):  # Check if text is a string
        return text.replace('\u200c', ' ').strip()
    return ''  # Return an empty string for non-string values


def find_duplicate_apps(df):
    # Normalize App and Developer columns
    df['App_normalized'] = df['App'].apply(normalize_text)
    df['Developer_normalized'] = df['Developer'].apply(normalize_text)

    # Remove rows where Developer is "#NAME?"
    df = df[df['Developer_normalized'] != '#NAME?']

    # Remove duplicates and keep only one entry
    df_unique = df.drop_duplicates(subset=['App_normalized', 'Developer_normalized'], keep='first')

    return df_unique

def convert_persian_digits(text):
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    english_digits = '0123456789'
    translation_table = str.maketrans(persian_digits, english_digits)
    return str(text).translate(translation_table)


def convert_installs(value):
    value = convert_persian_digits(value)
    value = value.replace('+', '').replace(',', '').replace(' ', '').lower()
    if 'هزار' in value:
        return int(float(value.replace('هزار', '')) * 1000)
    elif 'میلیون' in value:
        return int(float(value.replace('میلیون', '')) * 1000000)
    elif 'میلیارد' in value:
        return int(float(value.replace('میلیارد', '')) * 1000000000)
    else:
        return int(value) if value.isdigit() else 0


def clean_data(df):
    # Select only the columns we need
    df = df[['Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Price']]

    # Convert Persian digits to English digits
    for col in df.columns:
        df[col] = df[col].apply(convert_persian_digits)

    # Convert 'Price' to numeric, removing any non-numeric characters
    df['Price'] = df['Price'].replace(r'[^0-9.]', '', regex=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Convert 'Installs' to numeric, handling the '+' sign, commas, and Persian words
    df['Installs'] = df['Installs'].apply(convert_installs)

    # Convert 'Size' to numeric (in MB)
    df['Size'] = df['Size'].replace(r'[^0-9.]', '', regex=True).astype(float)

    # Convert 'Rating' to numeric
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

    # Convert 'Reviews' to numeric
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')

    # Handle missing values
    df['Installs'] = df['Installs'].fillna(0)
    df['Price'] = df['Price'].fillna(0)
    df['Reviews'] = df['Reviews'].fillna(0)
    df['Size'] = df['Size'].fillna(df['Size'].median())
    df['Rating'] = df['Rating'].fillna(df['Rating'].median())

    # Drop rows with missing values in critical columns
    df = df.dropna()

    return df


def prepare_features(df):
    # Mapping for categories that should be merged
    category_mapping = {
        'شبکه های اجتماعی': 'شبکه‌های اجتماعی',
        'شبیه سازی': 'شبیه‌سازی',
        'شخصی سازی': 'شخصی‌سازی',
        'کتاب ها و مطبوعات': 'کتاب‌ها و مطبوعات',
        'کلمات و دانستنی ها': 'کلمات و دانستنی‌ها'
    }

    # Apply the mapping to standardize categories
    df['Category'] = df['Category'].replace(category_mapping)

    # Using log transformation to reduce skewness in Price, Reviews, Size, and Installs
    df['Price_Log'] = np.log1p(df['Price'])

    # Create dummy variables for categories
    category_dummies = pd.get_dummies(df['Category'], prefix='Cat')

    # Combine original dataframe with dummy variables
    df_encoded = pd.concat([df, category_dummies], axis=1)

    # Create additional features (Reviews, Size, and Installs)
    df_encoded['Reviews_Log'] = np.log1p(df_encoded['Reviews'])
    df_encoded['Installs_Log'] = np.log1p(df_encoded['Installs'])
    df_encoded['Size_Log'] = np.log1p(df_encoded['Size'])

    # Drop unnecessary columns
    df_encoded = df_encoded.drop(['Price', 'Reviews', 'Installs', 'Size', 'Category', 'Price_Log'], axis=1)

    return df_encoded
