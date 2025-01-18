import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def load_german_credit_data():
    dataset_name = "GERMAN"
    
    original_data = pd.read_csv('../data/german_credit_data_updated.csv')

    # Dataset overview - German Credit Risk (from Kaggle):
    # 1. Age (numeric)
    # 2. Sex (text: male, female)
    # 3. Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
    # 4. Housing (text: own, rent, or free)
    # 5. Saving accounts (text - little, moderate, quite rich, rich)
    # 6. Checking account (numeric, in DM - Deutsch Mark)
    # 7. Credit amount (numeric, in DM)
    # 8. Duration (numeric, in month)
    # 9. Purpose (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)

    preprocessed_data = original_data.copy()

    # For savings and checking accounts, we will replace the missing values with 'none':
    preprocessed_data['Saving accounts'].fillna('none', inplace=True)
    preprocessed_data['Checking account'].fillna('none', inplace=True)

    # Dropping index column:
    preprocessed_data.drop(columns=['Unnamed: 0'], inplace=True)

    # Using pd.dummies to one-hot-encode the categorical features
    preprocessed_data["Job"] = preprocessed_data["Job"].map({0: 'unskilled_nonresident', 1: 'unskilled_resident',
                                                            2: 'skilled', 3: 'highlyskilled'})

    categorical_features = preprocessed_data.select_dtypes(include='object').columns
    numerical_features = preprocessed_data.select_dtypes(include='number').columns.drop('Credit Risk')

    preprocessed_data = pd.get_dummies(preprocessed_data, columns=categorical_features, dtype='int64')

    # Remapping the target variable to 0 and 1:
    preprocessed_data['Credit Risk'] = preprocessed_data['Credit Risk'].map({1: 0, 2: 1})

    # Make sure all column names are valid python identifiers (important for pd.query() calls):
    preprocessed_data.columns = preprocessed_data.columns.str.replace(' ', '_')
    preprocessed_data.columns = preprocessed_data.columns.str.replace('/', '_')

    y = preprocessed_data['Credit_Risk']
    X = preprocessed_data.drop(columns='Credit_Risk')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred)}')

    return dataset_name, preprocessed_data, categorical_features, X, y, X_train, X_test, y_train, y_test, clf


def load_taiwan_data():
    dataset_name = "TAIWAN"
    
    raw = pd.read_excel("../data/taiwan.xls", header=1)
    raw = raw.drop(columns=['ID'])

    preprocessed_data = raw.copy()

    # Mapping categorical veriables:
    preprocessed_data['SEX'] = preprocessed_data['SEX'].map({1: 'male', 2: 'female'})
    preprocessed_data['EDUCATION'] = preprocessed_data['EDUCATION'].map({1: 'graduate_school', 2: 'university', 3: 'high_school', 4: 'others'})
    preprocessed_data['MARRIAGE'] = preprocessed_data['MARRIAGE'].map({1: 'married', 2: 'single', 3: 'others'})

    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

    # Set all other columns to float
    for column in preprocessed_data.columns:
        if column not in categorical_features:
            preprocessed_data[column] = preprocessed_data[column].astype(int)

    # One-hot encoding
    preprocessed_data = pd.get_dummies(preprocessed_data, columns=categorical_features, dtype='int64', drop_first=True)

    # Stratified sampling
    preprocessed_data, _ = train_test_split(preprocessed_data, test_size=0.9, stratify=preprocessed_data['default payment next month'], random_state=42)

    y = preprocessed_data['default payment next month']
    X = preprocessed_data.drop(columns='default payment next month')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred)}')

    return dataset_name, preprocessed_data, categorical_features, X, y, X_train, X_test, y_train, y_test, clf

def load_pakdd2010_data():
    dataset_name = "PAKDD2010"
    
    raw = pd.read_csv("../data/PAKDD2010.tsv", sep="\t", encoding="unicode_escape", header=None)

    raw.columns = [
        "ID_CLIENT", "CLERK_TYPE", "PAYMENT_DAY", "APPLICATION_SUBMISSION_TYPE", "QUANT_ADDITIONAL_CARDS", 
        "POSTAL_ADDRESS_TYPE", "SEX", "MARITAL_STATUS", "QUANT_DEPENDANTS", "UNKNOWN",
        "STATE_OF_BIRTH", "CITY_OF_BIRTH", "NACIONALITY", "RESIDENCIAL_STATE", "RESIDENCIAL_CITY", 
        "RESIDENCIAL_BOROUGH", "FLAG_RESIDENCIAL_PHONE", "RESIDENCIAL_PHONE_AREA_CODE", "RESIDENCE_TYPE", 
        "MONTHS_IN_RESIDENCE", "FLAG_MOBILE_PHONE", "FLAG_EMAIL", "PERSONAL_MONTHLY_INCOME", "OTHER_INCOMES", 
        "FLAG_VISA", "FLAG_MASTERCARD", "FLAG_DINERS", "FLAG_AMERICAN_EXPRESS", "FLAG_OTHER_CARDS", 
        "QUANT_BANKING_ACCOUNTS", "QUANT_SPECIAL_BANKING_ACCOUNTS", "PERSONAL_ASSETS_VALUE", "QUANT_CARS", 
        "COMPANY", "PROFESSIONAL_STATE", "PROFESSIONAL_CITY", "PROFESSIONAL_BOROUGH", "FLAG_PROFESSIONAL_PHONE", 
        "PROFESSIONAL_PHONE_AREA_CODE", "MONTHS_IN_THE_JOB", "PROFESSION_CODE", "OCCUPATION_TYPE", 
        "MATE_PROFESSION_CODE", "EDUCATION_LEVEL", "FLAG_HOME_ADDRESS_DOCUMENT", "FLAG_RG", "FLAG_CPF", 
        "FLAG_INCOME_PROOF", "PRODUCT", "FLAG_ACSP_RECORD", "AGE", "RESIDENCIAL_ZIP_3", "PROFESSIONAL_ZIP_3", 
        "TARGET_LABEL_BAD"
    ]

    raw.drop(columns=["ID_CLIENT", "CLERK_TYPE", "QUANT_ADDITIONAL_CARDS"], inplace=True)

    # Taking a subset of columns:
    numeric_features = [
        "QUANT_DEPENDANTS", "RESIDENCIAL_PHONE_AREA_CODE", "MONTHS_IN_RESIDENCE", "PERSONAL_MONTHLY_INCOME", "QUANT_CARS", "PROFESSIONAL_PHONE_AREA_CODE",
        "MONTHS_IN_THE_JOB", "RESIDENCIAL_ZIP_3", "PROFESSIONAL_ZIP_3", "AGE"
    ]

    categorical_features = [
        "SEX", "MARITAL_STATUS", "EDUCATION_LEVEL", "FLAG_RESIDENCIAL_PHONE",
        "FLAG_MOBILE_PHONE", "COMPANY", "FLAG_PROFESSIONAL_PHONE", "OCCUPATION_TYPE"
    ]


    features = numeric_features + categorical_features
    preprocessed_data = raw[features + ["TARGET_LABEL_BAD"]]

    # Handling missing values. For numeric features, we will use the median value. For categorical features, we will use the mode.
    # Handle invalid values:
    preprocessed_data.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    preprocessed_data.replace(r'(?i)\bnull\b', np.nan, regex=True, inplace=True)
    preprocessed_data.replace("#DIV/0!", np.nan, inplace=True)

    # Convert all numeric features to float.
    for feature in numeric_features:
        preprocessed_data[feature] = preprocessed_data[feature].astype(float)

    for feature in numeric_features:
        preprocessed_data[feature] = preprocessed_data[feature].fillna(preprocessed_data[feature].median())

    for feature in categorical_features:
        preprocessed_data[feature] = preprocessed_data[feature].fillna(preprocessed_data[feature].mode().values[0])

        # if datatype is not object, convert to int
        if preprocessed_data[feature].dtype != 'object':
            preprocessed_data[feature] = preprocessed_data[feature].astype(int)

    # Handling categorical features
    # One-hot encoding
    preprocessed_data = pd.get_dummies(preprocessed_data, columns=categorical_features, drop_first=True, dtype='int64')

    # Stratified sampling
    preprocessed_data, _ = train_test_split(preprocessed_data, test_size=0.94, stratify=preprocessed_data['TARGET_LABEL_BAD'], random_state=42)

    y = preprocessed_data["TARGET_LABEL_BAD"]
    X = preprocessed_data.drop(columns='TARGET_LABEL_BAD')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred)}')
    
    return dataset_name, preprocessed_data, categorical_features, X, y, X_train, X_test, y_train, y_test, clf

def load_wdbc_dataset():
    raw_data = pd.read_csv('../data/wdbc.csv')
    # Specify header:
    raw_data.columns = ['ID', 'Diagnosis'] + [str for i in range(1, 4) for str in [f"radius_{i}", f"texture_{i}", f"perimeter_{i}", f"area_{i}", f"smoothness_{i}", f"compactness_{i}", f"concavity_{i}", f"concave_points_{i}", f"symmetry_{i}", f"fractal_dimension_{i}"]]

    preprocessed_data = raw_data.drop(columns=['ID'])

    preprocessed_data['Diagnosis'] = preprocessed_data['Diagnosis'].map({'M': 1, 'B': 0})
    preprocessed_data.rename(columns={'Diagnosis': 'DiagnosisIsMalignant'}, inplace=True)

    categorical_features = []
    
    X = preprocessed_data.drop(columns=['DiagnosisIsMalignant'])
    y = preprocessed_data['DiagnosisIsMalignant']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
    
    return "WDBC", preprocessed_data, categorical_features, X, y, X_train, X_test, y_train, y_test, clf