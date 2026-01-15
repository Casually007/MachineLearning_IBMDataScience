# One-vs-All and One-vs-One Multi class classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

def ova_model_train_eval(X_train, X_test, y_train, y_test, feature_columns):
    # Training logistic regression model using One-vs-All (default)
    model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
    model_ova.fit(X_train, y_train)
    # Predictions
    y_pred_ova = model_ova.predict(X_test)
    # Evaluation metrics for OvA
    print("One-vs-All (OvA) Strategy")
    print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")
    # Feature importance barplot
    feature_importance = np.mean(np.abs(model_ova.coef_), axis=0)
    plt.barh(feature_columns, feature_importance)
    plt.title("Feature Importance (One-vs-All)")
    plt.xlabel("Importance")
    plt.show()

def ovo_model_train_eval(X_train, X_test, y_train, y_test, feature_columns):
        # Training logistic regression model using One-vs-One
    model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
    model_ovo.fit(X_train, y_train)
    # Predictions
    y_pred_ovo = model_ovo.predict(X_test)

    # Evaluation metrics for OvO
    print("One-vs-One (OvO) Strategy")
    print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")

    # Collect all coefficients from each underlying binary classifier
    coefs = np.array([est.coef_[0] for est in model_ovo.estimators_])

    # Now take the mean across all those classifiers
    feature_importance = np.mean(np.abs(coefs), axis=0)

    # Plot feature importance
    plt.barh(feature_columns, feature_importance)
    plt.title("Feature Importance (One-vs-One)")
    plt.xlabel("Importance")
    plt.show()


def obesity_risk_pipeline(data_path, test_size=0.2, model_type='ova'):
    ## Load data
    data = pd.read_csv(data_path)
    data.head()

    ## Preprocessing
    # Standardizing continuous numerical features
    continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[continuous_columns])

    # Converting to a DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

    # Combining with the original dataset
    scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

    # Identifying categorical columns
    categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('NObeyesdad')  # Exclude target column

    # Applying one-hot encoding
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

    # Converting to a DataFrame
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

    # Combining with the original dataset
    prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

    # Encoding the target variable
    prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
    prepped_data.head()

    # Preparing final dataset
    X = prepped_data.drop('NObeyesdad', axis=1)
    y = prepped_data['NObeyesdad']
    
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Model training and evaluation
    if model_type == 'ova':
        ova_model_train_eval(X_train, X_test, y_train, y_test, X.columns)
    elif model_type == 'ovo':
        ovo_model_train_eval(X_train, X_test, y_train, y_test, X.columns)
    else:
        print(f"Incorrect model type. Please enter ova (One vs All) or ovo (One vs One)")

def main():
    file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
    obesity_risk_pipeline(file_path, test_size=0.1, model_type="ova")

if __name__ == '__main__':
    main()
