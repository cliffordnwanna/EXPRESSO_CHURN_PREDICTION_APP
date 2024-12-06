import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Function to create a directory
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Preprocess data
def preprocess_data(df):
    # Fill missing values
    df_filled = df.drop_duplicates()
    # Encode categorical columns
    label_encoders = {}
    for col in df_filled.columns:
        if df_filled[col].dtype == 'object':
            df_filled[col] = df_filled[col].astype(str).apply(lambda x: x.strip())  # Remove spaces
            le = LabelEncoder()
            df_filled[col] = le.fit_transform(df_filled[col])
            label_encoders[col] = le  # Save label encoders for future use
    return df_filled, label_encoders

# Feature selection based on correlation
def select_features(df, target_column='CHURN', threshold=0.1):
    correlation_matrix = df.corr()
    correlations = correlation_matrix[target_column].abs()
    selected_features = correlations[correlations > threshold].index.tolist()
    selected_features.remove(target_column)
    return selected_features

# Train model
def train_model():
    print("Loading dataset...")
    df = pd.read_csv("data/cleaned_dataset.csv",)
    # encoding="latin1"
    # Preprocess data
    print("Preprocessing data...")
    df_cleaned, label_encoders = preprocess_data(df)

    # Feature selection
    print("Selecting features...")
    selected_features = select_features(df_cleaned)
    print("Selected features:", selected_features)

    # Prepare dataset for training
    X = df_cleaned[selected_features]
    y = df_cleaned['CHURN']

    # Split dataset
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    print("Training model...")
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Test accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)

    # Save model and selected features
    create_directory("models")
    model_path = "models/churn_model.sav"
    features_path = "models/selected_features.pkl"
    label_encoders_path = "models/label_encoders.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    with open(features_path, 'wb') as f:
        pickle.dump(selected_features, f)
    with open(label_encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)

    print(f"Model saved to {model_path}")
    print(f"Selected features saved to {features_path}")
    print(f"Label encoders saved to {label_encoders_path}")

if __name__ == "__main__":
    train_model()
