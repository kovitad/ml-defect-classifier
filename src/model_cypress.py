import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
from src.config_loader import load_config  # Ensure config loader function is available

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class CypressDefectClassifier:
    def __init__(self, config_path, mlflow_config_path):
        # Load configuration files
        self.config = load_config(config_path)
        self.mlflow_config = load_config(mlflow_config_path)

        # Set up MLflow
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        mlflow.set_experiment(self.mlflow_config['experiment_name'])

        # Model and transformation components
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=self.config['tfidf_max_features'])
        self.svd_transformer = TruncatedSVD(n_components=self.config['svd_components'])
        self.classifier = RandomForestClassifier(
            n_estimators=self.config['model']['n_estimators'],
            random_state=self.config['model']['random_state'],
            class_weight=self.config['model']['class_weight']
        )

    def load_data(self):
        """Load and concatenate all CSV files in the specified folder."""
        folder_path = self.config['data']['folder_path']
        df_all = pd.DataFrame()
        logging.info(f'Reading all CSV files from {folder_path}...')

        if not os.path.exists(folder_path):
            logging.error(f"Directory {folder_path} does not exist.")
            raise FileNotFoundError(f"Directory {folder_path} does not exist.")

        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                logging.debug(f'Reading file: {file_path}')
                df = pd.read_csv(file_path)
                df['file'] = filename
                df_all = pd.concat([df_all, df], ignore_index=True)

        # Drop rows with missing values in critical columns
        df_all.dropna(subset=['defect_category', 'cypress_code'], inplace=True)
        
        # Combine error fields into a single column
        df_all['Error_Info'] = df_all['error_message'].astype(str) + " " + df_all['error_stack'].astype(str)
        
        return df_all

    def preprocess_data(self, df):
        """Label encode the defect category and vectorize features."""
        # Encode defect categories
        y = self.label_encoder.fit_transform(df['defect_category'])

        # TF-IDF and SVD on text features
        X_error_info = self.tfidf_vectorizer.fit_transform(df['Error_Info'])
        X_cypress_code = self.tfidf_vectorizer.fit_transform(df['cypress_code'])
        
        X_error_info_reduced = self.svd_transformer.fit_transform(X_error_info)
        X_cypress_code_reduced = self.svd_transformer.fit_transform(X_cypress_code)
        
        # Combine text-based and numeric features
        X_numeric = df[self.config['data']['numeric_columns']].values
        X_combined = np.hstack([X_error_info_reduced, X_cypress_code_reduced, X_numeric])
        
        return X_combined, y

    def balance_data(self, X, y):
        """Apply SMOTE to handle class imbalance."""
        smote = SMOTE(random_state=self.config['smote']['random_state'])
        return smote.fit_resample(X, y)

    def split_data(self, X, y):
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=self.config['split']['test_size'], random_state=42)

    def train_and_log_model(self, X_train, y_train, X_test, y_test):
        """Train the model, evaluate, and log details to MLflow."""
        with mlflow.start_run():
            # Train model
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logging.info(f"Model Accuracy on Test Set: {accuracy}")
            
            # Generate and log classification report
            unique_labels = np.unique(y_test)
            target_names = self.label_encoder.inverse_transform(unique_labels)
            report = classification_report(y_test, y_pred, target_names=target_names)
            logging.info(f"Classification Report:\n{report}")

            # Log parameters, metrics, and artifacts to MLflow
            mlflow.log_params(self.config['model'])
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_text(report, "classification_report.txt")
            mlflow.sklearn.log_model(self.classifier, "CypressRandomForestClassifier")
            
            # Save and log transformation objects and model
            self._save_and_log_artifacts()

    def _save_and_log_artifacts(self):
        """Save the trained model and transformation objects, and log them to MLflow."""
        model_dir = self.config['model']['output_dir']
        os.makedirs(model_dir, exist_ok=True)

        # Define file paths
        model_path = os.path.join(model_dir, self.config['model']['model_filename'])
        label_encoder_path = os.path.join(model_dir, self.config['model']['label_encoder_filename'])
        tfidf_vectorizer_path = os.path.join(model_dir, self.config['model']['tfidf_vectorizer_filename'])
        svd_transformer_path = os.path.join(model_dir, self.config['model']['svd_transformer_filename'])

        # Save artifacts
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.label_encoder, label_encoder_path)
        joblib.dump(self.tfidf_vectorizer, tfidf_vectorizer_path)
        joblib.dump(self.svd_transformer, svd_transformer_path)

        # Log artifacts to MLflow
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(label_encoder_path)
        mlflow.log_artifact(tfidf_vectorizer_path)
        mlflow.log_artifact(svd_transformer_path)

        # Clean up local files
        os.remove(model_path)
        os.remove(label_encoder_path)
        os.remove(tfidf_vectorizer_path)
        os.remove(svd_transformer_path)

    def run_pipeline(self):
        """Execute the full training pipeline."""
        df = self.load_data()
        X, y = self.preprocess_data(df)
        X_resampled, y_resampled = self.balance_data(X, y)
        X_train, X_test, y_train, y_test = self.split_data(X_resampled, y_resampled)
        self.train_and_log_model(X_train, y_train, X_test, y_test)

# Example usage in main.py
if __name__ == "__main__":
    config_path = "config/cypress_config.yaml"
    mlflow_config_path = "config/mlflow_config.yaml"
    cypress_classifier = CypressDefectClassifier(config_path=config_path, mlflow_config_path=mlflow_config_path)
    cypress_classifier.run_pipeline()
