import os
import re
import pandas as pd
import joblib
import mlflow
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from common_module.defect_categories import DEFECT_CATEGORIES

# Load configuration
from src.config_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DefectClassifier:
    def __init__(self, config_path, mlflow_config_path):
        # Load configurations
        self.config = load_config(config_path)
        self.mlflow_config = load_config(mlflow_config_path)
        
        self.model = RandomForestClassifier()
        self.vectorizer = TfidfVectorizer()

        mlflow.set_tracking_uri(self.mlflow_config["tracking_uri"])
        mlflow.set_experiment(self.mlflow_config["experiment_name"])

    def load_data(self, file_path):
        """Load and preprocess data."""
        try:
            df = pd.read_csv(file_path)
            logging.info(f'Dataset loaded with {len(df)} rows.')
            df = df.fillna('')
            df['combined_text'] = df['failed_test_step_name'] + ' ' + df['test_case_error']
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    @staticmethod
    def label_message(test_case_error, failed_test_step_name):
        """Label data based on defect patterns."""
        concatenated_error = failed_test_step_name + ' ' + test_case_error
        for pattern in DEFECT_CATEGORIES['coding_error_gui']:
            if re.search(pattern, concatenated_error):
                return 'coding_error_gui'
        for category, patterns_list in DEFECT_CATEGORIES.items():
            if category == 'coding_error_gui':
                continue
            for pattern in patterns_list:
                if re.search(pattern, test_case_error):
                    return category
        return 'error_unknown'

    def label_data(self, df):
        """Apply labeling to the dataset."""
        df['label'] = df.apply(lambda row: self.label_message(row['test_case_error'], row['failed_test_step_name']), axis=1)
        logging.info(f'Labeling complete. Counts: {df["label"].value_counts().to_dict()}')
        return df

    def preprocess_data(self, df):
        """Prepare data for training."""
        X = df['combined_text']
        y = df['label']
        logging.info(f'Number of labeled samples: {len(df)}')
        return X, y

    def vectorize_data(self, X):
        return self.vectorizer.fit_transform(X)

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=self.config['parameters']['test_size'], random_state=42)

    def train_model(self, X_train, y_train):
        """Train the model and track it in MLflow."""
        with mlflow.start_run():
            mlflow.log_params(self.config['parameters'])
            self.model.fit(X_train, y_train)
            mlflow.sklearn.log_model(self.model, "RobotModel")
            logging.info('Model training complete and logged to MLflow.')

    def evaluate_model(self, X_test, y_test):
        """Evaluate model and log metrics in MLflow."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logging.info(f'Accuracy: {accuracy}')
        logging.info(f'Classification report:\n{report}')
        logging.info(f'Confusion matrix:\n{cm}')

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")
        self.plot_confusion_matrix(cm)
        
        return y_pred, cm

    def plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.show()

    def tune_hyperparameters(self, X_train, y_train):
        """Perform hyperparameter tuning."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        logging.info(f'Best parameters found: {grid_search.best_params_}')
        return grid_search.best_params_
