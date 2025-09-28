import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
import logging

def train_robot_model(data, config, mlflow_config):
    mlflow.set_experiment(mlflow_config["experiment_name"])
    label_encoder = LabelEncoder()
    tfidf = TfidfVectorizer(max_features=200)
    svd = TruncatedSVD(n_components=50)
    
    # Data preparation
    y = label_encoder.fit_transform(data['defect_category'])
    X_text = tfidf.fit_transform(data['combined_text'])
    X_reduced = svd.fit_transform(X_text)
    
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_reduced, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=config['parameters']['test_size'])

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=config['parameters']['n_estimators'], max_depth=config['parameters']['max_depth'])
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        
        # Log parameters, metrics, and artifacts
        mlflow.log_param("n_estimators", config['parameters']['n_estimators'])
        mlflow.log_param("max_depth", config['parameters']['max_depth'])
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "RobotModel")
        
        logging.info(f"Robot Model Accuracy: {accuracy}")
        logging.info(f"Classification Report:\n{report}")

        return model, label_encoder, tfidf, svd
