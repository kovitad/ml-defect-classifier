from src.model_robot import DefectClassifier

# Initialize with configuration paths
classifier = DefectClassifier(config_path='config/robot_config.yaml', mlflow_config_path='config/mlflow_config.yaml')

# Load data, label, and train
data = classifier.load_data('data/raw/robot/robot_test_data.csv')
data = classifier.label_data(data)
X, y = classifier.preprocess_data(data)
X_vectorized = classifier.vectorize_data(X)
X_train, X_test, y_train, y_test = classifier.split_data(X_vectorized, y)

# Train and evaluate
classifier.train_model(X_train, y_train)
classifier.evaluate_model(X_test, y_test)
