ml-defect-classifier

Machine Learning Model for Defect Classification – Proof of Concept

📌 Overview

This project is a proof of concept for applying machine learning to classify test execution failures into predefined defect categories.
It demonstrates:

Parsing and analyzing automated test reports (XML/JSON).

Training a machine learning model (Random Forest) to classify defects.

Exposing a lightweight API for integration into CI/CD pipelines.

(Optional) Automating defect logging into an issue tracker such as Jira.

The goal is to reduce manual triage, improve defect visibility, and catch issues earlier in the release pipeline.

📂 Data Source

Test Results: Sampled from multiple application teams (unit, API, UI automated tests).

Data Extraction: Failed test execution results parsed into CSV (~47k records in the POC).

🧩 Defect Categories

The model classifies failures into categories such as:

Coding Error – Functional

Coding Error – GUI

Coding Error – Interface/API

Test Execution Error

Infrastructure Error

Test Data Error

Deployment Error

Non-Functional Error (e.g., performance, resource usage)

Unknown / Not a Defect (requires human investigation)

🤖 Machine Learning Model

Algorithm: Random Forest Classifier (initial POC).

Features: Test case names, error messages, log content.

Pipeline: Log parsing → feature engineering → model training → classification.

Evaluation: Accuracy measured on historical labeled test logs.

🔗 Workflow Integration

Data Preparation: Convert XML/JSON reports into structured CSV datasets.

Model Training: Train and validate the ML model on labeled data.

API Service: Wrap the model in a Flask API (Dockerized).

CI/CD Integration: Call the API from pipelines to classify test failures in real time.

Defect Logging (Optional): Use REST APIs (e.g., Jira) to automatically log classified defects.

🚀 Tech Stack

Languages/Frameworks: Python, scikit-learn, Flask

Containerization: Docker

Pipelines: Jenkins/GitHub Actions (for CI/CD integration)

Issue Tracking (optional): Jira REST API

📅 Roadmap

MVP (Short Term):

Rule-based + Random Forest classification

Automated defect logging with simple categories

Future Enhancements (Long Term):

Use LLMs for flexible log analysis

Improve accuracy with active learning & semi-supervised approaches

Establish full MLOps pipeline (e.g., MLflow, model versioning, retraining)

✅ Benefits

Automation: Reduces manual triage of test failures.

Consistency: Standardized defect classification across teams.

Speed: Faster defect feedback before UAT.

Scalability: Easy to plug into existing pipelines.
