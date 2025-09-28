# ml-defect-classifier
Title: Machine Learning Model for Defect Classification - Knowledge Transfer Documentation

Overview This document provides a comprehensive overview of the Machine Learning implementation for classifying test execution failures into predefined defect categories, automating defect logging into Jira, and integrating the process with Jenkins and TestOps for seamless operation. This documentation serves as a knowledge transfer guide for the team to understand the workflow, processes, and the machine learning model used for defect classification.

1. Data Source

Test Results: Extracted from 2,627 output.xml files across 12 application teams (KBIZ, PHUB, DA FUND, DA Custodian, WGW, WSA, etc.).

Data Extraction: Failed test execution results were extracted and compiled into a CSV file, containing 47,710 records.

2. Pattern Analysis and Defect Classification

Manual Classification: Collaborated with test automation engineers to identify and classify test failures.

Defect Categories: Classify defects into the following categories:

Coding Error - Functional: Issues in core functionality, assigned to the relevant user story owner.

Coding Error - GUI: UI-related issues, linked to the respective user story.

Coding Error - Interface: Defects in interaction or API, linked to the respective user story.

Test Execution Error: Issues in test automation scripts, assigned to Tech Lead or TAE.

Infrastructure Error: Defects due to environment or network issues, assigned to Tech Lead.

Test Data Error: Errors in test data mapping or transformation, linked to the respective user story.

Deployment Error: Defects from deployment issues, assigned to Tech Lead.

Non-Functional Error: Issues related to performance (e.g., response time, CPU usage), linked to the respective user story.

Special Categories: Some cases, such as "Unknown," "Not a Defect," and "Test Execution Error," are assigned to testers for further investigation.

3. Machine Learning Model Implementation

Model Training: We used a Random Forest Classifier for initial implementation, utilizing features such as test case name, error messages, and logs to classify defects.

Integration with Jenkins: The ML model is integrated with Jenkins for real-time classification upon test execution failure. Failure logs are passed to the ML model for classification.

Jira Defect Logging: Defects are automatically logged into Jira and assigned to the appropriate assignees using Jira REST API. A script automates defect logging based on classification results.

4. Workflow Integration

Data Preparation: A scheduled process (1 AM every Monday) extracts and prepares data from Jenkins: Jenkins Pipeline Link.

Model Training & Re-Training: Feature files for training are stored on the DTA OCP server: /data/testops/feature_files/train_mode.

API Integration: Automated defect classification and logging are managed through Jenkins pipelines and APIs for classification and Jira logging.

Classification API: API Endpoint Link

Jira Central API: Jira API Endpoint Link

5. Project Timeline Summary

Project Initiation: Defined scope, kickoff completed by June 15, 2024.

Requirements Gathering: Requirements were gathered by June 28, 2024.

ML Model Development: Completed POC and initial model, ongoing GitLab CI integration.

API and TestOps Integration: API development in progress; integrated automation with TestOps.

Testing and Validation: Completed unit and integration testing; currently validating the end-to-end workflow.

Go Live MVP 1.0: Scheduled for August 31, 2024, including Jenkins and CI implementation.

6. Benefits of API Integration for ML Models

Accessibility: Allows remote access and integration across different platforms.

Scalability: Load balancing and efficient resource management ensure high availability.

Security: Authentication mechanisms ensure data privacy and secure access.

7. Short-Term and Long-Term Plans

MVP 1.0 (Short-Term): Rule-based defect categorization using Random Forest Classifier, focus on addressing data leakage, reducing manual labeling effort with LLMs.

Long-Term: Leveraging LLMs for flexible log analysis, improving classification accuracy, and establishing an MLOps pipeline for sustainability.

8. Action Items and Next Steps

Team Review: Review the existing rule-based patterns and validate against new defect cases.

ML Model Improvements: Implement active learning and semi-supervised techniques to reduce manual labeling.

Deployment Preparation: Ensure readiness for MVP 1.0 go-live by completing integration and validation tasks.

9. MLflow Deployment and Usage Guide for MLOps in OCP 4 (DTA Team)

Overview This knowledge base entry provides a step-by-step overview of our MLflow setup in the OpenShift (OCP 4) environment. It includes instructions for deploying MLflow via CI/CD pipelines, training and registering models, and consuming the models in downstream applications.

High-Level Architecture

MLflow Server: Hosted on OCP 4 at MLflow Server Link

KSCM GitLab for Model Training & Registration: GitLab Repository Link

Pipeline for Model Training: Jenkins Pipeline Link

Model Consumption: Models are served through APIs available at Defect Classification API Documentation

1. MLflow Server Deployment

Pipeline Configuration The MLflow server is deployed using the OCP 4 CI/CD pipeline via Jenkins. Here are the key components and configurations:

Factory YAML File: The pipeline is managed using the factory configuration file: mlflow-server.factory-dev.yaml. Contact Peerapach Varalertsakul for more information.

YAML Overview

GitLab Repository: The pipeline pulls code from dtaguild/dta-tools/mlflow-server.

Stages:

pullCode: Pulls the latest code from the specified branch.

unitTest: Skips unit tests for this configuration.

buildImage: Uses Kaniko to build Docker images, with proxy settings for internal access.

buildK8S: Deploys MLflow to OCP with custom overlays, and uses environment variables from Vault for secure deployment.

OpenShift Site Configuration

Cluster: ocp4-test

Namespace: dta-sit

Vault Integration: Secrets are managed securely through Vault.

Jenkins Job for MLflow Server Deployment The Jenkins job executes the CI/CD pipeline, pulling the MLflow code, building the Docker image, and deploying to OCP.

Result The MLflow server can be accessed at: MLflow Server Link. Here, users can monitor experiments, view model performance metrics, and manage registered models.

2. Model Training and Registration

The project for training models and registering them to MLflow is located at:

Repository: New Defect Classification Models

Training and Registration Process

Setup MLflow Tracking: MLflow tracking is configured in the project to point to the MLflow server.

Model Training: The repository contains code for training machine learning models (cypress_defect_classifier and robot_defect_classifier) used for defect classification.

Training Data Paths:

Cypress: /data/manual/cypress/cypress_test_data.csv (Data generated using generative AI due to insufficient real data).

Robot: /data/manual/robot/robot_test_data.csv (Data generated from extraction, manual labeling, and classification using regular expressions).

Labeling for Robot Data: In the legacy system, the robot training data was labeled using a combination of manual efforts and automated classification based on regular expression patterns. The labeling logic is implemented in the src/defect_classifier.py file, where various error patterns are matched to predefined defect categories stored in main/common_module/defect_categories.py.

Model Registration: Once trained, models are registered in MLflow and saved to Artifactory.

Registered Models:

cypress_defect_classifier

robot_defect_classifier

Access Registered Models: Once the model training pipeline is complete, users can view the models at the MLflow Model Registry.

3. Model Consumption

The trained and registered models are consumed by an API project located at:

Repository: Defect Classification API

API Documentation: Swagger Documentation

Usage

Model Loading: The API uses the latest version of the models directly from MLflow.

Endpoints: The API provides endpoints to classify defects using the loaded models.

Prediction API: The API routes requests to the relevant ML models to predict defect classifications and returns results to clients.

API Endpoints The main endpoints available for defect classification are:

Cypress Defect Classification

Robot Defect Classification

These endpoints enable other applications to send test case data for classification, leveraging the latest registered ML models from MLflow.

4. Source Code and Jenkins Pipeline Overview

The following table summarizes the source code locations and Jenkins pipelines used for building and deploying each component of the defect classification system:

Component	Source Code Repository Link	Branch	Jenkins Pipeline Link
DTA Defect Classification API	Defect Classification API	main (prod), mlflow_feature (dev)	Jenkins Pipeline
DTA MLflow Server	MLflow Server Code	main	Jenkins Pipeline for MLflow
Legacy Defect Classification Model (Legacy Use)	Legacy Defect Classification Model	main	Jenkins Pipeline for Model Training
DTA New Defect Classification Model	New Defect Classification Model	main	Jenkins Pipeline for New Model Training

5. Key Points for Handover

Key Links

MLflow Server: MLflow Deployment

Jenkins CI/CD: Jenkins Pipeline for MLflow Deployment

Model Training Project: Defect Classification Models Repository

API Project for Consumption: [Defect Classification API Repository](https://kscm.kasikornbank.com
