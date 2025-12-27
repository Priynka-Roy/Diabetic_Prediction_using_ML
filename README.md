# Diabetic_Prediction_using_ML

![output](output.png)

## Overview
This project aims to build a **Diabetes Prediction Model** using **Logistic Regression** and **Random Forest** algorithms. The dataset used for this analysis is the **Diabetes Prediction Dataset**, and the implementation is done in **R**. The objective is to classify individuals as diabetic or non-diabetic based on various medical attributes.

## Dataset
The dataset contains 768 observations with the following features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/height in m²)
- **DiabetesPedigreeFunction**: Diabetes pedigree function score
- **Age**: Age in years
- **Outcome**: Diabetes diagnosis (0 = No, 1 = Yes)

## Project Workflow
1. **Data Preprocessing**
   - Checking for missing values
   - Normalizing numerical features
   - Splitting the dataset (80% training, 20% testing)

2. **Model Training**
   - Logistic Regression Model
   - Random Forest Model

3. **Model Evaluation**
   - Accuracy, Sensitivity, and Specificity
   - Confusion Matrix
   - ROC Curve and AUC Score
   
4. **Performance Comparison**
   - Logistic Regression Accuracy: **80.39%**
   - Random Forest Accuracy: **81.70%**
   - AUC (Area Under Curve):
     - Logistic Regression: **0.7569**
     - Random Forest: **0.7846**
   
## Visualizations
- **Confusion Matrices**: Shows the correct and incorrect predictions
- **ROC Curves**: Compares model performance using True Positive and False Positive Rates
- **Feature Importance Plot (Random Forest)**: Displays key features impacting predictions
- **Model Accuracy Comparison**: Bar chart for visualizing accuracy differences

## Installation & Usage
To run this project locally, follow these steps:

1. Install required R packages:
   ```r
   install.packages(c("tidyverse", "caret", "randomForest", "e1071", "ggplot2", "pROC"))
   ```
2. Clone the repository and set the working directory:
   ```r
   df <- read.csv("path/to/1_diabetes.csv")
   ```
3. Execute the R script:
   ```r
   source("diabetes_prediction.R")
   ```

## Model Deployment
- The trained **Random Forest model** is saved as `diabetes_rf_model.rds`.
- Can be reloaded using:
   ```r
   loaded_model <- readRDS("diabetes_rf_model.rds")
   ```

## Future Improvements
- Try advanced algorithms like **Gradient Boosting (XGBoost)**
- Improve feature engineering techniques
- Deploy the model as a web app for real-time predictions

