import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("Data Loading :")
df = pd.read_csv('diabetes.csv')


print(f"Dataset Shape: {df.shape}")
print("First 5 rows of the dataset:")
print(df.head(), "\n")



print("Data Preprocessing :")
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
print("1. Replaced invalid 0 values with NaN.")

def categorize_bmi(bmi):
    if pd.isna(bmi): return np.nan
    if bmi < 18.5: return 0
    elif bmi < 25: return 1
    elif bmi < 30: return 2
    else: return 3
df['BMI_Category'] = df['BMI'].apply(categorize_bmi)
print("2. Created new engineered feature: 'BMI_Category'.")

insulin_95th = df['Insulin'].quantile(0.95)
df['Insulin'] = df['Insulin'].clip(upper=insulin_95th)
print("3. Handled outliers by clipping 'Insulin' at the 95th percentile.")


X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



print("Pipeline Creation :")
preprocessing_steps = [
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
]


pipeline = Pipeline(preprocessing_steps + [
    ('classifier', RandomForestClassifier(random_state=42))
])
print("Pipeline created successfully with Imputer, Scaler, and Classifier.\n")



print("Primary Model Selection :")
print("Model Chosen: RandomForestClassifier")
print("Justification: Random Forest is highly effective for medical datasets like this one because it naturally handles non-linear relationships and complex interactions between features .\n It is robust against overfitting due to its ensemble nature and handles remaining outliers well.\n")



print("Model Training :")
pipeline.fit(X_train, y_train)
print("Model trained successfully on the training dataset.\n")


print("Cross-Validation :")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores (5 folds): {cv_scores}")
print(f"Average CV Score: {np.mean(cv_scores):.4f} (Standard Deviation: +/- {np.std(cv_scores):.4f})\n")


print("Hyperparameter Tuning :")

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5, 10]
}

print("Parameters being tested :")
for key, values in param_grid.items():
    print(f" - {key}: {values}")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest Parameters Found :")
print(grid_search.best_params_)
print(f"Best CV Score during tuning : {grid_search.best_score_:.4f}\n")


print("Best Model Selection :")
best_model = grid_search.best_estimator_
print("Final best-performing model successfully selected based on hyperparameter tuning.\n")



print("Model Performance Evaluation  :")
y_pred = best_model.predict(X_test)
print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Comprehensive Classification Report:")
print(classification_report(y_test, y_pred))





print("Saving the Model  :")
filename = 'diabetes_prediction_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(best_model, file)
print(f"Model successfully saved as: {filename}")



