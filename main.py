from preprocessing import clean_data
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# ➕ NEW: Imbalanced-learn imports for SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Step 1: Load and clean data
df = clean_data('Churn_data.csv')

# Step 2: Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define column types
categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]
numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

# Step 4: Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# ==============================
# MODEL 1: RANDOM FOREST
# ==============================
rf_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

print("\n Random Forest Results:")
print(classification_report(y_test, rf_pred))

cm_rf = confusion_matrix(y_test, rf_pred)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=[0, 1])
disp_rf.plot(cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.show()

# ==============================
# MODEL 2: BASIC LOGISTIC REGRESSION
# ==============================
lr_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000))
])

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

print("\n Basic Logistic Regression Results:")
print(classification_report(y_test, lr_pred))

cm_lr = confusion_matrix(y_test, lr_pred)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=[0, 1])
disp_lr.plot(cmap='Purples')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# ==============================
# MODEL 3: TUNED LOGISTIC REGRESSION (SMOTE + GridSearchCV)
# ==============================
smote_pipeline = ImbPipeline(steps=[
    ('preprocessing', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', LogisticRegression(max_iter=1000))
])

param_grid = {
    'model__penalty': ['l2'],
    'model__C': [0.01, 0.1, 1, 10],
    'model__solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(smote_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

smote_pred = grid_search.predict(X_test)

print("\n Tuned Logistic Regression (SMOTE + GridSearch) Results:")
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, smote_pred))

cm_smote = confusion_matrix(y_test, smote_pred)
disp_smote = ConfusionMatrixDisplay(confusion_matrix=cm_smote, display_labels=[0, 1])
disp_smote.plot(cmap='Greens')
plt.title("Confusion Matrix - Tuned Logistic Regression (SMOTE + GridSearch)")
plt.show()

# ==============================
# Save All Models
# ==============================
joblib.dump(rf_pipeline, 'model_random_forest.pkl')
joblib.dump(lr_pipeline, 'model_logistic_regression_basic.pkl')
joblib.dump(grid_search.best_estimator_, 'model_logistic_regression_smote.pkl')


print("\n Models saved: random forest, basic logistic regression, tuned logistic regression with SMOTE.")

