# Customer Churn Prediction

This project focuses on predicting customer churn for a telecom company using supervised machine learning models. 
It includes data preprocessing, model training, hyperparameter tuning, and evaluation, providing a complete machine learning 
pipeline for practical use.

---

## Dataset

- Source: Telco Customer Churn Dataset
- Rows: 7,043
- Columns: 21
- Target Variable: `Churn` (binary: Yes/No)

Features include:
- Demographics (e.g., gender, senior citizen status, dependents)
- Service subscriptions (e.g., InternetService, PhoneService)
- Billing details (e.g., MonthlyCharges, TotalCharges)
- Contract and payment methods

---

## Data Preprocessing

Performed in `preprocessing.py`:
- Converted `TotalCharges` to numeric
- Filled missing values with median
- Dropped irrelevant column `customerID`
- Encoded categorical variables using `OneHotEncoder`
- Scaled numerical columns using `StandardScaler`

---

## Models Trained

### 1. Random Forest (Baseline)
- Included class weighting for imbalance
- Accuracy: 80%
- Recall for churners: 47%
- output image:
  ![Screenshot 2025-04-30 at 12 20 36 PM](https://github.com/user-attachments/assets/5e5d3d0a-d7d5-455f-a6b8-492d91dc1db3)


### 2. Logistic Regression (Basic)
- Used class weighting
- Accuracy: 75%
- Recall for churners: 83%
- -output image:
  ![Screenshot 2025-04-30 at 12 20 48 PM](https://github.com/user-attachments/assets/aa7abb7b-7f7d-48c2-9ddd-2f56cc108e8d)


### 3. Logistic Regression with SMOTE and GridSearchCV (Final Model)
- Applied SMOTE to balance training data
- Performed hyperparameter tuning using GridSearchCV
- Accuracy: 76%
- Recall for churners: 83%
- F1-score for churners: 65%
- Best parameters: `C=0.1`, `penalty='l2'`, `solver='liblinear'`
- output image:
- ![Screenshot 2025-04-30 at 12 20 58 PM](https://github.com/user-attachments/assets/160c9074-f19e-427c-80b3-942c8b7d027f)


---

## Project Structure


## How to Run
1. Clone repo
    ```bash
   git clone https://github.com/Jaiswal-Devpriya/customer-churn-prediction.git
   cd customer-churn-prediction
3. Create and activate a virtual environment
  python3 -m venv .venv
  source .venv/bin/activate

4. Install dependencies with `pip install -r requirements.txt`
   
5. Run `main.py`

## Author
Devpriya Jaiswal
Master’s in Data Science, Arizona State University
LinkedIn: [Jaiswal-Devpriya](https://www.linkedin.com/in/jaiswal-devpriya)
