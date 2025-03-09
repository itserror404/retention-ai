# retention-ai: Early Churn Detection and Customer Retention Strategies

[![Stars](https://img.shields.io/github/stars/itserror404/retention-ai?style=social)](https://github.com/itserror404/retention-ai)
[![Forks](https://img.shields.io/github/forks/itserror404/retention-ai?style=social)](https://github.com/itserror404/retention-ai)
[![Watchers](https://img.shields.io/github/watchers/itserror404/retention-ai?style=social)](https://github.com/itserror404/retention-ai)


![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB5B2E?style=for-the-badge&logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-FF6F00?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

🔗 **Live Project:** [retention-ai.onrender.com](https://retention-ai.onrender.com)
## Introduction

Retention-AI is a machine learning project for early churn detection and customer retention. It processes the Telco Customer Churn dataset, trains an XGBoost model, and serves predictions via a FastAPI interface deployed on Render. The system provides real-time churn analysis with feature explanations, helping businesses retain at-risk customers through data-driven insights.

Key Features
- Early Churn Prediction: Detects at-risk customers before they leave.
- Data-Driven Insights: Uses explainable AI to highlight churn risk factors.
- Scalable Architecture: Built with FastAPI and deployed on Render for real-time predictions.
- End-to-End Workflow: From data preprocessing to model deployment. 


## Try It Out! 🚀

The model is deployed and ready for you to test. Visit our [Live Demo](https://retention-ai.onrender.com) to see Retention-AI in action! 

## Demo

Watch a quick video showcasing how to use Retention-AI:



## Table of Contents


- [Data Processing](#data-processing)
- [Model Development](#model-development)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [API Development](#api-development)
- [Deployment](#deployment)
- [Contact](#contact)



## Data Processing

### Dataset Used: Telco Customer Churn  
The **Telco Customer Churn** dataset was selected as it contains customer demographics, account details, and service usage information, making it suitable for predicting churn.  

### Data Cleaning & Preprocessing  

#### 1. Handling Missing Values  
- Identified missing or incorrect values in the **TotalCharges** column.  
- Converted it to numeric format (`float64`), replacing non-numeric values with `NaN`.  
- Imputed missing values with `0`.  

#### 2. Encoding Categorical Variables  
- **Binary Encoding:** Converted categorical variables with two unique values (e.g., `gender`, `Partner`, `Churn`) into binary (`0` and `1`).  
- **One-Hot Encoding:** Applied one-hot encoding for categorical columns with more than two values (`Contract`, `InternetService`, `PaymentMethod`), dropping the first category to avoid multicollinearity.  

#### 3. Handling "No phone service"/"No internet service" Entries  
- Replaced `"No phone service"` and `"No internet service"` with `"No"` in relevant columns (`MultipleLines`, `OnlineSecurity`, `OnlineBackup`, etc.).  
- Mapped `"No"` to `0` and `"Yes"` to `1`.  

#### 4. Feature Type Adjustments  
- Ensured one-hot encoded columns were correctly cast to integers (`int`).  

#### 5. Dropping Unnecessary Columns  
- Removed `customerID`, as it is an identifier with no predictive value.  

#### 6. Saving Processed Data  
- The cleaned dataset was saved as `telco_cleaned.csv` in the processed data folder for model training.  

#### Absence of Temporal Data  
Since this dataset isn’t time-series, the model can’t track gradual disengagement, limiting its ability to predict churn before clear signs appear.

📂 For full preprocessing details, check [`data-processing.ipynb`](notebooks/data-processing.ipynb).  


## Model Development
## Training

The model is trained using the **XGBoost classifier**, a powerful gradient boosting algorithm well-suited for **imbalanced classification problems like churn prediction**. The dataset is split into:
- **Training Set (80%)** – Used to train the model.
- **Testing Set (16%)** – Used to evaluate the model.
- **Demo Set (4%)** – Saved separately for frontend testing.


### Hyperparameter Tuning
The model was trained with the following hyperparameters:
- `max_depth=6`: Limits tree depth to prevent overfitting.
- `learning_rate=0.01`: Ensures gradual convergence.
- `n_estimators=300`: Uses 300 boosting rounds.
- `scale_pos_weight=0.8 * (non-churn / churn ratio)`: Adjusts for the **imbalance in churned vs. non-churned customers**.

## Evaluation
### Model Performance
The model achieves **79.33% accuracy** in predicting customer churn.
### Classification Report

```
           precision    recall  f1-score   support

           0       0.91      0.79      0.85       823
           1       0.59      0.79      0.67       304

    accuracy                           0.79      1127
   macro avg       0.75      0.79      0.76      1127 
weighted avg       0.82      0.79      0.80      1127
```

This classification report provides insights into the model's performance:

*   **Precision**: Of all the customers predicted as churned, precision tells us the percentage that actually churned. A precision of 0.59 for churn means that when the model predicts a customer will churn, it is correct 59% of the time.
*   **Recall**: Of all the customers who actually churned, recall tells us the percentage that the model correctly predicted. A recall of 0.79 for churn means that the model correctly identifies 79% of customers who will churn.
*   **F1-Score**: The F1-score is the harmonic mean of precision and recall. It provides a single score that balances both concerns.

In this context:

*   **High recall for churned customers (0.79)** means the model correctly identifies most at-risk customers, which is **crucial for proactive retention strategies**.
*   **Precision for churned customers (0.59)** indicates some false positives but is acceptable given the focus on minimizing churn loss.

  

### Feature Importance
The model assigns importance scores to all input features based on how much they contribute to churn prediction. After training, the **most impactful features** identified were:

1. Contract Type (One year, Two year)
2. Internet Service Type (Fiber Optic)
3. Streaming Services Usage
![feature_importance](https://github.com/user-attachments/assets/fd9d1758-4ca9-4579-9ead-032d796668c4)


### Model Storage
The trained model is saved as: [`model_training.py`](src/model_training.py)


## API Development

This FastAPI application serves as a lightweight backend for churn prediction, allowing users to input customer details and receive predictions on whether they are likely to churn. It also provides explanations for the predictions using SHAP values.  

### Endpoints  

- **`GET /`** – Serves the homepage with a form for user input.  
- **`POST /predict`** – Accepts customer details via a form, processes the input, and returns a churn prediction along with feature importance explanations.  

### Functionality  

1. **Model Loading**  
   - The pre-trained XGBoost model is loaded from `models/churn_model.json`.  
   - A SHAP explainer is initialized to provide feature importance for individual predictions.  

2. **Prediction Flow**  
   - User inputs are collected via an HTML form and passed to the `/predict` endpoint.  
   - The input is formatted to match the trained model’s expected features.  
   - The model predicts the probability of churn and returns a classification (Churn / Not Churn).  
   - The top three most influential features for the prediction are extracted using SHAP values.  

3. **Feature Mapping**  
   - Since FastAPI form inputs use underscores (`_`), a feature mapping is applied to match the model’s expected column names (e.g., `Contract_One_year` → `Contract_One year`).  

4. **Result Presentation**  
   - The prediction result, probability, and top influencing features are displayed on an HTML results page (`result.html`).  

When the API returns a prediction, it also provides information on the factors influencing that prediction. Each factor has a number and an arrow, and can be interpreted as follows:

*   **▲ Positive Number** → Increases churn risk
*   **▼ Negative Number** → Decreases churn risk
*   **▲ Large Positive Number** → Strong churn factor
*   **▼ Large Negative Number** → Strong retention factor

<img width="691" alt="Screenshot 2025-03-09 at 4 21 44 PM" src="https://github.com/user-attachments/assets/7574a293-7142-46b8-8a73-ed191e9c1387" />


This API enables quick and interpretable customer churn predictions, making it easy to integrate into a broader customer retention system.  

## Deployment 

The API is deployed on **Render** using a free instance. Deployment was straightforward:  
1. Connected the GitHub repository to Render.  
2. Added build and run commands, including installing dependencies from `requirements.txt`.  
3. Render automatically handles deployment and hosting.  

 **Note:** Since this is a free instance, it spins down when inactive, causing delays of **50+ seconds** for the first request.  
**Live Demo:** [https://retention-ai.onrender.com](https://retention-ai.onrender.com)  

## Contact

[Maimuna Zaheer] - [mz2934@nyu.edu]

Project Link: [https://github.com/itserror404/retention-ai](https://github.com/itserror404/retention-ai)

