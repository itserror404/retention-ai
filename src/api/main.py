from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import shap
import numpy as np

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request, Form, UploadFile, File, Depends
import os
import shutil
import matplotlib.pyplot as plt


# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = xgb.XGBClassifier()
model.load_model("/Users/maimunaz/Downloads/churn_prediction/models/churn_model.json")
explainer = shap.TreeExplainer(model)

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    Contract_One_year: int
    Contract_Two_year: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

# Feature name mapping to match trained model's expected format
feature_mapping = {
    "Contract_One_year": "Contract_One year",
    "Contract_Two_year": "Contract_Two year",
    "InternetService_Fiber_optic": "InternetService_Fiber optic",
    "PaymentMethod_Credit_card_automatic": "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic_check": "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed_check": "PaymentMethod_Mailed check",
}

class CustomerBatch(BaseModel):
    customers: list[CustomerData] 
    
@app.post("/predict")
def predict(batch: CustomerBatch):
    try:
        # Initialize list to store predictions
        predictions = []

        # Iterate over each customer in the batch
        for customer in batch.customers:
            # Convert input data to a DataFrame
            input_data = pd.DataFrame([customer.dict()])

            # Rename columns to match trained model's feature names
            input_data = input_data.rename(columns=feature_mapping)

            # Make prediction
            probabilities = model.predict_proba(input_data)[:, 1]  # Probability of churn (1)
            prediction = (probabilities >= 0.5).astype(int)  # Convert to 1/0 prediction

            # SHAP explanation
            shap_values = explainer.shap_values(input_data)[0]
            print("SHAP values for the customer:", shap_values)

            
            # Select top 3 most important features
            feature_importance = dict(sorted(
                zip(input_data.columns, shap_values),
                key=lambda x: abs(x[1]), reverse=True
            )[:3])

            # Add the result to the predictions list
            predictions.append({
                "churn_prediction": int(prediction[0]),  # Convert NumPy float to Python int
                "churn_probability": float(probabilities[0]),  # Return probability of churn
                "explanation": {k: float(v) for k, v in feature_importance.items() if abs(v) > 0.001 }
            })

        return {"predictions": predictions}  # Return all predictions as a list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
