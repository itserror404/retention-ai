from fastapi import FastAPI, HTTPException, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import xgboost as xgb
import pandas as pd
import shap
import numpy as np
import os
from fastapi.responses import HTMLResponse


import os

# Initialize FastAPI app
app = FastAPI()


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels
MODEL_PATH = os.path.join(project_root, "models", "churn_model.json")

print(f"Loading model from: {MODEL_PATH}")
print(f"File exists: {os.path.exists(MODEL_PATH)}")

# Load model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
 
explainer = shap.TreeExplainer(model)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.dirname(BASE_DIR)
templates_path = os.path.join(SRC_DIR, "templates")
templates = Jinja2Templates(directory=templates_path)


static_path = os.path.join(SRC_DIR, "static")
print(static_path)
os.makedirs(static_path, exist_ok=True)
os.makedirs(templates_path, exist_ok=True)

# Set up Jinja2 templates

 # Static files for CSS, images
app.mount("/static", StaticFiles(directory=static_path), name="static")
# Feature name mapping to match trained model's expected format
feature_mapping = {
    "Contract_One_year": "Contract_One year",
    "Contract_Two_year": "Contract_Two year",
    "InternetService_Fiber_optic": "InternetService_Fiber optic",
    "PaymentMethod_Credit_card_automatic": "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic_check": "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed_check": "PaymentMethod_Mailed check",
}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    gender: int = Form(...),
    SeniorCitizen: int = Form(...),
    Partner: int = Form(...),
    Dependents: int = Form(...),
    tenure: int = Form(...),
    PhoneService: int = Form(...),
    MultipleLines: int = Form(...),
    OnlineSecurity: int = Form(...),
    OnlineBackup: int = Form(...),
    DeviceProtection: int = Form(...),
    TechSupport: int = Form(...),
    StreamingTV: int = Form(...),
    StreamingMovies: int = Form(...),
    PaperlessBilling: int = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...),
    Contract_One_year: int = Form(...),
    Contract_Two_year: int = Form(...),
    InternetService_Fiber_optic: int = Form(...),
    InternetService_No: int = Form(...),
    PaymentMethod_Credit_card_automatic: int = Form(...),
    PaymentMethod_Electronic_check: int = Form(...),
    PaymentMethod_Mailed_check: int = Form(...),
):
    try:
        # Convert input to DataFrame
        
        input_values = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "PaperlessBilling": PaperlessBilling,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
            "Contract_One_year": Contract_One_year,
            "Contract_Two_year": Contract_Two_year,
            "InternetService_Fiber_optic": InternetService_Fiber_optic,
            "InternetService_No": InternetService_No,
            "PaymentMethod_Credit_card_automatic": PaymentMethod_Credit_card_automatic,
            "PaymentMethod_Electronic_check": PaymentMethod_Electronic_check,
            "PaymentMethod_Mailed_check": PaymentMethod_Mailed_check,
        }
         
        input_data = pd.DataFrame([input_values])
        
        # Rename columns to match trained model
        input_data = input_data.rename(columns=feature_mapping)

        # Predict churn probability
        probabilities = model.predict_proba(input_data)[:, 1]
        prediction = (probabilities >= 0.5).astype(int)

        # SHAP explanation
        shap_values = explainer.shap_values(input_data)[0]
        feature_importance = dict(sorted(
            zip(input_data.columns, shap_values),
            key=lambda x: abs(x[1]), reverse=True
        )[:3])
        
        

        # Prepare result
        result = {
            "churn_prediction": "Churn" if prediction[0] == 1 else "Not Churn",
            "churn_probability": round(float(probabilities[0]), 2),
            "explanation": feature_importance
        }
        
        result.update(input_values)

        return templates.TemplateResponse("result.html", {"request": request, "result": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
