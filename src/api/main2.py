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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "../templates"))

# Mount static files and templates

static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../static"))
app.mount("/static", StaticFiles(directory=static_dir), name="static")



# Load the trained model
model = xgb.XGBClassifier()
model.load_model("/Users/maimunaz/Downloads/churn_prediction/models/churn_model2.json")

explainer = shap.TreeExplainer(model)



class CustomerData(BaseModel):
    Tenure: int
    CityTier: int
    WarehouseToHome: int
    Gender: int
    HourSpendOnApp: float
    NumberOfDeviceRegistered: int
    SatisfactionScore: int
    NumberOfAddress: int
    Complain: int
    OrderAmountHikeFromlastYear: float
    CouponUsed: int
    OrderCount: int
    DaySinceLastOrder: int
    CashbackAmount: float
    PreferredLoginDevice_Mobile_Phone: int
    PreferredLoginDevice_Phone: int
    PreferredPaymentMode_COD: int
    PreferredPaymentMode_Cash_on_Delivery: int
    PreferredPaymentMode_Credit_Card: int
    PreferredPaymentMode_Debit_Card: int
    PreferredPaymentMode_E_wallet: int
    PreferredPaymentMode_UPI: int
    PreferedOrderCat_Grocery: int
    PreferedOrderCat_Laptop_Accessory: int
    PreferedOrderCat_Mobile: int
    PreferedOrderCat_Mobile_Phone: int
    PreferedOrderCat_Others: int
    MaritalStatus_Married: int
    MaritalStatus_Single: int

# Feature name mapping to match trained model's expected format
feature_mapping = {
    "PreferredLoginDevice_Mobile_Phone": "PreferredLoginDevice_Mobile Phone",
    "PreferredLoginDevice_Phone": "PreferredLoginDevice_Phone",
    "PreferredPaymentMode_COD": "PreferredPaymentMode_COD",
    "PreferredPaymentMode_Cash_on_Delivery": "PreferredPaymentMode_Cash on Delivery",
    "PreferredPaymentMode_Credit_Card": "PreferredPaymentMode_Credit Card",
    "PreferredPaymentMode_Debit_Card": "PreferredPaymentMode_Debit Card",
    "PreferredPaymentMode_E_wallet": "PreferredPaymentMode_E wallet",
    "PreferredPaymentMode_UPI": "PreferredPaymentMode_UPI",
    "PreferedOrderCat_Grocery": "PreferedOrderCat_Grocery",
    "PreferedOrderCat_Laptop_Accessory": "PreferedOrderCat_Laptop & Accessory",
    "PreferedOrderCat_Mobile": "PreferedOrderCat_Mobile",
    "PreferedOrderCat_Mobile_Phone": "PreferedOrderCat_Mobile Phone",
    "PreferedOrderCat_Others": "PreferedOrderCat_Others",
    "MaritalStatus_Married": "MaritalStatus_Married",
    "MaritalStatus_Single": "MaritalStatus_Single"
}

class CustomerBatch(BaseModel):
    customers: list[CustomerData] 
    
@app.post("/predict")
def predict(batch: CustomerBatch):
    try:
        predictions = []

        for customer in batch.customers:
            input_data = pd.DataFrame([customer.dict()])
            input_data = input_data.rename(columns=feature_mapping)
            
            probabilities = model.predict_proba(input_data)[:, 1]
            prediction = (probabilities >= 0.5).astype(int)
            
            shap_values = explainer.shap_values(input_data)[0]
            print("SHAP values for the customer:", shap_values)
            
            feature_importance = dict(sorted(
                zip(input_data.columns, shap_values),
                key=lambda x: abs(x[1]), reverse=True
            )[:3])

            predictions.append({
                "churn_prediction": int(prediction[0]),
                "churn_probability": float(probabilities[0]),
                "explanation": {k: float(v) for k, v in feature_importance.items() if abs(v) > 0.001 }
            })

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    file_path = f"src/uploads/{file.filename}"
    os.makedirs("src/uploads", exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Load and process CSV
    df = pd.read_csv(file_path)
    df = df.drop(columns=["Churn"], errors="ignore")  # Remove target column if present
    predictions = model.predict(df)
    
    df["Churn Prediction"] = predictions
    df.to_csv(file_path, index=False)  # Save results

    return templates.TemplateResponse("results.html", {
        "request": request, "filename": file.filename, "data": df.head(10).to_dict(orient="records")
    })


@app.get("/predict", response_class=HTMLResponse)
async def get_predict(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def post_predict(request: Request, tenure: int = Form(...), gender: int = Form(...), satisfaction: int = Form(...)):
    input_data = pd.DataFrame([[tenure, gender, satisfaction]], columns=["Tenure", "Gender", "SatisfactionScore"])
    prediction = model.predict(input_data)[0]
    
    return templates.TemplateResponse("predict.html", {
        "request": request, "prediction": prediction
    })
