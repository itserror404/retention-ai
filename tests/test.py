import requests

url = "http://127.0.0.1:8000/predict"

# Wrap the data inside a dictionary with a key like 'customers' or 'data'
data = {
    "customers": [  # <-- Wrap the list inside a dictionary
        {
            "gender": 0, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0, "tenure": 1,
            "PhoneService": 1, "MultipleLines": 1, "OnlineSecurity": 0, "OnlineBackup": 0,
            "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 1,
            "PaperlessBilling": 0, "MonthlyCharges": 85.8, "TotalCharges": 85.8, "Contract_One_year": 0,
            "Contract_Two_year": 0, "InternetService_Fiber_optic": 1, "InternetService_No": 0,
            "PaymentMethod_Credit_card_automatic": 0, "PaymentMethod_Electronic_check": 0,
            "PaymentMethod_Mailed_check": 1
            
   
        },
        {
            "gender": 1, "SeniorCitizen": 1, "Partner": 1, "Dependents": 0, "tenure": 50,
            "PhoneService": 1, "MultipleLines": 1, "OnlineSecurity": 1, "OnlineBackup": 1,
            "DeviceProtection": 1, "TechSupport": 1, "StreamingTV": 1, "StreamingMovies": 1,
            "PaperlessBilling": 1, "MonthlyCharges": 90, "TotalCharges": 4500, "Contract_One_year": 0,
            "Contract_Two_year": 1, "InternetService_Fiber_optic": 1, "InternetService_No": 0,
            "PaymentMethod_Credit_card_automatic": 1, "PaymentMethod_Electronic_check": 0,
            "PaymentMethod_Mailed_check": 0
        }
    ]
}

response = requests.post(url, json=data)
print(response.json())
