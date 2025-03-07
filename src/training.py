import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import shap

import numpy as np
import pandas as pd

# load cleaned dataset
data = pd.read_excel("data/processed/train_test.xlsx")

# Separate features and target
X = data.drop(columns=["Churn"])
y = data["Churn"]

# Split data into training (80%) and holdout (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))




# Initialize SHAP explainer
explainer = shap.Explainer(model)  
shap_values = explainer(X_test) 

#shap.summary_plot(shap_values, X_test, plot_type="bar")



# Compute mean absolute SHAP values per feature
shap_importance = np.abs(shap_values.values).mean(axis=0)

# Create a DataFrame to show feature importance
shap_importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean SHAP Value': shap_importance
}).sort_values(by="Mean SHAP Value", ascending=False)

# Print the top 10 most important features
print(shap_importance_df.head(10))

model.save_model("/Users/maimunaz/Downloads/churn_prediction/models/churn_model2.json")