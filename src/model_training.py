import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# load cleaned dataset
df = pd.read_csv("data/processed/telco_cleaned.csv")

# split dataset into features and target
X= df.drop(columns=["Churn"])
y= df["Churn"]


# split dataset into training and testing 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Split X_temp further: 80% test, 20% demo
X_test, X_demo, y_test, y_demo = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)


# Save demo set separately for frontend testing
X_demo.to_csv("demo_X.csv", index=False)
y_demo.to_csv("demo_y.csv", index=False)

# train model
# model= xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')


ratio = df["Churn"].value_counts()[0] / df["Churn"].value_counts()[1]
# model = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=ratio)

model = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=0.8 * ratio, 
                          max_depth=6, learning_rate=0.01,n_estimators=300)

model.fit(X_train, y_train)

# prediction time!
y_pred = model.predict(X_test)
accuracy= accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
importances = model.feature_importances_
plt.barh(X_train.columns, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.show() 

importances = model.feature_importances_

# Create a DataFrame for better readability
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})

# Sort by importance (descending order)
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the results
print(feature_importance_df)

model.save_model("/Users/maimunaz/Downloads/churn_prediction/models/churn_model.json")
