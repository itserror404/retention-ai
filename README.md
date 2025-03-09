# retention-ai

# Retention-AI: Early Churn Detection and Customer Retention Strategies

[![Stars](https://img.shields.io/github/stars/itserror404/retention-ai?style=social)](https://github.com/itserror404/retention-ai)
[![Forks](https://img.shields.io/github/forks/itserror404/retention-ai?style=social)](https://github.com/itserror404/retention-ai)
[![Watchers](https://img.shields.io/github/watchers/itserror404/retention-ai?style=social)](https://github.com/itserror404/retention-ai)

## Introduction

Retention-AI is a machine learning project focused on early churn detection and providing data-driven strategies for customer retention. This repository contains the code, models, and documentation necessary to understand, reproduce, and deploy our churn prediction system.

### Try It Out!

The model is deployed and ready for you to test. Visit our [Live Demo](https://retention-ai.onrender.com) to see Retention-AI in action!

## Demo

Watch a quick video showcasing how to use Retention-AI:



## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Development](#model-development)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [API Development](#api-development)
- [Deployment](#deployment)
- [Usage](#usage)
- [Contact](#contact)

## Overview

overview of your project, its goals, and its significance in the field of customer retention.

## Key Features

- Early Churn Prediction
- Data-Driven Insights
- Scalable Architecture
- End-to-End Workflow

## Installation


## Project Structure

retention-ai/
├── config/
├── data/
├── early-churn-detection/
├── models/
├── notebooks/
├── src/
├── tests/
├── Procfile
├── README.md
├── requirements.txt
└── runtime.txt


## Data Processing
# Data Processing  

## Dataset Used: Telco Customer Churn  
The **Telco Customer Churn** dataset was selected as it contains customer demographics, account details, and service usage information, making it suitable for predicting churn.  

## Data Cleaning & Preprocessing  

### 1. Handling Missing Values  
- Identified missing or incorrect values in the **TotalCharges** column.  
- Converted it to numeric format (`float64`), replacing non-numeric values with `NaN`.  
- Imputed missing values with `0`.  

### 2. Encoding Categorical Variables  
- **Binary Encoding:** Converted categorical variables with two unique values (e.g., `gender`, `Partner`, `Churn`) into binary (`0` and `1`).  
- **One-Hot Encoding:** Applied one-hot encoding for categorical columns with more than two values (`Contract`, `InternetService`, `PaymentMethod`), dropping the first category to avoid multicollinearity.  

### 3. Handling "No phone service"/"No internet service" Entries  
- Replaced `"No phone service"` and `"No internet service"` with `"No"` in relevant columns (`MultipleLines`, `OnlineSecurity`, `OnlineBackup`, etc.).  
- Mapped `"No"` to `0` and `"Yes"` to `1`.  

### 4. Feature Type Adjustments  
- Ensured one-hot encoded columns were correctly cast to integers (`int`).  

### 5. Dropping Unnecessary Columns  
- Removed `customerID`, as it is an identifier with no predictive value.  

### 6. Saving Processed Data  
- The cleaned dataset was saved as `telco_cleaned.csv` in the processed data folder for model training.  

### Absence of Temporal Data  
This dataset lacks a **time-series** component, meaning it does not track customer behavior over time. This limits the model’s ability to detect gradual disengagement trends, which would typically be useful for predicting churn **before** obvious signs appear.  

📂 For full preprocessing details, check [`data-processing.ipynb`](notebooks/data-processing.ipynb).  


## Model Development

### Training

model training process, feature selection, algorithm choice, and hyperparameter tuning.

### Evaluation

evaluation of model, metrics used and their significance

## API Development

structure and functionality of your FastAPI application (main.py).  key endpoints and their purposes.

## Deployment

deployed the project on Render

## Usage

instructions on how to use the project, including how to make predictions using the deployed model.


## Contact

[Maimuna Zaheer] - [mz2934@nyu.edu]

Project Link: [https://github.com/itserror404/retention-ai](https://github.com/itserror404/retention-ai)

