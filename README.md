🩺 Ultimate Diabetes Prediction: From Data to Deployment
![Alt text](/images/app_pic.png)

Welcome to the Ultimate Diabetes Prediction project! This repository documents the end-to-end journey of building a machine learning model to predict diabetes risk, culminating in a live, interactive web application.

**Live Demo:** [**Try the app here!**](https://smart-diabetes-check.streamlit.app/)

git clone https://github.com/Digam-hue/diabetes-prediction-app.git
The Story: Why This Project?

Diabetes is a global health crisis, affecting millions of lives. Early detection is crucial to managing the condition and preventing severe complications. But can we leverage data to identify at-risk individuals before it's too late?

This project was born from that question. My goal was to go beyond just training a model and to build a complete, real-world solution: from messy, raw data to a clean, deployable application that could, in theory, help a user understand their risk. It’s a journey through the entire machine learning lifecycle, demonstrating skills in data cleaning, exploratory analysis, modeling, and web development.

🗺️ The Workflow: Our Project Roadmap

Every good journey has a map. Ours looked like this, moving from raw materials to a finished product:

Data Cleaning & EDA: Like a detective, I started by examining the raw dataset for clues and inconsistencies. The goal was to understand the data's story, clean up any messy parts, and uncover hidden patterns.

Feature Engineering & Preprocessing: With a clean dataset, the next step was to prepare the features for our machine learning models. This involved transforming categorical data into numbers and scaling features to ensure fairness and accuracy.

Modeling & Tuning: This is where the magic happens! I trained a variety of classification models to see which could best learn the patterns of diabetes. Not content with off-the-shelf performance, I used hyperparameter tuning to fine-tune the best models for peak accuracy.

Deployment: A model is only useful if people can interact with it. I built a simple and intuitive web application using Streamlit, allowing anyone to input their data and receive a prediction.

📊 The Data: Understanding Our Patient Records

The foundation of this project is the Dataset of Diabetes, which contains anonymized medical and demographic data from patients.

Key Features:

Demographics: AGE, Gender

Biometrics: BMI (Body Mass Index)

Lab Results: HbA1c (Glycated Hemoglobin), Chol (Cholesterol), TG (Triglycerides), Urea, Cr (Creatinine), and lipid profiles (HDL, LDL, VLDL).

Target Variable: CLASS - The patient's diagnosis (Non-Diabetic, Pre-Diabetic, or Diabetic).

💡 Key Findings from Exploratory Data Analysis (EDA)

Before building any models, I let the data tell its story. Two key insights emerged:

HbA1c is the Strongest Predictor: The analysis revealed a very strong relationship between a patient's HbA1c level and their diabetes status. As expected, patients diagnosed as Diabetic had significantly higher HbA1c levels, confirming its importance as a primary diagnostic marker. This gave me confidence that our models would have a powerful feature to learn from.

![Alt text](/images/boxplot.png)

A Challenging Class Imbalance: The dataset was heavily skewed towards diabetic patients, with very few records for pre-diabetic individuals. This presented a classic machine learning challenge: how do you build a model that can accurately identify a rare class? This discovery directly influenced my modeling strategy, forcing me to use techniques like stratified splitting and evaluation metrics like the F1-score, which are robust to class imbalance.

⚙️ Model Performance: Finding the Champion

After training and tuning several models, a clear winner emerged. Here’s how the top contenders performed on the unseen test data:

Model	            Test Accuracy	Test F1-Score(Weighted) Test ROC AUC
Random Forest	     0.9950	        0.9949	                0.9978
Gradient Boosting	 0.9949	        0.9949	                0.9969
Support Vector (SVC) 0.9600	        0.9563	                0.9935
Logistic Regression	 0.9450	        0.9335	                0.9901

The Random Forest Classifier was selected as the final model due to its outstanding performance across all metrics, especially its near-perfect F1-score and ROC AUC. This indicates it is not only accurate but also highly reliable at distinguishing between the different patient classes.

🚀 How to Run This Project Locally

Interested in exploring the code yourself ? Here’s how to get the app running on your machine.

Prerequisites:

Python 3.8+

Git

Setup Steps:

# 1. Clone the repository
git clone https://github.com/Digam-hue/diabetes-prediction-app.git
cd diabetes-prediction-app

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py



Thank you for checking out my project. Feel free to connect with me on LinkedIn
link: https://www.linkedin.com/in/digambar-baditya-b522b12a5/

# diabetes-prediction-app
A Machine Learning web app to predict diabetes using logistic regression and other models
