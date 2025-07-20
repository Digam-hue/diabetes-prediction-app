# 🩺 Ultimate Diabetes Prediction: An End-to-End Machine Learning Project

![Diabetes Prediction App Demo](/images/app_pic.png)

Welcome to the Ultimate Diabetes Prediction project! This repository documents the complete journey of building a machine learning model to predict diabetes risk, culminating in a live, interactive web application built with Streamlit.

**Live Demo:** [**🚀 Try the Live App Here!**](https://smart-diabetes-check.streamlit.app/)

git clone https://github.com/Digam-hue/diabetes-prediction-app.git


---

## The Problem: Can We Predict Diabetes Early?

Diabetes is a global health crisis affecting millions. Early detection is crucial to prevent severe complications, but often the signs are missed. This project explores whether machine learning can help identify at-risk individuals early by analyzing common medical data.

My goal was to build a complete, real-world solution: from messy raw data to a clean, deployable application. This project showcases the entire machine learning lifecycle, demonstrating skills in data cleaning, exploratory analysis, model training, and web deployment.

---

## 🗺️ Project Workflow

This project follows a structured machine learning pipeline:

1.  **Data Cleaning & EDA:** Investigated the raw dataset to clean inconsistencies, handle outliers, and uncover initial patterns through visualization.
2.  **Feature Engineering & Preprocessing:** Transformed and scaled the data to prepare it for modeling, including one-hot encoding for categorical features and stratified splitting to handle class imbalance.
3.  **Modeling & Hyperparameter Tuning:** Trained multiple classification models (from Logistic Regression to Gradient Boosting) and used `RandomizedSearchCV` to fine-tune the best performers for optimal accuracy and reliability.
4.  **Deployment:** Developed and deployed a user-friendly web application with Streamlit, allowing anyone to interact with the final model.

---

## 📊 The Dataset: Patient Medical Records

The project is based on the [Dataset of Diabetes](https://data.mendeley.com/datasets/wj9rwkp9c2/1), containing anonymized medical data.

**Key Features Include:**
*   **Demographics:** `AGE`, `Gender`
*   **Biometrics:** `BMI` (Body Mass Index)
*   **Lab Results:** `HbA1c`, `Cholesterol`, `Triglycerides`, `Urea`, `Creatinine`, and lipid profiles (`HDL`, `LDL`, `VLDL`).
*   **Target:** `CLASS` - Diagnosis (Non-Diabetic, Pre-Diabetic, or Diabetic).

---

## 💡 Key Insights from EDA

My analysis revealed two critical insights that shaped the modeling strategy:

1.  **HbA1c is a Powerful Predictor:** The boxplot below clearly shows that HbA1c levels are highly correlated with diabetes status. Patients diagnosed as Diabetic have significantly higher HbA1c levels, confirming its importance as a primary diagnostic marker. This strong signal gave confidence that an effective model was achievable.

    ![HbA1c Distribution by Class](/images/boxplot.png)

2.  **The Challenge of Class Imbalance:** The dataset was heavily skewed towards diabetic patients. This classic problem meant that simply optimizing for accuracy would be misleading. My solution was to use stratified data splitting and focus on metrics like the **weighted F1-score** and **ROC AUC** to ensure the model performed well across all classes, especially the rare ones.

---

## ⚙️ Model Performance Comparison

After training and tuning, the models were evaluated on the unseen test set. The Random Forest classifier demonstrated superior performance.

| Model                 | Test Accuracy | F1-Score (Weighted) | ROC AUC |
|-----------------------|:-------------:|:-------------------:|:-------:|
| **🏆 Random Forest**  | **0.9950**    | **0.9949**          | **0.998** |
| Gradient Boosting     | 0.9949        | 0.9949              | 0.997   |
| Support Vector (SVC)  | 0.9600        | 0.9563              | 0.994   |
| Logistic Regression   | 0.9450        | 0.9335              | 0.990   |

The **Random Forest Classifier** was selected as the final model due to its outstanding and balanced performance, indicating high accuracy and reliability in distinguishing between patient classes.

---

## 🚀 Run the App Locally

To explore and run this app on your local machine:

```bash
# 1. Clone the Repository
git clone https://github.com/Digam-hue/diabetes-prediction-app.git
cd diabetes-prediction-app

# 2. Create and Activate a Virtual Environment
# On Unix/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run the Streamlit App
streamlit run app.py
The app will open in your browser.
If not, visit: http://localhost:8501

🛠️ Tech Stack
Category	Tools
Data Analysis	Python, Pandas, NumPy
Visualization	Matplotlib, Seaborn
Machine Learning	Scikit-learn
Web App	Streamlit
Deployment	Streamlit Community Cloud
Version Control	Git & GitHub

🙏 Thank You
Thanks for checking out the project!
If you found it useful or interesting, feel free to ⭐️ the repo.


🔗 Connect with me on LinkedIn (link: https://www.linkedin.com/in/digambar-baditya-b522b12a5/)