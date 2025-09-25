😴 Sleep Health Predictor

An interactive machine learning app built with Streamlit that predicts sleep health outcomes or estimates continuous values (depending on your dataset).
Designed to be super easy for anyone to use — just upload your data, click Train, and start making predictions.

<p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-blue" /> <img src="https://img.shields.io/badge/Streamlit-App-red" /> <img src="https://img.shields.io/badge/ScikitLearn-ML-yellowgreen" /> </p>
🚀 Features

Layman-friendly UI
No technical jargon — just three simple steps:

Choose the target column

Train the model

Get results (single entry or batch upload)

Automatic task detection

If your target is categorical → trains a classifier

If your target is numeric → trains a regressor

Robust training pipeline

Preprocessing: imputation, scaling, one-hot encoding

Handles rare classes gracefully (drops labels with <2 samples)

Uses Random Forest (classification or regression)

Prediction modes

Single entry: Fill out a form → see a result instantly

Batch upload: Upload a CSV → download predictions file

Safe schema alignment
Automatically aligns input columns with training schema (no “missing column” errors, even if Person ID or other identifiers are present).

🧰 Tech Stack

Python 3.10+

Streamlit → interactive app

Pandas → data handling

Scikit-learn → preprocessing & ML models

Joblib → model persistence

Plotly (optional) → quick visualizations in EDA

📂 Project Structure
Sleep-Health-Predictor/
│
├── app.py                 # Streamlit app (UI + prediction logic)
├── requirements.txt       # Python dependencies
├── data/
│   └── sleep.csv          # Dataset (replace with your own)
└── src/
    ├── __init__.py
    ├── preprocess.py      # Feature engineering & preprocessing
    └── train.py           # Training logic (classification/regression)

⚡ Getting Started
1. Clone the repo
git clone https://github.com/your-username/sleep-health-predictor.git
cd sleep-health-predictor

2. Install dependencies
pip install -r requirements.txt

3. Add your dataset

Place your CSV inside the data/ folder.

Default file name is sleep.csv.

Example columns might include:

Person ID (ignored automatically)

Age, Gender, Occupation

Sleep Duration, Heart Rate, Stress Level

Sleep Disorder (target column)

4. Run the app
streamlit run app.py

🎯 Usage Walkthrough
Step 1 — Choose target

From the dropdown, pick the column you want to predict (e.g., Sleep Disorder).

Step 2 — Train

Click Train. The app:

Splits your data into training/test sets

Builds the preprocessing + model pipeline

Saves the trained model into sleep_model.joblib

Step 3 — Predict

Single Entry: Fill in form values → get Our best guess (classification) or Estimated value (regression).

Batch Upload: Upload a CSV → download predictions file with an added prediction column.

📊 Example Results
Single Entry
Input: 
  Age = 30
  Sleep Duration = 6.5
  Stress Level = High
  Occupation = Healthcare
Output:
  Our best guess: Insomnia

Batch Upload

Input file (input.csv):

Age,Sleep Duration,Stress Level,Occupation
28,7,Low,Engineer
42,5,High,Healthcare


Output file (predictions.csv):

Age,Sleep Duration,Stress Level,Occupation,prediction
28,7,Low,Engineer,No Disorder
42,5,High,Healthcare,Insomnia

📸 Screenshots
<p align="center"> <img src="docs/screenshot-train.png" width="600" /> <br/> <em>Step 2 — Training the model</em> </p> <p align="center"> <img src="docs/screenshot-predict.png" width="600" /> <br/> <em>Step 3 — Making predictions</em> </p>
🔮 Future Improvements

Add model selector (XGBoost, Logistic Regression, etc.)

Add explainability (feature importance, SHAP plots)

Add ROC/PR curves in the training results

Deploy app to Streamlit Cloud for one-click demos

🙌 Author

Jaskirat Singh Nandhra
Data Engineer • Machine Learning Enthusiast • Set Designer 🎭
LinkedIn
 | GitHub
