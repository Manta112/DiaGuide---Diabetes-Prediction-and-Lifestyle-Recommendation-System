# DiaGuide - AI Diabetes Prediction System

## Steps to Run

1. Open Command Prompt in this folder
2. Install dependencies:
   pip install -r requirements.txt

3. Train models (run ONCE - fixes all compatibility issues):
   python train_model.py

4. Start the app:
   streamlit run app.py

5. Browser opens automatically at http://localhost:8501

## Project Structure
DiaGuide_Streamlit/
├── app.py               ← Streamlit app (run this)
├── train_model.py       ← Run once to train models
├── requirements.txt
├── dataset/
│   └── diabetes.csv     ← PIMA dataset (offline)
└── models_saved/        ← Auto-created by train_model.py
