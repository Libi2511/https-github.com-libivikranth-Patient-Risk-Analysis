📊 Healthcare Patient Risk Analysis
AI/ML Intern Assignment – Novintix
This project analyzes healthcare patient data to perform EDA, test result prediction, anomaly detection, and AI-based medical recommendation generation using machine learning techniques.
The project demonstrates end-to-end data handling, model development, visualization, and automated healthcare insights.

📁 Project Structure
📦 Healthcare-Patient-Risk-Analysis
│
├── healthcare_dataset.csv
├── healthcare_analysis.ipynb            # Task 1 - EDA Notebook
│
├── task2_supervised.py                  # Task 2 - ML Classification
├── task3_anomaly_detection.py           # Task 3 - Isolation Forest
├── task4_ai_recommendation.py           # Task 4 - AI Doctor Recommendation
│
├── main.py                              # Runs Tasks 2–4 automatically
│
├── predictions.csv                      # Output of Task 2
├── confusion_matrix.png                 # Task 2 Visualization
├── feature_importance.png               # Task 2 Visualization
│
├── detected_anomalies.csv               # Output of Task 3
├── anomaly_detection.png                # Task 3 Visualization
│
└── ai_doctor_recommendation.txt         # Sample recommendation 
📝 Tasks

✅ Task 1 – Exploratory Data Analysis (EDA)
Performed detailed analysis on:
Age distribution
Billing Amount distribution
Room allocation patterns
Frequency of Medical Conditions
Admission Type & Medication usage
Gender & Test Result distribution
Tools used: Pandas, Matplotlib, Seaborn
Output:
📘 healthcare_analysis.ipynb

✅ Task 2 – Supervised Learning: Test Result Prediction
Built a Random Forest Classifier to predict whether patient test results are:
Normal
Abnormal
Inconclusive
Features Used:
Age, Gender, Blood Type, Medical Condition, Billing Amount, Room Number, Admission Type, Medication
Evaluation Metrics:
Accuracy
Precision, Recall, F1-Score
Confusion Matrix
Feature Importance
Key Findings:
Medical Condition, Billing Amount, and Age were the strongest predictors.
Outputs:
predictions.csv
confusion_matrix.png
feature_importance.png

✅ Task 3 – Unsupervised Learning: Anomaly Detection
Used Isolation Forest to detect unusual or fraudulent billing patterns.
Highlights:
Identified ~5% of records as anomalies
Both high-billing and low-billing anomalies detected
Anomalies occurred across all medical conditions
Outputs:
detected_anomalies.csv
anomaly_detection.png

✅ Task 4 – AI Doctor Recommendation Generator
Developed an AI-based system to generate personalized medical recommendations using:
Age
Medical Condition
Medication
Predicted Test Result
Provides:
Condition-specific advice
Age-specific care recommendations
Medication instructions
Emergency warning signs

Output:
ai_doctor_recommendation.txt
🚀 How to Run the Project
1. Install required libraries
pip install pandas numpy scikit-learn matplotlib seaborn
2. Run all tasks automatically
python main.py
3. Run tasks individually
python task2_supervised.py
python task3_anomaly_detection.py
python task4_ai_recommendation.py
4. View EDA Notebook
jupyter notebook healthcare_analysis.ipynb

⭐ Key Insights from the Project
Medical Condition is the most influential factor in predicting test outcomes.
Around 5% of billing entries show abnormal or suspicious values.
Elderly patients require additional monitoring and personalized recommendations.
The AI Recommendation module produces human-like guidance for patient care.
The pipeline is modular, scalable, and fully automated.

🧠 Technologies Used
Python 3
Pandas, NumPy
Scikit-Learn
Matplotlib, Seaborn
Jupyter Notebook
Machine Learning (Supervised + Unsupervised)

📌 Future Improvements
Add deep learning-based medical text recommendations
Deploy as a web API using Flask/FastAPI
Build a dashboard using Streamlit
Integrate real-time patient data monitoring
