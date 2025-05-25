EduPredict AI – Unlocking Student Potential with AI!
EduPredict AI is a Streamlit web application designed to predict student performance across Math, Reading, and Writing scores based on demographic and background information. It uses a Multi-Task Learning model (deep learning with TensorFlow) and provides visual analytics, model training, and prediction in an intuitive interface.

🚀 Features
🔐 Secure Login System
📂 Upload & Preview CSV Dataset
⚙️ Automated Data Preprocessing (Encoding + Scaling)
🧠 Multi-Task Learning Model using TensorFlow
📈 Model Training with Live Feedback
📊 Correlation Heatmap of Predictions
📌 Actual vs Predicted Score Visualizations
🎯 Score Prediction for New Student Inputs
📜 Sample Classification Report (Table Format)

🧠 Tech Stack
Frontend/UI: Streamlit
Data Handling: pandas, numpy
ML/DL: TensorFlow, Keras, Scikit-learn
Visualization: matplotlib, seaborn

1.Clone the repository
git clone https://github.com/your-username/EduPredict-AI.git
cd EduPredict-AI

2.Install dependencies
pip install -r requirements.txt

3.Run the Streamlit app
streamlit run edupredict_app.py

4.Login credentials
Username: admin
Password: password

5.📊 Dataset Requirements
File Type: CSV
Expected Columns:
gender
race/ethnicity
parental level of education
lunch
test preparation course
math score
reading score
writing score

🔍 Sample Dataset
You can use the StudentsPerformance.csv dataset from Kaggle.

✅ Future Enhancements
User-based session storage for multiple users
Auto-grading using classification for grade levels
Model performance summary & export options
PDF report generation for predictions


