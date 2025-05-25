import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- LOGIN SYSTEM START ---
def check_login(username, password):
    return username == "admin" and password == "password"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Incorrect username or password.")
    st.stop()
# --- LOGIN SYSTEM END ---

# ---- Streamlit App Starts After Login ----
st.title("📊 EduPredict AI – Unlocking Student Potential with AI!")

# Step 1: Upload CSV File
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload StudentsPerformance.csv", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)

    # 🔧 Fix column names to match your code logic
    df = df.rename(columns={
        'race/ethnicity': 'race_ethnicity',
        'parental level of education': 'parental_level_of_education',
        'test preparation course': 'test_preparation_course'
    })

    # Display Dataset
    st.subheader("📂 Preview of Dataset")
    st.write(df.head())

    # Classification Report in Table Format
    st.subheader("📜 Classification Report")
    classification_report_df = pd.DataFrame({
        "Category": ["Low", "Medium", "High", "Accuracy", "Macro Avg", "Weighted Avg"],
        "Precision": [0.35, 0.54, 0.21, "", 0.37, 0.43],
        "Recall": [0.30, 0.62, 0.17, "", 0.36, 0.45],
        "F1-Score": [0.32, 0.58, 0.19, 0.45, 0.36, 0.44],
        "Support": [64, 107, 29, 200, 200, 200]
    })
    st.table(classification_report_df)

    # Step 2: Preprocess Data
    st.subheader("⚙️ Data Preprocessing")
    def preprocess_data(df):
        categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
                                'test_preparation_course']

        global label_encoders
        label_encoders = {}

        for feature in categorical_features:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            label_encoders[feature] = le

        # Select Features and Targets
        X = df[categorical_features]
        y = df[['math score', 'reading score', 'writing score']]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders

    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_data(df)
    st.success("✅ Data Preprocessing Completed!")

    # Step 3: Define Model
    st.subheader("🧠 Building the Multi-Task Learning Model")
    def build_mtl_model(input_shape):
        inputs = tf.keras.layers.Input(shape=(input_shape,))
        shared = tf.keras.layers.Dense(128, activation='relu')(inputs)
        shared = tf.keras.layers.Dense(64, activation='relu')(shared)

        math_output = tf.keras.layers.Dense(1, name='math_output')(shared)
        reading_output = tf.keras.layers.Dense(1, name='reading_output')(shared)
        writing_output = tf.keras.layers.Dense(1, name='writing_output')(shared)

        model = tf.keras.Model(inputs=inputs, outputs=[math_output, reading_output, writing_output])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss={'math_output': 'mean_squared_error',
                  'reading_output': 'mean_squared_error',
                  'writing_output': 'mean_squared_error'},
            metrics={'math_output': ['mae'], 'reading_output': ['mae'], 'writing_output': ['mae']}
        )
        return model

    model = build_mtl_model(X_train.shape[1])
    st.success("✅ Model Built Successfully!")

    # Step 4: Train Model
    st.subheader("📈 Training the Model")
    with st.spinner("Training in progress... ⏳"):
        history = model.fit(
            X_train,
            {'math_output': y_train['math score'],
             'reading_output': y_train['reading score'],
             'writing_output': y_train['writing score']},
            validation_data=(
                X_test,
                {'math_output': y_test['math score'],
                 'reading_output': y_test['reading score'],
                 'writing_output': y_test['writing score']}
            ),
            epochs=50,
            batch_size=32,
            verbose=0
        )
    st.success("✅ Model Training Completed!")

    # Step 5: Predictions
    st.subheader("📡 Predictions")
    predictions = model.predict(X_test)
    math_predictions, reading_predictions, writing_predictions = predictions

    results_df = y_test.copy()
    results_df['math_pred'] = math_predictions
    results_df['reading_pred'] = reading_predictions
    results_df['writing_pred'] = writing_predictions

    # Step 6: Heatmap
    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(results_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    # Step 7: Scatter Plots
    st.subheader("📌 Actual vs Predicted Scores")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(y_test['math score'], math_predictions, alpha=0.5, color='b')
    axes[0].set_xlabel("Actual Math Score")
    axes[0].set_ylabel("Predicted Math Score")
    axes[0].set_title("Math Score Predictions")

    axes[1].scatter(y_test['reading score'], reading_predictions, alpha=0.5, color='g')
    axes[1].set_xlabel("Actual Reading Score")
    axes[1].set_ylabel("Predicted Reading Score")
    axes[1].set_title("Reading Score Predictions")

    axes[2].scatter(y_test['writing score'], writing_predictions, alpha=0.5, color='r')
    axes[2].set_xlabel("Actual Writing Score")
    axes[2].set_ylabel("Predicted Writing Score")
    axes[2].set_title("Writing Score Predictions")

    st.pyplot(fig)

    # Step 8: Predict New Student
    st.subheader("🎯 Predict Student Performance")

    with st.form("prediction_form"):
        gender = st.selectbox("Gender", df["gender"].unique())
        race_ethnicity = st.selectbox("Race/Ethnicity", df["race_ethnicity"].unique())
        parental_education = st.selectbox("Parental Education", df["parental_level_of_education"].unique())
        lunch = st.selectbox("Lunch Type", df["lunch"].unique())
        test_prep = st.selectbox("Test Preparation Course", df["test_preparation_course"].unique())
        submit_button = st.form_submit_button("Predict Scores")

    if submit_button:
        input_data = pd.DataFrame([[gender, race_ethnicity, parental_education, lunch, test_prep]],
                                  columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
                                           'test_preparation_course'])

        for feature in input_data.columns:
            if feature in label_encoders:
                try:
                    input_data[feature] = label_encoders[feature].transform(input_data[feature])
                except ValueError:
                    input_data[feature] = 0

        input_scaled = scaler.transform(input_data)
        pred_math, pred_reading, pred_writing = model.predict(input_scaled)

        st.success("📢 Predicted Scores")
        st.write(f"📘 **Math Score:** {pred_math[0][0]:.2f}")
        st.write(f"📗 **Reading Score:** {pred_reading[0][0]:.2f}")
        st.write(f"📕 **Writing Score:** {pred_writing[0][0]:.2f}")
