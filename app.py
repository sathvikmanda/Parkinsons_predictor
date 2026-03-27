import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from utils import evaluate_model, risk_level

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(layout="wide", page_title="Parkinson Dashboard")

# -------------------------------
# CUSTOM CSS (IMPORTANT)
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #f4f8fb;
}

.card {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}

.title {
    font-size: 32px;
    font-weight: bold;
    color: #0a9396;
}

.subtitle {
    color: #555;
    font-size: 16px;
}

.metric-card {
    background-color: #e6f4f1;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODELS
# -------------------------------
rf = pickle.load(open("model/rf_model.pkl", "rb"))
svm = pickle.load(open("model/svm_model.pkl", "rb"))
knn = pickle.load(open("model/knn_model.pkl", "rb"))

df = pd.read_csv("data/parkinsons.csv")

if "name" in df.columns:
    df = df.drop("name", axis=1)

X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

feature_names = X.columns

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="title">🏥 Parkinson’s Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-based Voice Analysis for Early Diagnosis</div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Diagnosis", "📊 Analytics", "📂 Batch Processing"])

# -------------------------------
# TAB 1: DIAGNOSIS
# -------------------------------
with tab1:

    st.markdown("### 🧑 Patient Input Panel")

    inputs = {}

    # 🟢 SECTION 1
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎤 Voice Pitch")

        c1, c2, c3 = st.columns(3)
        inputs["MDVP:Fo(Hz)"] = c1.number_input("Average Pitch", 120.0)
        inputs["MDVP:Fhi(Hz)"] = c2.number_input("Max Pitch", 150.0)
        inputs["MDVP:Flo(Hz)"] = c3.number_input("Min Pitch", 100.0)

        st.markdown('</div>', unsafe_allow_html=True)

    # 🟢 SECTION 2
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📉 Voice Stability")

        c1, c2 = st.columns(2)

        inputs["MDVP:Jitter(%)"] = c1.number_input("Jitter (%)", 0.01)
        inputs["MDVP:Jitter(Abs)"] = c1.number_input("Jitter Abs", 0.00001)
        inputs["MDVP:RAP"] = c1.number_input("RAP", 0.01)
        inputs["MDVP:PPQ"] = c1.number_input("PPQ", 0.01)
        inputs["Jitter:DDP"] = c1.number_input("DDP", 0.01)

        inputs["MDVP:Shimmer"] = c2.number_input("Shimmer", 0.03)
        inputs["MDVP:Shimmer(dB)"] = c2.number_input("Shimmer dB", 0.5)
        inputs["Shimmer:APQ3"] = c2.number_input("APQ3", 0.03)
        inputs["Shimmer:APQ5"] = c2.number_input("APQ5", 0.03)
        inputs["MDVP:APQ"] = c2.number_input("APQ", 0.03)
        inputs["Shimmer:DDA"] = c2.number_input("DDA", 0.03)

        st.markdown('</div>', unsafe_allow_html=True)

    # 🟢 SECTION 3
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔊 Voice Quality")

        c1, c2 = st.columns(2)
        inputs["NHR"] = c1.number_input("Noise Ratio", 0.02)
        inputs["HNR"] = c2.number_input("Harmonics Ratio", 20.0)

        st.markdown('</div>', unsafe_allow_html=True)

    # 🟢 SECTION 4
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("⚙️ Advanced Features")

        c1, c2, c3 = st.columns(3)
        inputs["RPDE"] = c1.number_input("RPDE", 0.5)
        inputs["DFA"] = c2.number_input("DFA", 0.7)
        inputs["spread1"] = c3.number_input("Spread1", -5.0)

        c4, c5 = st.columns(2)
        inputs["spread2"] = c4.number_input("Spread2", 0.2)
        inputs["D2"] = c5.number_input("D2", 2.0)

        inputs["PPE"] = st.number_input("PPE", 0.3)

        st.markdown('</div>', unsafe_allow_html=True)

    model_choice = st.selectbox("Model", ["Random Forest", "SVM", "KNN"])

    if st.button("🧠 Run Diagnosis"):

        input_array = np.array([inputs[col] for col in feature_names]).reshape(1, -1)

        if model_choice == "Random Forest":
            model = rf
        elif model_choice == "SVM":
            model = svm
        else:
            model = knn

        pred = model.predict(input_array)
        prob = model.predict_proba(input_array)
        confidence = max(prob[0]) * 100

        st.markdown("### 📊 Diagnosis Result")

        col1, col2, col3 = st.columns(3)

        col1.metric("Prediction", "Parkinson’s" if pred[0]==1 else "Healthy")
        col2.metric("Confidence", f"{confidence:.2f}%")
        col3.metric("Risk Level", risk_level(confidence/100))

        st.bar_chart(prob[0])

# -------------------------------
# TAB 2: ANALYTICS
# -------------------------------
with tab2:

    st.markdown("### 📊 Model Analytics")

    models = {"Random Forest": rf, "SVM": svm, "KNN": knn}
    results = []

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        results.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"]
        })

    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

    st.subheader("Accuracy Comparison")
    st.bar_chart(df_results.set_index("Model")["Accuracy"])

    selected_model = st.selectbox("Confusion Matrix", list(models.keys()))
    cm = evaluate_model(models[selected_model], X_test, y_test)["confusion_matrix"]

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

# -------------------------------
# TAB 3: BATCH
# -------------------------------
with tab3:

    st.markdown("### 📂 Batch Processing")

    file = st.file_uploader("Upload CSV")

    if file:
        data = pd.read_csv(file)
        preds = rf.predict(data)
        data["Prediction"] = preds
        st.write(data)