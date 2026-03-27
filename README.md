# 🧠 Parkinson’s Disease Detection System

A **Machine Learning + Signal Processing based web application** for early detection of Parkinson’s disease using voice features.

---

## 🚀 Project Overview

Parkinson’s Disease (PD) is a progressive neurological disorder that affects movement and speech. Early detection is critical for better management and treatment.

This project uses **machine learning models** to analyze **voice signal features** such as pitch, jitter, shimmer, and noise levels to predict whether a patient is likely to have Parkinson’s disease.

The system is deployed as an **interactive hospital-style dashboard** using Streamlit.

---

## 🎯 Key Features

* 🧑 **Patient Diagnosis Panel**

  * Input voice parameters manually
  * Clean, grouped UI (Pitch, Stability, Quality)

* 🤖 **Multiple ML Models**

  * Random Forest
  * Support Vector Machine (SVM)
  * K-Nearest Neighbors (KNN)

* 📊 **Model Analytics Dashboard**

  * Accuracy comparison
  * Precision & Recall
  * Confusion matrix visualization

* 📈 **Prediction Insights**

  * Confidence score
  * Risk level (Low / Medium / High)

* 📂 **Batch Prediction**

  * Upload CSV files
  * Predict multiple patients at once

* 🏥 **Hospital-style UI**

  * Dashboard layout
  * Metric cards
  * Clean clinical interface

---

## 🧠 Technologies Used

* **Python**
* **Streamlit** (Frontend Dashboard)
* **Scikit-learn** (Machine Learning)
* **Pandas & NumPy** (Data Processing)
* **Matplotlib & Seaborn** (Visualization)

---

## 📂 Project Structure

```
parkinsons_app/
│
├── data/
│   └── parkinsons.csv
│
├── model/
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│
├── train_models.py
├── app.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```
git clone <your-repo-link>
cd parkinsons_app
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Train Models

```
python train_models.py
```

### 4. Run the Application

```
streamlit run app.py
```

---

## 📊 How It Works

1. User inputs voice parameters OR uploads dataset
2. Features are processed and passed to ML models
3. Selected model predicts:

   * Parkinson’s or Healthy
   * Confidence score
   * Risk level
4. Results are visualized in dashboard format

---

## 🧪 Machine Learning Approach

* Dataset: UCI Parkinson’s Dataset
* Feature-based classification using biomedical voice measurements
* Models trained and compared using:

  * Accuracy
  * Precision
  * Recall

---

## ⚠️ Limitations

* Uses pre-extracted voice features (not raw audio in production)
* Not a substitute for clinical diagnosis
* Requires further validation with real-world medical data

---

## 🔮 Future Improvements

* 🎤 Real-time voice recording & feature extraction
* 🧠 Explainable AI (SHAP)
* 🌐 Cloud deployment
* 🏥 Patient history tracking system
* 📊 Advanced analytics dashboard

---

## 📌 Conclusion

This project demonstrates how **machine learning + signal processing** can assist in early detection of neurological disorders like Parkinson’s disease through non-invasive voice analysis.


