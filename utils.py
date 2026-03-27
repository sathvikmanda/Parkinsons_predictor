import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }


def risk_level(prob):
    if prob < 0.6:
        return "Low Risk"
    elif prob < 0.8:
        return "Medium Risk"
    else:
        return "High Risk"