import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/parkinsons.csv")

if "name" in df.columns:
    df = df.drop("name", axis=1)

X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
rf = RandomForestClassifier(n_estimators=200)
svm = SVC(probability=True)
knn = KNeighborsClassifier()

rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)

# Save models
pickle.dump(rf, open("model/rf_model.pkl", "wb"))
pickle.dump(svm, open("model/svm_model.pkl", "wb"))
pickle.dump(knn, open("model/knn_model.pkl", "wb"))

print("✅ All models trained & saved")