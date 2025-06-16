import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load data and preprocessing objects
df = pd.read_csv("../data/diabetes_prediction_dataset.csv")
print(df.info())
df = pd.get_dummies(df, columns=["gender", "smoking_history"], drop_first=True)

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

with open("../models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

# Report
print("\n📄 Classification Report:\n")
print(classification_report(y, y_pred))

# Confusion matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
