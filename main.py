import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# load data
df = pd.read_csv("parkinsons.csv").dropna()

# features must match config.yaml
X = df[["DFA", "HNR"]]
y = df["status"]

# model: scale + SVM
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", C=10, gamma="scale"))
])

model.fit(X, y)

# save model
joblib.dump(model, "my_model.joblib")
