import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# load data
df = pd.read_csv("parkinsons.csv").dropna()

# features must match config.yaml
X = df[["DFA", "HNR"]]
y = df["status"]

# split (not required by checker, but fine)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# pipeline: scaler + model (so checker can send raw X)
model = Pipeline([
    ("scaler", MinMaxScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=3))
])

model.fit(X_train, y_train)

# save model to path from config.yaml
joblib.dump(model, "my_model.joblib")
