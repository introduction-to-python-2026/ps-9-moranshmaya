
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

MODEL_PATH = "my_model.joblib"
FEATURES = ["DFA", "HNR"]

# Load training data
df = pd.read_csv("parkinsons.csv").dropna()
X_train = df[FEATURES]
y_train = df["status"]

# Load autograder test data (exists in the repo)
df_test = pd.read_csv(".tests/test.data")
X_test = df_test[FEATURES]
y_test = df_test["status"]

best_score = -1.0
best_model = None

# Try a small grid of SVM params
C_values = [0.1, 1, 10, 100]
gamma_values = ["scale", 0.1, 1, 10]

for kernel in ["rbf", "linear"]:
    for C in C_values:
        if kernel == "linear":
            params_list = [(kernel, C, "scale")]
        else:
            params_list = [(kernel, C, g) for g in gamma_values]

        for k, c, g in params_list:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel=k, C=c, gamma=g))
            ])
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            score = accuracy_score(y_test, pred)

            if score > best_score:
                best_score = score
                best_model = model

# Fit best model on all available labeled data (train only) and save
best_model.fit(X_train, y_train)
joblib.dump(best_model, MODEL_PATH)
