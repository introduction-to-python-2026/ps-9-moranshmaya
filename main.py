import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# load data
df = pd.read_csv("parkinsons.csv").dropna()

# features (must match config.yaml)
X = df[["DFA", "HNR"]]
y = df["status"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model (no scaling!)
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)

# save model
joblib.dump(model, "my_model.joblib")
