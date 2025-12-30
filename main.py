# Download the data from your GitHub repository
https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit

import pandas as pd

df = pd.read_csv('parkinsons.csv')
# Drop rows with any missing values
df = df.dropna()
df.head()

import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df, hue='status')

plt.show()

x = df[['DFA', 'HNR']]
y = df['status']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

print(x_scaled[:5])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=44)

print(f'x_train shape: {x_train.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)

model

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

accuracy
import joblib
joblib.dump(model, "my_model.joblib")
           
