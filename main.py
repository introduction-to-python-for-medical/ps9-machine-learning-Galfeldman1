import pandas as pd
df = pd.read_csv('/content/parkinsons.csv')
df.head()
features = ['HNR', 'RPDE']
target = 'status'
X = df[features]
y = df[target]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(3)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")


