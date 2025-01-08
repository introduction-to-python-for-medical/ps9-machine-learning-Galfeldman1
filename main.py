%load_ext autoreload
%autoreload 2

# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('/content/parkinsons.csv')

# Display the first few rows of the DataFrame (optional)
df.head()
features = [ 'DFA', 'PPE']
target = 'status'

# Prepare features and target
X = df[features]
y = df[target]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.svm import SVC

# Initialize the SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Train the model
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate the accuracy
accuracy = accuracy_score(y_val, y_pred)

print(f"Accuracy: {accuracy}")
import joblib

joblib.dump(model, 'my_model.joblib')
