# mlops_assignment/
# ├── data/
# │   ├── train.csv
# │   ├── test.csv
# ├── src/
# │   ├── train_model.py
# │   ├── evaluate_model.py
# │   ├── drift_check.py
# ├── .github/
# │   └── workflows/
# │       └── ci.yml
# ├── requirements.txt
# └── README.md


import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# Log parameters, metrics, and model
mlflow.log_param("n_estimators", 50)
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.log_model(model, "model")

print("Logged accuracy:", acc)