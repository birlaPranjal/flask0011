import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load your dataset (replace 'data.csv' with your actual data file)
data = pd.read_csv("data.csv")

# Split the data into features (X) and target (y)
X = data[['age', 'income', 'investment_experience', 'financial_dependents']]
y = data['risk_appetite']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your machine learning model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a .pkl file
joblib.dump(model, 'decision_tree_model.pkl')
