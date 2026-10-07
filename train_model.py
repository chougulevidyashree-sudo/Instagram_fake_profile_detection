import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset

df = pd.read_csv('dataset.csv')

# Features and labels
X = df.drop('label', axis=1)
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, predictions))

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print('Model Trained Successfully')