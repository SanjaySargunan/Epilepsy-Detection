import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib  # For saving and loading models
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

# Load the data
data = pd.read_csv('audio_features.csv')

# Augment the data by replicating it
augmented_data = pd.concat([data] * 5, ignore_index=True)

X = augmented_data.drop('label', axis=1)
y = augmented_data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'eggscaler.sav')

# Define CatBoost model
cat_model = CatBoostClassifier(verbose=0, iterations=500, learning_rate=0.1, depth=6)

# Define XGBoost model
xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metric='mlogloss')

# Create Voting Classifier with soft voting
voting_clf = VotingClassifier(
    estimators=[('catboost', cat_model), ('xgboost', xgb_model)],
    voting='soft'
)

# Train the model
voting_clf.fit(X_train_scaled, y_train)

# Save the trained model
joblib.dump(voting_clf, 'Egg_ensemble.sav')

# Predict on test data
y_pred = voting_clf.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

report = classification_report(y_test, y_pred)
print(report)

confusion = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
