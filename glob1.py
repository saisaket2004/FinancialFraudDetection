import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("C:\\Users\\tadak\\Downloads\\archive\\Synthetic_Financial_datasets_log.csv")

# Check for non-numeric columns (e.g., 'nameOrig', 'nameDest', or other IDs) and drop them
# Assuming 'nameOrig', 'nameDest' are non-numeric columns that shouldn't be processed
data = data.drop(['nameOrig', 'nameDest'], axis=1, errors='ignore')  # Drop non-numeric columns

# Convert categorical columns (e.g., 'type') to numeric using LabelEncoder or OneHotEncoder
# Example: If you have a 'type' column, encode it
if 'type' in data.columns:
    label_encoder = LabelEncoder()
    data['type'] = label_encoder.fit_transform(data['type'])

# Handle the target variable and features
X = data.drop('isFraud', axis=1)  # Features (all columns except 'isFraud')
y = data['isFraud']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply scaling to numerical features only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=['float64', 'int64']))
X_test_scaled = scaler.transform(X_test.select_dtypes(include=['float64', 'int64']))

# Replace the scaled features back into the original dataframe
X_train[X_train.select_dtypes(include=['float64', 'int64']).columns] = X_train_scaled
X_test[X_test.select_dtypes(include=['float64', 'int64']).columns] = X_test_scaled

# Initialize and train a model (Logistic Regression as an example)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Precision, Recall, F1-Score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

print(f"ROC-AUC: {roc_auc:.4f}")

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plotting ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
