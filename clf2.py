import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import classification_report


# Assuming 'final_df' is your DataFrame with the HSV values and the corresponding labels

final_df = pd.read_csv(r'C:\Users\anto3\Desktop\DAKI\miniprojekt-DUAS-main\concatenated_data.csv')

final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and labels
X = final_df[['Hue', 'Saturation', 'Value']]  # Features
y = final_df['Label']  # Labels

# Encode string labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier
clf = xgb.XGBClassifier(objective='multi:softprob', num_class=len(label_encoder.classes_), seed=42)

# Train the classifier using the encoded labels
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Evaluate the classifier using the encoded labels
performance = classification_report(y_test, predictions)
print(performance)



accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
