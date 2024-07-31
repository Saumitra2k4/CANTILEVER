#please download the creditcard dataset from kaggle 

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data
df = pd.read_csv('creditcard.csv')

# Data exploration
print(df.head())
print(df.info())
print(df.describe())

# Feature scaling
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df.drop(columns=['Time'], inplace=True)

# Splitting the data
X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling imbalanced data
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Building a Neural Network with Keras
model = Sequential()
model.add(Dense(16, input_dim=X_train_res.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_res, y_train_res, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred_keras = (model.predict(X_test) > 0.5).astype("int32")

print("Neural Network Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_keras))
print("Neural Network Classification Report:")
print(classification_report(y_test, y_pred_keras))

# ROC-AUC Curve
y_pred_keras_prob = model.predict(X_test).ravel()

fpr, tpr, thresholds = roc_curve(y_test, y_pred_keras_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
