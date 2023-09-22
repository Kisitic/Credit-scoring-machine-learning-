# Credit-scoring-machine-learning-
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc

import matplotlib.pyplot as plt

# Load the dataset

data = pd.read_excel('traindata.xlsx')  # Replace 'traindata.csv' with your actual file name



# Preprocessing: Convert categorical variables into numerical using one-hot encoding

data = pd.get_dummies(data, columns=['Occupation', 'payement_of_min_amount', 'payment_behaviour'])



# Split features and target variable

X = data.drop('credit_score', axis=1)  # Features

y = data['credit_score']  # Target variable



# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Standardize features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



# Implement Logistic Regression model

logreg = LogisticRegression()

logreg.fit(X_train_scaled, y_train)



# Implement Decision Tree model

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train_scaled, y_train)



# Make predictions

y_pred_logreg = logreg.predict(X_test_scaled)

y_pred_decision_tree = decision_tree.predict(X_test_scaled)



# Evaluate the models using precision, recall, and F1-score

precision_logreg = precision_score(y_test, y_pred_logreg)

recall_logreg = recall_score(y_test, y_pred_logreg)

f1_logreg = f1_score(y_test, y_pred_logreg)



precision_decision_tree = precision_score(y_test, y_pred_decision_tree)

recall_decision_tree = recall_score(y_test, y_pred_decision_tree)

f1_decision_tree = f1_score(y_test, y_pred_decision_tree)



print("Logistic Regression Metrics:")

print(f"Precision: {precision_logreg:.2f}")

print(f"Recall: {recall_logreg:.2f}")

print(f"F1-score: {f1_logreg:.2f}\n")



print("Decision Tree Metrics:")

print(f"Precision: {precision_decision_tree:.2f}")

print(f"Recall: {recall_decision_tree:.2f}")

print(f"F1-score: {f1_decision_tree:.2f}")



# Make predictions

Y_pred = logreg.predict(X_test_scaled)



# Calculate Confusion Matrix

Conf_matrix = confusion_matrix(y_test, y_pred)

Print(“Confusion Matrix:”)

Print(conf_matrix)



# Calculate ROC curve and AUC

Y_prob = logreg.predict_proba(X_test_scaled)[:, 1]

Fpr, tpr, thresholds = roc_curve(y_test, y_prob)

Roc_auc = auc(fpr, tpr)



# Plot ROC curve

Plt.figure()

Plt.plot(fpr, tpr, color=’darkorange’, lw=2, label=f’ROC curve (area = {roc_auc:.2f})’)

Plt.plot([0, 1], [0, 1], color=’navy’, lw=2, linestyle=’—‘)

Plt.xlim([0.0, 1.0])

Plt.ylim([0.0, 1.05])

Plt.xlabel(‘False Positive Rate’)

Plt.ylabel(‘True Positive Rate’)

Plt.title(‘Receiver Operating Characteristic’)

Plt.legend(loc=”lower right”)

Plt.show()



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

# Implement Random Forest model

random_forest = RandomForestClassifier()

random_forest.fit(X_train_scaled, y_train)



# Implement Gradient Boosting model

gradient_boosting = GradientBoostingClassifier()

gradient_boosting.fit(X_train_scaled, y_train)



# Implement a simple Neural Network

model = Sequential([

    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),

    Dense(64, activation='relu'),

    Dense(1, activation='sigmoid')

])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)



# Make predictions

y_pred_random_forest = random_forest.predict(X_test_scaled)

y_pred_gradient_boosting = gradient_boosting.predict(X_test_scaled)

y_pred_nn = model.predict(X_test_scaled)

y_pred_nn = (y_pred_nn > 0.5).astype(int)



# Evaluate the models using precision, recall, and F1-score

precision_random_forest = precision_score(y_test, y_pred_random_forest)

recall_random_forest = recall_score(y_test, y_pred_random_forest)

f1_random_forest = f1_score(y_test, y_pred_random_forest)



precision_gradient_boosting = precision_score(y_test, y_pred_gradient_boosting)

recall_gradient_boosting = recall_score(y_test, y_pred_gradient_boosting)

f1_gradient_boosting = f1_score(y_test, y_pred_gradient_boosting)



precision_nn = precision_score(y_test, y_pred_nn)

recall_nn = recall_score(y_test, y_pred_nn)

f1_nn = f1_score(y_test, y_pred_nn)



print("Random Forest Metrics:")

print(f"Precision: {precision_random_forest:.2f}")

print(f"Recall: {recall_random_forest:.2f}")

print(f"F1-score: {f1_random_forest:.2f}\n")



print("Gradient Boosting Metrics:")

print(f"Precision: {precision_gradient_boosting:.2f}")

print(f"Recall: {recall_gradient_boosting:.2f}")

print(f"F1-score: {f1_gradient_boosting:.2f}\n")



print("Neural Network Metrics:")

print(f"Precision: {precision_nn:.2f}")

print(f"Recall: {recall_nn:.2f}")

print(f"F1-score: {f1_nn:.2f}")

import pandas as pd



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc

import matplotlib.pyplot as plt

# Implement Random Forest model

random_forest = RandomForestClassifier()

random_forest.fit(X_train_scaled, y_train)



# Make predictions

y_pred = random_forest.predict(X_test_scaled)



# Calculate Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")

print(conf_matrix)



# Calculate ROC curve and AUC

y_prob = random_forest.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)



# Plot ROC curve

plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")

plt.show()

