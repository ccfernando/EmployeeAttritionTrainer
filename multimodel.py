import pandas as pd
import sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import joblib  # For saving the scaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier  # Import MLPClassifier

from sklearn.neural_network import MLPClassifier  # Import MLPClassifier

# Load dataset
df = pd.read_csv('data/train.csv')

# Encode target variable (Attrition)
label_encoder = LabelEncoder()
df['Attrition'] = label_encoder.fit_transform(df['Attrition'])

# Remove unnecessary columns
columns_to_remove = ['Over18', 'EmployeeCount', 'EmployeeNumber', 'StandardHours']
df.drop(columns=columns_to_remove, inplace=True)

# One-hot encode categorical variables (dropping first column to avoid multicollinearity)
df = pd.get_dummies(df, drop_first=True)

# Handle missing values
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Separate features and target
X = df.drop('Attrition', axis=1).values
y = df['Attrition'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler as a pickle file

# Define hyperparameters for the algorithms
param_dist_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_dist_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

param_dist_lr = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear', 'saga']
}

param_dist_nn = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],  # Number of neurons in hidden layers
    'activation': ['relu', 'tanh'],  # Activation function for the hidden layers
    'solver': ['adam', 'sgd'],  # Optimization algorithm
    'learning_rate': ['constant', 'invscaling', 'adaptive'],  # Learning rate
    'max_iter': [200, 500, 1000]  # Number of iterations
}

param_dist_lgbm = {
    'num_leaves': [31, 50, 100],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}

# Initialize the models
models = {
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Neural Network': MLPClassifier(),  # Replaced KNN with Neural Network
    'LightGBM': lgb.LGBMClassifier()
}

# Hyperparameter tuning for each model
param_dists = {
    'Random Forest': param_dist_rf,
    'Support Vector Machine': param_dist_svm,
    'Logistic Regression': param_dist_lr,
    'Neural Network': param_dist_nn,  # Neural Network hyperparameters
    'LightGBM': param_dist_lgbm
}

# Use Stratified KFold for cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

best_models = {}
best_scores = {}

# Iterate over each model
for model_name in models:
    print(f"\nTuning hyperparameters for {model_name}...")

    model = models[model_name]
    param_dist = param_dists[model_name]

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, scoring='accuracy',
                                       cv=cv, verbose=1, n_jobs=-1, random_state=42)

    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Get the best model
    best_model = random_search.best_estimator_
    best_models[model_name] = best_model
    best_scores[model_name] = random_search.best_score_

    print(f"Best parameters for {model_name}: {random_search.best_params_}")

# Evaluate the models on the test set
test_scores = {}

for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    test_scores[model_name] = accuracy
    print(f"{model_name} Test Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score for {model_name}: {f1:.2f}")
    print(f"Confusion Matrix for {model_name}:")
    print(cm)
    print(f"Classification Report for {model_name}:")
    print(report)



# Save the best model (based on highest test accuracy)
best_model_name = max(test_scores, key=test_scores.get)
best_model = best_models[best_model_name]
joblib.dump(best_model, f'{best_model_name}_model.pkl')
print(f"Best model {best_model_name} saved.")
