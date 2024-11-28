import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib  # For saving the scaler

# Load dataset
df = pd.read_csv('data/train.csv')

# Encode target variable (Attrition)
label_encoder = LabelEncoder()
df['Attrition'] = label_encoder.fit_transform(df['Attrition'])

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler as a pickle file

# Define the XGBoost model
model = xgb.XGBClassifier(eval_metric='logloss', objective='binary:logistic', n_jobs=-1)

# Define the parameter distribution for RandomizedSearchCV (without n_estimators)
param_dist = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 2, 3]
}

# RandomizedSearchCV with 3-fold cross-validation (faster than GridSearchCV)
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=3, verbose=1, n_jobs=-1, random_state=42)

# Fit RandomizedSearchCV (without early stopping)
random_search.fit(X_train, y_train)

# Best parameters found by RandomizedSearchCV
print("\nBest parameters found: ", random_search.best_params_)

# Get the best model from RandomizedSearchCV
best_model = random_search.best_estimator_

# Train the best model with early stopping manually
# Create the DMatrix for early stopping
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set up parameters for XGBoost with early stopping
params = random_search.best_params_
params['eval_metric'] = 'logloss'
params['objective'] = 'binary:logistic'

# Set number of boosting rounds explicitly
num_boost_round = 300

# Train with early stopping
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evallist, early_stopping_rounds=10, verbose_eval=True)

# Save the model in JSON format
bst.save_model('employee_attrition_model.json')
print("\nModel saved.")

# Make predictions using the best model (after early stopping)
y_pred = bst.predict(dtest)
y_pred = np.round(y_pred)  # Convert probabilities to binary output (0 or 1)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy * 100:.2f}%')
