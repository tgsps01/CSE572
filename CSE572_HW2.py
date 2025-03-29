import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training data
train_data = pd.read_csv('train.csv')

# Create a copy of the data for preprocessing
X = train_data.copy()

# Drop unnecessary columns
X = X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values
# For Age, we'll use median imputation
age_imputer = SimpleImputer(strategy='median')
X['Age'] = age_imputer.fit_transform(X[['Age']])

# For Embarked, we'll use mode imputation
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])

# Feature Engineering
# Create family size feature
X['FamilySize'] = X['SibSp'] + X['Parch'] + 1

# Create is_alone feature
X['IsAlone'] = (X['FamilySize'] == 1).astype(int)

# Create title feature from Name
X['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Group rare titles
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
X['Title'] = X['Title'].replace(rare_titles, 'Rare')

# Encode categorical variables
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])
X['Embarked'] = le.fit_transform(X['Embarked'])
X['Title'] = le.fit_transform(X['Title'])

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'Fare', 'FamilySize']
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Prepare target variable
y = X['Survived']
X = X.drop('Survived', axis=1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Preprocessing completed!")
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print("\nFeature names:")
print(X_train.columns.tolist())

# Define the parameter grid for GridSearchCV
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}

# Create a decision tree classifier
dt = DecisionTreeClassifier(random_state=42)

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit the GridSearchCV
print("\nPerforming Grid Search...")
grid_search.fit(X_train, y_train)

# Print the best parameters
print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Get the best model
best_dt = grid_search.best_estimator_

# Make predictions on validation set
y_pred = best_dt.predict(X_val)

# Print model performance metrics
print("\nModel Performance on Validation Set:")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Create confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('dt_confusion_matrix.png')
plt.close()

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(best_dt, 
          feature_names=X_train.columns,
          class_names=['Not Survived', 'Survived'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

# Print feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_dt.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('dt_feature_importance.png')
plt.close()

# Perform 5-fold cross-validation
print("\nPerforming 5-fold Cross-Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_dt, X, y, cv=kf, scoring='accuracy')

# Print cross-validation results
print("\nCross-Validation Results:")
print(f"Individual fold accuracies: {cv_scores}")
print(f"Average accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Visualize cross-validation scores
plt.figure(figsize=(8, 6))
plt.plot(range(1, 6), cv_scores, 'bo-', label='Fold Accuracy')
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label='Mean Accuracy')
plt.fill_between(range(1, 6), 
                cv_scores.mean() - cv_scores.std(),
                cv_scores.mean() + cv_scores.std(),
                alpha=0.2, color='r')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('5-Fold Cross-Validation Results')
plt.legend()
plt.grid(True)
plt.savefig('dt_cross_validation_scores.png')
plt.close()

print("\n" + "="*50)
print("RANDOM FOREST MODEL")
print("="*50)

# Define the parameter grid for Random Forest GridSearchCV
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 6, 7, 8, 9, 10, None],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', None]
}

# Create a Random Forest classifier
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Create GridSearchCV object for Random Forest
rf_grid_search = GridSearchCV(
    estimator=rf,
    param_grid=rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit the GridSearchCV for Random Forest
print("\nPerforming Grid Search for Random Forest...")
rf_grid_search.fit(X_train, y_train)

# Print the best parameters for Random Forest
print("\nBest parameters for Random Forest:", rf_grid_search.best_params_)
print("Best cross-validation score:", rf_grid_search.best_score_)

# Get the best Random Forest model
best_rf = rf_grid_search.best_estimator_

# Make predictions on validation set
rf_y_pred = best_rf.predict(X_val)

# Print Random Forest model performance metrics
print("\nRandom Forest Model Performance on Validation Set:")
print("Accuracy:", accuracy_score(y_val, rf_y_pred))
print("\nClassification Report:")
print(classification_report(y_val, rf_y_pred))

# Create confusion matrix for Random Forest
rf_cm = confusion_matrix(y_val, rf_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('rf_confusion_matrix.png')
plt.close()

# Print Random Forest feature importance
rf_feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_rf.feature_importances_
})
rf_feature_importance = rf_feature_importance.sort_values('importance', ascending=False)
print("\nRandom Forest Feature Importance:")
print(rf_feature_importance)

# Plot Random Forest feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=rf_feature_importance)
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()

# Perform 5-fold cross-validation for Random Forest
print("\nPerforming 5-fold Cross-Validation for Random Forest...")
rf_cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')

# Print Random Forest cross-validation results
print("\nRandom Forest Cross-Validation Results:")
print(f"Individual fold accuracies: {rf_cv_scores}")
print(f"Average accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")

# Visualize Random Forest cross-validation scores
plt.figure(figsize=(8, 6))
plt.plot(range(1, 6), rf_cv_scores, 'bo-', label='Fold Accuracy')
plt.axhline(y=rf_cv_scores.mean(), color='r', linestyle='--', label='Mean Accuracy')
plt.fill_between(range(1, 6), 
                rf_cv_scores.mean() - rf_cv_scores.std(),
                rf_cv_scores.mean() + rf_cv_scores.std(),
                alpha=0.2, color='r')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('Random Forest 5-Fold Cross-Validation Results')
plt.legend()
plt.grid(True)
plt.savefig('rf_cross_validation_scores.png')
plt.close()

# Compare Decision Tree and Random Forest performance
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print("\nDecision Tree vs Random Forest Performance:")
print(f"Decision Tree Average CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Random Forest Average CV Accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")
