#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Necessary imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report


# In[2]:


# Load dataset (Replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('/kaggle/input/feature-engineering-data/train_sample.csv')

# 1. Data Preparation

# Convert 'click_time' and 'attributed_time' to datetime format
df['click_time'] = pd.to_datetime(df['click_time'])
df['attributed_time'] = pd.to_datetime(df['attributed_time'], errors='coerce')

# Feature Engineering: Extract hour and day of the week from click_time
df['hour'] = df['click_time'].dt.hour
df['day_of_week'] = df['click_time'].dt.dayofweek

# Time since last click per IP (Time difference in seconds between consecutive clicks)
df['time_since_last_click'] = df.groupby('ip')['click_time'].diff().dt.total_seconds()

# Time to attribution (for attributed clicks only)
df['time_to_attribution'] = (df['attributed_time'] - df['click_time']).dt.total_seconds()

# Rolling count of clicks per IP over a 5-click window
df['rolling_click_count'] = df.groupby('ip')['click_time'].rolling(window=5).count().reset_index(drop=True)

# Aggregated Features: Click count per IP
ip_click_count = df.groupby('ip')['click_time'].count().reset_index(name='ip_click_count')
df = pd.merge(df, ip_click_count, on='ip', how='left')

# Attribution rate per channel (conversion rate for each channel)
channel_attribution_rate = df.groupby('channel')['is_attributed'].mean().reset_index(name='channel_attribution_rate')
df = pd.merge(df, channel_attribution_rate, on='channel', how='left')

# Drop columns that won't be useful for prediction (like 'ip', 'click_time', 'attributed_time')
df_model = df.drop(columns=['ip', 'click_time', 'attributed_time'])

# Handle missing values by filling NaNs with 0 (you can also try imputing them)
df_model = df_model.fillna(0)

# Separate features (X) and target (y)
X = df_model.drop(columns=['is_attributed'])
y = df_model['is_attributed']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[3]:


# 2. Feature Selection with RandomForestClassifier

# Initialize RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the RandomForest model on the training data
rf.fit(X_train, y_train)

# Get feature importance from RandomForest
feature_importances = rf.feature_importances_
features = X_train.columns

# Create a DataFrame for better visualization of feature importance
feat_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df)
plt.title('Feature Importance from Random Forest', fontsize=15)
plt.show()



# In[5]:


# 3. Modeling: Logistic Regression

# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)

# Fit the Logistic Regression model
log_reg.fit(X_train, y_train)

# Predict on the test set using Logistic Regression
y_pred = log_reg.predict(X_test)




# In[6]:


# 4. Model Evaluation: Logistic Regression

# Accuracy for Logistic Regression
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy:.4f}')

# Precision and Recall for Logistic Regression
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f'Logistic Regression Precision: {precision:.4f}')
print(f'Logistic Regression Recall: {recall:.4f}')

# Confusion Matrix for Logistic Regression
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Logistic Regression Confusion Matrix', fontsize=15)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report for Logistic Regression
class_report = classification_report(y_test, y_pred)
print("Logistic Regression Classification Report:\n", class_report)



# In[7]:


# 5. Modeling: RandomForestClassifier

# Predict using the RandomForest model
y_pred_rf = rf.predict(X_test)



# In[8]:


# 6. Model Evaluation: RandomForestClassifier

# Accuracy for RandomForest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'RandomForest Accuracy: {accuracy_rf:.4f}')

# Precision and Recall for RandomForest
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
print(f'RandomForest Precision: {precision_rf:.4f}')
print(f'RandomForest Recall: {recall_rf:.4f}')



# In[9]:


# Confusion Matrix for RandomForest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('RandomForest Confusion Matrix', fontsize=15)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



# In[11]:


# Classification Report for RandomForest
class_report_rf = classification_report(y_test, y_pred_rf)
print("RandomForest Classification Report:\n", class_report_rf)


# In[ ]:




