#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[2]:


# Load dataset
df = pd.read_csv('Restaurant_revenue.csv')


# In[3]:


# Display the first few rows of the dataset
print(df.head())


# In[4]:


# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le


# In[5]:


# Preprocess the dataset
# Handle missing values if any
df.fillna(method='ffill', inplace=True)

X = df.drop('Monthly_Revenue', axis=1)
y = df['Monthly_Revenue']


# In[6]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[8]:


# Train and evaluate Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)


# In[9]:


# Calculate evaluation metrics for Decision Tree
mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
print(f"Decision Tree MAE: {mae_dt}")
print(f"Decision Tree RMSE: {rmse_dt}")


# In[10]:


# Train and evaluate Nearest Neighbor model
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)


# In[11]:


# Calculate evaluation metrics for Nearest Neighbor
mae_knn = mean_absolute_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
print(f"Nearest Neighbor MAE: {mae_knn}")
print(f"Nearest Neighbor RMSE: {rmse_knn}")


# In[12]:


# Comparative analysis
print(f"Comparative Analysis:")
print(f"Decision Tree - MAE: {mae_dt}, RMSE: {rmse_dt}")
print(f"Nearest Neighbor - MAE: {mae_knn}, RMSE: {rmse_knn}")


# In[ ]:




