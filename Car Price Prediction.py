#!/usr/bin/env python
# coding: utf-8

# ## Now these days cars measures how luxury do you have. But for many people it is  just a service which gives value to their life. In the machine learning this area is quite popular. Predicting the price of car is a major research topic. However price of car depends on many factors. Brand is one of the factor, if you ignore the brand of car the important factors can be
# * Safety
# * Mileage
# * Fuel type
# * Comfort
# * Engine
# * Horsepower  
# and many more 
# 
# ## So I will walk through a model which predict the car price

# # Car Price Prediction Model using Python

# In[7]:


# importing libraries used in this project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# # Importing data set
# ## Data set is from Kaggle (https://www.kaggle.com/goyalshalini93/car-price-prediction-linear-regression-rfe/data?select=CarPrice_Assignment.csv)

# In[8]:


car_data = pd.read_csv('/Users/manishkumar/Downloads/car.csv')


# In[9]:


# to get data information
car_data.head(5)


# In[10]:


# To get how many rows and columns are there
car_data.shape


# ## To proceed further, we need to check for null value whether my data has null values or not. If now you will not check then it will not be a good model to predict price

# In[12]:


car_data.isnull().sum()

#isnull() function will show you null values if any exist. It shows FALSE if no null value otherwise TRUE
#.sum() wil give you aggregate result on the column at once.


# ## As we can see we do not have any null value in our data set

# In[14]:


# To get more information about your data
car_data.info()


# In[16]:


car_data.describe()


# # So price is the our target column. We need to predict the price of cars.

# # Lets give a look to brands

# In[20]:


car_data.CarName


# In[21]:


# Another way to access a single column which I like more sometimes
car_data['CarName']


# # A look at price distribution. After this we make thoght about our model prediction

# In[29]:


sns.set_style('darkgrid')


# In[30]:


plt.figure(figsize=(15, 10))
sns.distplot(car_data.price)
plt.show()


# # Now we will look to the others features and correlation of data set

# In[34]:


car_data.corr()


# # Correlation plot

# In[35]:


plt.figure(figsize=(20, 15))
correlations = car_data.corr()
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()


# # We will use Decision Tree regression algorithm to train our model
# 
# 
# Steps before train our model
# * Split data
# * Train data
# * Test data
# then we train our model using algorithm

# In[36]:


predict = "price"
# features to predict price of car
car_data = car_data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]
x = np.array(car_data.drop([predict], 1))
y = np.array(car_data[predict])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

from sklearn.metrics import mean_absolute_error
model.score(xtest, predictions)


# In[ ]:




