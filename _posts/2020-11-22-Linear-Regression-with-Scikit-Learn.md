---
date: 2020-11-22 23:30:00
layout: post
title: Linear Regression with Scikit-Learn
subtitle: "Blog 2 in Scikit-Learn series"
description: >-
  `Scikit-learn` is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection and evaluation, and many other utilities.
image: >-
  https://blog.developerspoint.org/assets/img/banner_2.jpg
optimized_image: >-
  https://blog.developerspoint.org/assets/img/banner_2.jpg
category: Scikit-Learn
tags:
  - python
  - blog
  - scikit-learn
author: Ketan Bansal
---


# Introduction

In supervised machine learning, there are two algorithms: Regression algorithm and Classification algorithm. For example, predicting house prices is a regression problem, and predicting whether houses can be sold is a classification problem.

![alt text](https://blog.developerspoint.org/assets/img/scikit2_cover.png)

_The term “linearity” in algebra refers to a linear relationship between two or more variables._

In the simple linear regression discussed in this article, if you draw this relationship in a two-dimensional space, you will get a straight line.

Let’s consider a scenario where we want to determine the linear relationship between the house square feet and the house selling price. when we given the square feet of a house, can we estimate how much money can be sold?

We know that the formula of a regression line is basically: **y = mx + b**
where y is the predicted target label, m is the slope of the line, and b is the y intercept.

If we plot the independent variable (Square Feet) on the x-axis and dependent variable (Sale Price) on the y-axis, linear regression gives us a straight line that best fits the data points, as shown in the figure below.

![alt text](https://blog.developerspoint.org/assets/img/demo_graph.png)

### Linear Regression in Python with Scikit-Learn

In this section, we will learn how to use the Python Scikit-Learn library for machine learning to implement regression functions. We will start with a simple linear regression involving two variables. In this regression task we will predict the Sales Price based upon the Square Feet of the house. This is a simple linear regression task as it involves just two variables.


### Code

**Importing Libraries:**

To import necessary libraries for this task, execute the following import statements:

```
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
```

**Importing Dataset**

The dataset is stored on Google Drive, in case the following code block don't work, please download and dataset from the link manually.
We import the data from CSV file using _Pandas_:
```
# Drive link: https://drive.google.com/file/d/1PNN8xQnCv7NA56JTF_igTdem86U_ap9J/view?usp=sharing

!wget https://drive.google.com/uc?id=1PNN8xQnCv7NA56JTF_igTdem86U_ap9J&export=download
!mv uc?id=1PNN8xQnCv7NA56JTF_igTdem86U_ap9J HousingPrices.csv

df = pd.read_csv('HousingPrices.csv')
```
**Understand the data**

Let's explore the data for ourselves:
`df.shape` gives us the shape of the data:

![alt text](https://blog.developerspoint.org/assets/img/run_0.PNG)

The dataset contain 1,460 rows and 2 columns. Let’s take a look at what our dataset actually looks like. enter the `df.head()` which will retrieves the first 5 records from our dataset.

![alt text](https://blog.developerspoint.org/assets/img/run_1.PNG)

To see statistical details of the dataset, we can use `df.describe()`:

![alt text](https://blog.developerspoint.org/assets/img/run_2.PNG)

Finally, we can draw data points on the two-dimensional graph to focus on the dataset and see if we can manually find any relationship between the data. We use `df.plot()` function of the pandas dataframe and pass it the column names for `x` coordinate and `y` coordinate, which are _**SquareFeet**_ and _**SalesPrice**_ respectively. We can use the below script to create the graph:
```
df.plot(x='SquareFeet', y='SalePrice', style='*')
plt.title('Square Feet vs Sale Price')
plt.xlabel('Square Feet')
plt.ylabel('Sale Price')
plt.show()
```
![alt text](https://blog.developerspoint.org/assets/img/run_3.png)

**Preparing the data**
Now we have ideas about the details of data statistics. The next step is to divide the data into _attributes_ and _target labels_. Attributes are independent variables, and target labels are dependent variables whose values are to be predicted. In our dataset we only have two columns. We want to predict the Sales Price based upon the Square Feet of the house. Therefore our attribute set will consist of the _**SquareFeet**_ column, and the label will be the _**SalesPrice**_ column. To extract the attributes and labels, execute the following script:

```
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values
```
The attributes are stored in the X variable. We specified _**-1**_ as the range for columns since we wanted our attribute set to contain all the columns except the last one, which is _**SalesPrice**_. Similarly the y variable contains the labels. We specified 1 for the label column since the index for _**SalesPrice**_ column is 1. Remember, the column indexes start with 0, with 1 being the second column.

Now that we have our attributes and labels, the next step is to split this data into training and test sets. We’ll do this by using Scikit-Learn built-in `train_test_split()` method:
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
The above script splits 80% of the data to training set while 20% of the data to test set. The `test_size` variable is where we actually specify the proportion of test set.

**Modelling**

We have split our data into training and testing sets, and now is finally the time to train our model. Execute following script:
```
def get_cv_scores(model):
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')
    
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')
    
lr = LinearRegression().fit(X_train, y_train)
get_cv_scores(lr)
```
Output:
CV Mean:  0.5147052949885541
STD:  0.06563558755274343

With Scikit-Learn it is extremely straight forward to implement linear regression models, as all you really need to do is import the LinearRegression class, instantiate it, and call the fit() method along with our training data. This is about as simple as it gets when using a machine learning library to train on your data.
In the theory section we said that linear regression model basically finds the best value for the intercept and slope, which results in a line that best fits the data. To see the value of the intercept and slop calculated by the linear regression algorithm for our dataset, execute the following script to retrieve the intercept and slope:

![alt text](https://blog.developerspoint.org/assets/img/run_4.PNG)

_This means that for every one unit of change in Square Feet, the change in the SalesPrice is about 110.26._

**Predictions**

Now that we have trained our algorithm, it’s time to make some predictions. To do so, we will use our test data and see how accurately our algorithm predicts the percentage score. To make pre-dictions on the test data, execute the following script:

```
y_pred = lr.predict(X_test)

plt.scatter(X_train, y_train)
plt.plot(X_test, y_pred, color='red')
plt.show()
```

![alt text](https://blog.developerspoint.org/assets/img/run_5.png)

The y_pred is a numpy array that contains all the predicted values for the input values in the X_test series.

To compare the actual output values for X_test with the predicted values, execute the following script and the output looks like this:

![alt text](https://blog.developerspoint.org/assets/img/run_6.PNG)

Although our model is not very accurate, the predicted value is close to the actual value.

**Evaluation**

The final step is to evaluate the performance of the algorithm. This step is particularly important for comparing the performance of different algorithms on specific data sets. For regression algorithms, three evaluation indicators are usually used:
- Mean Absolute Error (MAE) is the mean of the absolute value of the errors.
- Mean Squared Error (MSE) is the mean of the squared errors.
- Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors

The Scikit-Learn library pre-built the functions that can be used to find out these values for us. Let’s find the values for these metrics using our test data. Execute the following script: 

![alt text](https://blog.developerspoint.org/assets/img/run_7.PNG)

You can see that the value of root mean squared error is 62560.28, which is large than 10% of the mean value of the Sales Price i.e. 180921.2. This means that our algorithm is just average.

There are many factors that contribute to this inaccuracy, some of which are listed here:
- The features we used may not have had a high enough correlation to the values we were trying to predict.
- We assume that this data has a linear relationship, but this is not the case. visual data can help you determine.

At the end, I hope that you can learn the how to use the simple linear regression techniques. You can also find the full project on the [GitHub repository](https://github.com/ketan-b/Scikit-Learn-Blog).

### Resources

- [scikit-learn documentation](https://scikit-learn.org/)
- [scikit-learn Repository](https://github.com/scikit-learn/scikit-learn)

### Continued in next Week....
This was a simple linear regression with dataset that had only two variables in next few weeks we will be discussing how to improve the model using more independent variables, stay tunned..
