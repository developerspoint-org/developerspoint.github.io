
---
date: 2020-12-27 23:30:00
layout: post
title: Logistic Regression with Scikit-Learn
subtitle: "Blog 3 in Scikit-Learn series"
description: >-
  `Scikit-learn` is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection and evaluation, and many other utilities.
image: >-
  https://blog.developerspoint.org/assets/img/banner_3.jpeg
optimized_image: >-
  https://blog.developerspoint.org/assets/img/banner_3.jpeg
category: Scikit-Learn
tags:
  - python
  - blog
  - scikit-learn
author: Ketan Bansal
---


# Introduction

In my previous Blog, I explained about Linear Regression with Scikit Learn and how it works. Let’s See why Logistic Regression is one of the important topic to understand.  
Here’s the [**link to my previous article on Linear Regression**](https://blog.developerspoint.org/Linear-Regression-with-Scikit-Learn/) in case you missed it.

![alt text](https://blog.developerspoint.org/assets/img/logistic_banner.jpg)

### Logistic Regression in Python with Scikit-Learn

Logistic Regression is a popular statistical model used for binary classification, that is for predictions of the type this or that, yes or no, etc. Logistic regression can, however, be used for multiclass classification, but here we will focus on its simplest application. It is one of the most frequently used machine learning algorithms for binary classifications that translates the input to 0 or 1.

For example: 0 for negative and 1 for positive.

![alt text](https://blog.developerspoint.org/assets/img/logistic_graph.jpeg)

Some applications of classification are:
- Email: spam / not spam
- Online transactions: fraudulent / not fraudulent
- Tumor: malignant / not malignant

Linear regression is not capable of predicting probability. If you use linear regression to model a binary response variable, for example, the resulting model may not restrict the predicted Y values within 0 and 1. Here's where logistic regression comes into play, where you get a probability score that reflects the probability of the occurrence at the event.

![alt text](https://blog.developerspoint.org/assets/img/linear_vs_logistic_regression.jpg)

In the formula of the logistic model,  
when **b0+b1X == 0**, then the p will be 0.5,  
similarly,**b0+b1X > 0**, then the p will be going towards 1 and  
**b0+b1X < 0**, then the p will be going towards 0.

### Logistic Regression on Digits Dataset

**Loading the Data**

The digits dataset is one of datasets scikit-learn comes with that do not require the downloading of any file from some external website. The code below will load the digits dataset.

```
from sklearn.datasets import load_digits  
digits = load_digits()
```

**Showing the Images and the Labels**

This section is just to show what the images and labels look like. It usually helps to visualize your data to see what you are working with.

```
import numpy as np   
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))  

for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):  
  plt.subplot(1, 5, index + 1)  
  plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)  
  plt.title('Training: %i\n' % label, fontsize = 20)
```

![alt text](https://blog.developerspoint.org/assets/img/3_1.png)

**Splitting Data into Training and Test Sets**

We make training and test sets to make sure that after we train our classification algorithm, it is able to generalize well to new data.

```
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
```
## Scikit-learn Modeling

**Import the model you want to use**
  In sklearn, all machine learning models are implemented as Python classes
```
from sklearn.linear_model import LogisticRegression
```
**Make an instance of the Model**
  
```
logisticRegr = LogisticRegression()
```
**Training the model on the data, storing the information learned from the data**
  Model is learning the relation between digits and labels
```
logisticRegr.fit(x_train, y_train)
```
**Predict labels for new data (new images)**
  Uses the information the model learned during the model training process

```
#predict for one image
logisticRegr.predict(x_test[0].reshape(1,-1))

#predict for multiple images
logisticRegr.predict(x_test[0:10])

#for the entire dataset
predictions = logisticRegr.predict(x_test)
```
![alt text](https://blog.developerspoint.org/assets/img/3_2.PNG)

**Model Performance**

`Accuracy = correct predictions / total number of data points`

```
#Use score method to get accuracy of model  
score = logisticRegr.score(x_test, y_test)  
print(score)
```
![alt text](https://blog.developerspoint.org/assets/img/3_3.PNG)

**Confusion Matrix**
A confusion matrix is a table that is used to describe the performance of a classification model on a set of test data for which the true values are known. For making confusion matrices more understandable, I will be sharing the result from our Model.

```
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions)  
print(cm)
```
![alt text](https://blog.developerspoint.org/assets/img/3_4.PNG)

At the end, I hope that you can learn the how to use the simple linear regression techniques. You can also find the full project on the [GitHub repository](https://github.com/ketan-b/Scikit-Learn-Blog).

### Resources

- [scikit-learn documentation](https://scikit-learn.org/)
- [scikit-learn Repository](https://github.com/scikit-learn/scikit-learn)

### Continued in next Week....
This was a simple logistic regression with binary classification in next few weeks we will be discussing how to improve the model using more independent variables, stay tunned..
