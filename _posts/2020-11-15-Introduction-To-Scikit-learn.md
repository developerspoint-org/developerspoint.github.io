---
date: 2020-11-15 15:00:00
layout: post
title: Introduction To Scikit-Learn
subtitle: "Welcome to our 1st Blog in Scikit-learn Series"
description: >-
  `Scikit-learn` is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection and evaluation, and many other utilities.
image: >-
  https://blog.developerspoint.org/assets/img/banner.png
optimized_image: >-
  https://blog.developerspoint.org/assets/img/banner.png
category: Scikit-Learn
tags:
  - python
  - blog
  - scikit-learn
author: Ketan Bansal
---


# Introduction

Scikit learn is a machine learning library for python programming language which offers various important features for machine learning such as classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to inter-operate with the python numerical and scientific libraries like **Numpy** and **SciPy**.

![alt text](https://blog.developerspoint.org/assets/img/scikitlearn.png)

_We will discuss each algorithm and its implementation with codes in detail later in the second part of this series._

### Supervised Algorithms In Scikit-Learn

Since you are already familiar with machine learning you already know that there are two types of algorithms i.e supervised and unsupervised algorithms.
So, we will see what scikit-learn has to offer in supervised algorithms.

_The problem of supervised learning can be broken into two_ :

![alt text](https://blog.developerspoint.org/assets/img/diff.png)

**Classification**: Samples belong to two or more classes, and we want to learn from already labeled data on how to predict the class of unlabeled data. An example would be the handwritten digit recognition example, in which the aim is to assign each input vector to one of a finite number of discrete categories. Another way to think of classification is as a discrete (as opposed to continuous) form of supervised learning where one has a limited number of categories and for each of the n samples provided, one is to try to label them with the correct category or class.

Checkout the implementation of the above example: 
```
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
_, axes = plt.subplots(2, 4)
images_and_labels = list(zip(digits.images, digits.target))
for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train)

# Now predict the value of the digit on the second half:
predicted = classifier.predict(X_test)

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

plt.show()
```

**Output:**
    accuracy                           0.97       899
   macro avg       0.97      0.97      0.97       899
weighted avg       0.97      0.97      0.97       899

![alt text](https://blog.developerspoint.org/assets/img/handwritten.png)

**Regression**: If the desired output consists of one or more continuous variables, then the task is called regression. An example of a regression problem would be the prediction of the diabetes in which input consist of the age, sex, body mass index, average blood pressure, etc.

Checkout the implementation of the above example: 
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
```
_**Output:**_
Coefficients: 
 [938.23786125]
Mean squared error: 2548.07
Coefficient of determination: 0.47
![alt text](https://blog.developerspoint.org/assets/img/diabetes.png)
_**It's okay if you don't understand the code, we will be discussing it in detail later.**_

#### Scikit Learn supports following models :
- Generalized Linear Models
- Kernel ridge regression
- Support Vector Machines
- Stochastic Gradient Descent
- Linear and Quadratic Discriminant Analysis
- Naive Bayes
- Decision Trees
- Ensemble methods
- Multiclass and multilabel algorithms
- Feature selection
- Nearest Neighbors
- Gaussian Processes
- Cross decomposition
- Semi-Supervised
- Isotonic regression
- Probability calibration
- Neural network models (supervised)

### Unsupervised Algorithms In Scikit-Learn

Now, let us see what scikit learn offers us in unsupervised algorithms.

![alt text](https://blog.developerspoint.org/assets/img/cluster1.png)

In this the training data consists of a set of input vectors without any corresponding target values/labels. The goal in such problems is to discover groups of similar data within the data, where it is called **clustering**.

Checkout the implementation of K-means clustering on Handwritten digits example: 
```
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

X_digits, y_digits = load_digits(return_X_y=True)
data = scale(X_digits)

n_samples, n_features = data.shape
n_digits = len(np.unique(y_digits))
labels = y_digits

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')

# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
```
_**Output:**_
n_digits: 10,    n_samples 1797,   n_features 64
__________________________________________________________________________________
init    time  inertia homo  compl v-meas  ARI AMI silhouette
k-means++ 0.40s 69510 0.610 0.657 0.633 0.481 0.629 0.129
random    0.30s 69907 0.633 0.674 0.653 0.518 0.649 0.131
PCA-based 0.05s 70768 0.668 0.695 0.681 0.558 0.678 0.142
__________________________________________________________________________________
![alt text](https://blog.developerspoint.org/assets/img/cluster.png)
_**It's okay if you don't understand the code, we will be discussing it in detail later.**_

##### Scikit Learn supports these models :

- Gaussian mixture models
- Decomposing signals in components (matrix factorization problems)
- Covariance estimation
- Novelty and Outlier Detection
- Manifold learning
- Clustering
- Biclustering
- Density Estimation
- Neural network models (unsupervised).



### Model Selection and Evaluation

As we know learning the parameters of a prediction function and testing it on the same data is a methodological mistake or it can be called as cheating : a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called **overfitting**. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test.

![alt text](https://blog.developerspoint.org/assets/img/model.png)

_The above image shows three cases underfitting, ideal and overfitting scenarios respectively._

#### Model selection contains the following :
- Cross-validation: evaluating estimator performance
- Tuning the hyper-parameters of an estimator
- Model evaluation: quantifying the quality of predictions
- Model persistence
- Validation curves: plotting scores to evaluate models

### Dataset transformations

These are represented by classes with a fit method, which learns model parameters (e.g. mean and standard deviation for normalization) from a training set, and a transform method which applies this transformation model to unseen data. fit_transform may be a more convenient and efficient for modelling and transforming the training data simultaneously.

It has following sub-categories :
- Pipeline and FeatureUnion: combining estimators
- Feature extraction
- Preprocessing data
- Unsupervised dimensionality reduction
- Random Projection
- Kernel Approximation
- Pairwise metrics, Affinities and Kernels
- Transforming the prediction target (y)

### Dataset Loading Utilities

The `sklearn.datasets` package embeds some small yet useful datasets.

- The Olivetti faces dataset
- The 20 newsgroups text dataset
- Downloading datasets from the mldata.org repository
- The Labeled Faces in the Wild face recognition dataset
- Forest covertypes
- RCV1 dataset
- Boston House Prices dataset
- Breast Cancer Wisconsin (Diagnostic) Database
- Diabetes dataset
- Optical Recognition of Handwritten Digits Data Set
- Iris Plants Database
_and many more.._

### Resources

- [scikit-learn documentation](https://scikit-learn.org/)
- [scikit-learn Repository](https://github.com/scikit-learn/scikit-learn)

### Continued in next Week....
This was a brief intro of what we will be discussing in next few weeks, stay tunned..
