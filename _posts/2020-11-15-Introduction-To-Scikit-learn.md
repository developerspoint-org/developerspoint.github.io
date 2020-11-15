---
date: 2020-11-15 15:00:00
layout: post
title: Introduction To Scikit-Learn
subtitle: "Welcome to our 1st Blog in Scikit-learn Series"
description: >-
  `Scikit-learn` is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection and evaluation, and many other utilities.
image: >-
  https://lh3.googleusercontent.com/cIjt8kYv0Adz4gVG5ZBzkeGyBbn4F-w9kLVocf4s1939WekKsLmzdcYm8gojW97eDAG7xtV5RBSfQkJVTSTk5zdzuR9PNHxEFitGnqjVPVg8bO5cz0ptZ9fp0yCRak_hINB56AVPc-F_CZL1v-DF6Rxaxoov2ul3LNeF3m3TtjEvSZOGD6_ZHUUCT1DyHe2lRnjebsT_frVMa6egpiLigr0vpXiGoH6EQKBlD3ZpRE3pZu9c71_Zp4HYxLfZJYs_VwluX4yOcvzMzcONQmUyM2qhJD4bTbguwWAtfdXjpx1HMTr380J1c3eS_SsHnTbKJ-Z5Cd5ceHgCtEm2ALua_G_QhBCIVYiMMvAmnb7fRooeZJwrweKWgzBfjPncNaGixFwWYTCxt77M6VULLoZHHmGXoMOGBoCULzguSDNZ-2E0aI5c0N0MGxjxa-DhM77T3rZ7xXyizo22whj6xy9RO-QrUeGvG6SIIrfGsZyViYRAjaInnS31t3P6czKlgsdBC1qO9ZbmHQdMLdKgvpz2Jbm6MEBezouqsyVN6FwRkmdEp62yuV5v7GD77k1795W4uQCqggAulEMJnLzrF5qcx6iCrtRVoxC2bVrk3BM7Nv-SC_Tpfz3P9nlKUS72APPBTHxuK13dbhKC1jjSh_pvYckhzXOoTWwCw60MMOaeH5SS5seojGdiZ9OKLYbCzA=w1292-h969-no?authuser=0
optimized_image: >-
  https://lh3.googleusercontent.com/cIjt8kYv0Adz4gVG5ZBzkeGyBbn4F-w9kLVocf4s1939WekKsLmzdcYm8gojW97eDAG7xtV5RBSfQkJVTSTk5zdzuR9PNHxEFitGnqjVPVg8bO5cz0ptZ9fp0yCRak_hINB56AVPc-F_CZL1v-DF6Rxaxoov2ul3LNeF3m3TtjEvSZOGD6_ZHUUCT1DyHe2lRnjebsT_frVMa6egpiLigr0vpXiGoH6EQKBlD3ZpRE3pZu9c71_Zp4HYxLfZJYs_VwluX4yOcvzMzcONQmUyM2qhJD4bTbguwWAtfdXjpx1HMTr380J1c3eS_SsHnTbKJ-Z5Cd5ceHgCtEm2ALua_G_QhBCIVYiMMvAmnb7fRooeZJwrweKWgzBfjPncNaGixFwWYTCxt77M6VULLoZHHmGXoMOGBoCULzguSDNZ-2E0aI5c0N0MGxjxa-DhM77T3rZ7xXyizo22whj6xy9RO-QrUeGvG6SIIrfGsZyViYRAjaInnS31t3P6czKlgsdBC1qO9ZbmHQdMLdKgvpz2Jbm6MEBezouqsyVN6FwRkmdEp62yuV5v7GD77k1795W4uQCqggAulEMJnLzrF5qcx6iCrtRVoxC2bVrk3BM7Nv-SC_Tpfz3P9nlKUS72APPBTHxuK13dbhKC1jjSh_pvYckhzXOoTWwCw60MMOaeH5SS5seojGdiZ9OKLYbCzA=w1292-h969-no?authuser=0
category: blog
tags:
  - welcome
  - python
  - blog
  - scikit-learn
author: Ketan Bansal
---

# Introduction

Scikit learn is a machine learning library for python programming language which offers various important features for machine learning such as classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to inter-operate with the python numerical and scientific libraries like **Numpy** and **SciPy**.

![alt text](https://lh3.googleusercontent.com/IM1-_HG_kGXwsfyuyESEMMuj49CGsdXS8KmdbzCOie4eoo_jE7AeT0u_p9gOEskIHP_qf8eA3YcOoGzlX_nZsMgCUHsK-dUyPMkWKht-71ClMvemKIa6mELWPc24RV-fyHfCXlDjaqljugVVELiVCtZ_LVtA4XyN8xyA2lZR_0YUo6OZUd2K8R05ADudtn7mdYIWMMy_POBgnhyrbl9Szw1DJNMqK7HG_dc-Jo9gAtzJ_dZxgg_6xindy40ucsJjJtvhtYjVcm2t3xs_Sn-wL5w9mJ759llSY6-WMbvNRNvMNq19srase9kaIha_HtfYFRXEH0WNsfQ6DN1V3AAt9T45UUrpRoonHBXoEZTwTW3eSuSSO8qY9Ku0xb4OxbbzEiHOVzslK3VOwkw1fZUQxCSGZm_RcHQtEto012tENgnOVrRGDf0xwEUybX0Z84xL2BzCULIapsBFdB0qnmfFmrrpDy9sm8FpmGAGSRKVPVmB4OwvcKSi9vvBeg81uc7B59pm1bQV-WbvYfCw8rBaMCj6Qj1Jnp2WtHT4IU5tCxv7PLVPdBvZ_TIxSN2TE0lvkZnuFsY-f1yQQwIIHJo3huhq3jCIp2Fn2Jkh0WzdkfLnYrBJOWKJGwToW-f2Hb3lnakX63_MqRZLLuuSVY8IuXeD6MEMpB6MHZ8h7S1ntsLtu4WCHq-BfSclLDbjsg=w615-h215-no?authuser=0)

_We will discuss each algorithm and its implementation with codes in detail later in the second part of this series._

### Supervised Algorithms In Scikit-Learn

Since you are already familiar with machine learning you already know that there are two types of algorithms i.e supervised and unsupervised algorithms.
So, we will see what scikit-learn has to offer in supervised algorithms.

_The problem of supervised learning can be broken into two_ :

![alt text](https://lh3.googleusercontent.com/DxdPPzBM69RJvSGtKn7HulpYX-2IhDctSprGrOSxEaLe2oreh1GFhJ90rp4cLVR0gWMN-G4e9BMCAMWJi1knHwkk_n_OTjSxMPIFbIgM1hLQUYM92VrK94RWIVyqNZn1fvVjnYKGbuZvF7qfOw6n9kpvquoSFKvZ5ZxA3Gh6urxmsoFhXftQT1ddWl9SQbjGcPTbODxebc7v2d-AN2tajnBdtWRJZCfvpBk4HUwMqHJW-mCcLHGUItRDqEpIIkl4yW2EJ4wppECTVzKSyExOMdBhEiVdA9AlT_rThggzgd3PH49-2cdwhbpf8KDTegPQ4qEV5VGSfwg6uCJ9nMJ6WFGfVzYEuRrCEg39f5uwXsL7JPFRFEWwv371duoddm7gHNqdzLnGmZ5GRYCa_kkug6sxB6Ln-I1pQQG1wsaTSsiOdCpNX5G2SSDNqdqR5YpMUdgekhMpIczsVAW7azxK5BQ57yZxB2fgozsLJ2aAylTLoocs8VVo4IgMkMZP0eR34RPphw3BRqhe_epvOtZffh61GY0rIvIIoKjYQqbd2L8LONQZ4wvrhGsqQ-7djuZVI76214FqeX-icdyvV2qJRVHfCly3oEhSUwpi8ONaRES_h7owLyeGbdpiBq13AGiQYZmegDnRc3b17o6jqyZVZXuYx81JatVnZ-m1Y0P6vYgKM3sdp8QrqdXtL2KXPg=w608-h308-no?authuser=0)

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

![alt text](https://lh3.googleusercontent.com/-z9MAk_BGM7wfcsy0MuB7CPAWN1gnnLOuL-1oF4X_256lXq6706TW6UM3cnHoDKvN144RJb5qAq9jfdGqdHWs137k1lYfGSTiF03Bgiz3Rp8T3Mf3ZSwm0_M6imt4w_Ni1kVcYVbX3_sYOe6qI9lI_CBWic0OpdZVCyw2YQwYSEEeZX_qgB54bqt0SnEX09upnYsXDgzwDRa8McO0dJtZU-7RBmMtZiQ4VrPz5lb4tbEo-xYY99J2kfCs61Y0nLcRCrBqFNG8_kHaIRo1E9D_A8KJs6-cyuHiByVAUoNbZZmVZc33pKCR3741bE6qWWHtiubxnz66qqF7SBVNRzt9kBRM1tb3HtNibs7jJ5h_B9smafF7gi9vT_VdjOb6YkklYL4QPupGNgjjOgV7mnFbEoLScgLsyMruUo3aDqSbvjtKXLWx9JAV4SKpi4AXAr16UzCWvvHZ_1LC5TD9awD3vVQCo0mt-oU1_dzanS2ESS_8YnOiSzUlZNX3fYs2zeuwo9BoLE8DYw-w_1TTrFrtkMuew6RXkaKAsysr7Zb-gCGyS1vT4bcYVOWVBz56jSbxe-Dd4VCJfIWY8KRDA4mJmETZM2QSE8wlc3R5INwsi8ETMp7cuX4O1kEpl3F7jZCFlJ_C0qgzHKfrWwg_4i-L3kxtSOSJp6fbwA2DXbD_kVbocUP2tFnJQTjdknwsw=w351-h221-no?authuser=0)

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
![alt text](https://lh3.googleusercontent.com/6bykQ2vzDyad38TpRnB7pda1ITSwc0D7tgqz0ab8-jaA971yCqffR8yLSrZjOYmWWLalKRiEjPB_LV-6ykjUBzCbvUSv_RBbv2iKDaNbF-2V3e5w6IU5CqHIM01aYLnsuUaySy3KGX18BL5SR3CkB_ynPdBSvAo0A0BtuqbgAhKSsh5en6dml4UPo5fVQ4Cx2BPTTIPOSyFTPMoqugByFds-0xjyflI4LdWkN1htBsZzm7TkO30F5p_kBW09XQeR32wCzCunYdFrUPf_51o_dHyv3kPF-CjF2oj5r9rZtYZxiiO_6ctk85zCpqdoDQ8R97cdIDtH1BIhTfccmt29VXVcnAOEUjetkCPXN_OjejrcluNLj0mYYpUEbmKZmiFuWu_mFZrr1PpdNQaP0WLnvRphtedwnZxJ4D7_eFV5jMilr6ndq7S6sTgupd-CxQ5u07pKUwCJSFF4HrW7jLPQewiKJXBnQ5f7Lo105Bp0o-svwA-Oj7K3GsUAIyC7tJJlqbt9-zPhZKFoLRaleuuTjIn5VQ7c9WMpkGIE5YrBC65iZ9ArWrI12rvJt-kbDLjDeaOG5pPzsJ3nYq8woVvG-6TsowSTZQmMIlVIC4bDDx9hNQ1GmZVA4ArADREz3HIIc82i-O8kTDHm-LgZKuTBOiWOc8HeNkyUAPff3gGr5HeEGzPGHkUDz_YaCUPe2w=w340-h226-no?authuser=0)
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

![alt text](https://lh3.googleusercontent.com/Wy4g2pdRGr4lR2dq65TgSoyurKOFdfzuiStCg6M9t-4dikRdP9s1owllUYRqfyq3Nu3ZYoYnKK6-7VdSzWnZgTmmdANGYqteQU-6UeI16Q41Zx3FEKnVXb90c5V1INJcBA-iSSAWmrqmoHJXgBz-3V4cK-ruCf7p4VX5o1TdPSNWsOCxjHgn6V4W6oXAf6XrI0KIC0CMUkcuJz4wyTlzR_Z3hjYNz-VgAjLp_SES4FM9YAQhJeUJlUVZGBU9xagAxNkSx2jgkjBCRA_8t4T5OetS9xu2wodMJ9Ef1E2Rx1NtUOjc8i7TW8BTfmtU5n--x8ukce2WKpTifDESjYfX-NLreiH5N1sGKxrS2CW9D3_c7Bzp2B0BQJagOCYJGAFFbF_uZzAl4PEKvNuD2a6WdYpno5Cqrh87xe7GQgirzC9hTrc0P4v3VwV6wF8oUlwz7P6dvKWg32twvTjRU2LM7DFqYwhXGbtPMV5FQCwE3AFhGjYMPC6a-53wqif-0uWK6AXKCVU2IsMdp9VPdbI43IusEetfHaZrGMN_Ybx9OWqPB9eS7EWr2_JRIKzSI_FwSqVw2rzxLdfs0D4yn-PsIRxuv_uX9DebTafsl3Ec3pGgxMuuda-jlka52l4eWbcOLdZILOxXHjaZO2-KPRnsPNT771v211-qSB8LS9B8LD4-SM1zSFqCEb4eUpx2GQ=w561-h283-no?authuser=0)

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
![alt text](https://lh3.googleusercontent.com/SSgabN_C4T2YZPUywEqyBiOAEifes9hUUDWGfhEKkEX4HRXPdEVNxpN5CXJ594g6aCwKV8vAumtvLOk04SLh-0nP3q27zVHhv9LnE-5iGqePcxWl6swCgZdG3uL_F799gibFwQj_plPl5k5djgEYoPUT3h_3oSaf72E00aTOa9dFtLHbkRfU2U7QiUb0BIHX8bOOCVmcAh7HHruRrBQ4h8JEFqYeICT6Re-9LZ6LE4vsqq5ShlaxXxSaW9U4HHYyOPtOhZKOOkUW9EEmAcaN9syrnOVh2fAOhcV7eSwtxI6jEqrtCCNSTQI5x-fNm-HHu5JYr7QH-jh8oQF2i1DhHq9o5dM4acs4NjTD2u2l8fvLleDxp30N1Rs_uI9xLg0Ajxr1N6XBSKCeK1Teek1q4f7dTgmZW5cEZ3Op__UjZW85eJp0MWSgtteLpGli_D6gjcLBUP0ZXknDfpzfd_xQlRbLFnK_qQK6zh_FBPgXWPe8svkvNRFT3WfizT8g9okzt1WdPA3ZWCcHPYao_wh3dHLiqX3E9PWLxlFFCbUWWqrHr4HDCHk9gMLbyl7EpDHuPcu-v5TSgAJaYZZBKPkTu38wmip0lY5933I-o3xIg3b6l5-Dvydra6mwsui2PLL_uvL31ZEKYLKbSWLF_WBX7MI6dULNrBn10LXVuVrjMB5vgNzTMzWGyJKT7vxhjg=w379-h266-no?authuser=0)
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

![alt text](https://lh3.googleusercontent.com/tcA5I62ye_bXJY_bXpWq2cZBxayQRwwXCJNM6Ezb-d4tEj-rczfLbnvI4pnMv4ZPpnBdQCr-QW048Am19f4FndrSkt-qPno02HhmLN5fRj0I6WHcdWwmzz8lmvae98QC65uP3YOXB6Zi65A9C_KU_YS0MQkoKDqecU1HEIDGB3gEBmxXQ96Cg54jifbRbGPaRy40selcZ8WTnM9wNy2tGAvqj3fi1mm7MnrImyIJVyY2wscl_Cp7xDPFG-YrNdKuaH1nVnBUQT8etXYputJvSCGrxvDX6xXxqbPdZ22YdWown8ZoOHk7KB_JRe0bjKHRcqi8Ec6Cw8zkIC-gAZkGS7O18Z0eaxv24h8xwS1TWG8cx30QZSqtO_hLd-HogYSNSQTU2mxSo4tCf_i7eFoeWVrNnZZFjie4quFbZ9WMvCp5j_KY9HbRgSrC18eveo-HK_gew2RSE8qGrijfgCU0ZA-cHDntbu6QUDCSU33e3lvk9g-vwl7VkuWtfVFn7MJjx4RE39wJ_jB1Xore8cXiCc__rSGfIHKvHk1EQWxGt_cN8R6Uq8hLjNjazcMcyz_bhoupU9Q6YpWmAvfdpGeUWWiJ2qRG4R1ByXkrHJeddhd-LAado9ey4eCYhoDwj1O6tH26MdcNW9vDHxC5xjcSrVd7KW-aojxRMvOF8NYjnfoR4ueJmg-prU8-TqGkKA=w1400-h500-no?authuser=0)

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
