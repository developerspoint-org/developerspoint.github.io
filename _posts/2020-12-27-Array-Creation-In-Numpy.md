---
date: 2020-12-27 23:59:59
layout: post
title: Array Creation and Basic Operations on Arrays
subtitle: "Welcome to our 4th Blog in NumPy Series"
description: >-
  One of the most important objects is an N-dimensional array type known as ndarray.
image: >-
  /assets/img/numpy-blog4.jpg
optimized_image: >-
  /assets/img/numpy-blog4.jpg
category: numpy
tags:
  - python
  - blog
  - NumPy
author: Sai Charan.M
paginate: true
---

# NumPy Blog - 4

Hola Readers 👨‍🎓👩‍🎓...
Last week we saw what are NumPy arrays and different Data Types in NumPy in Detail [Here is the Link for the previous week's Blog in case if you have missed it](http://blog.developerspoint.org/Introduction-to-NumPy-Arrays/). In this Blog we are going to perceive about......

1.  Most Important NumPy Data Types
2.  Array Creation Using Various Methods

## Most Important NumPy Data Types

### One Dimensional Array

One of the most important objects is an N-dimensional array type known as ndarray.

_We can think of a one-dimensional array as a column or a row of a table with one or more elements:_

![alt text](/assets/img/1darray.png)_Example 1D Array_

All of the items that are stored in ndarray are required to be of the same type. This implies that the ndarray is a block of homogeneous data. ndarray has striding information. This numerical value is the number of bytes of the next element in a dimension.

This helps the array to navigate through memory and does not require copying the data.

Each ndarray contains a pointer that points to its memory location in the computer. It also contains its dtype, its shape, and tuples of strides. The strides are integers indicating the number of bytes it has to move to reach the next element in a dimension.

To create an array:

     import numpy as np    a = np.array([1,2,3])

### Multi-Dimensional Array

A multidimensional array has more than one column.

_We can consider a multi-dimensional array to be an Excel Spreadsheet — it has columns and rows. Each column can be considered as a dimension._

![alt text](/assets/img/2darray.png)_Example 2D Array_

We can instantiate an array object:

    numpy.array([,.,.,.,])
    e.g.
    numpy.array([1,2]) #1D
    numpy.array([[1,2],[10,20]]) #2D#For complex types
    numpy.array([1,2], dtype=complex) #1D complex

**If you want to create a 3-D Array:**

- This will create 3 arrays with 4 rows and 5 columns each with random integers.

       `3DArray = np.random.randint(10, size=(3, 4, 5))`

## Array Creation Using Various Methods

- First and Foremost thing to do is..

```
      import numpy as np
      import numpy
```

There are a number of different ways to create an array. This section will provide an overview of the most common methodologies:

### Array Creation Routines

**1 .If you want to create an array without any element:**

    numpy.empty(2) #this will create 1D array of 2 elements
    numpy.empty([2,3]) #this will create 2D array (2 rows, 3 columns each)

**2. If you want to create an array with 0s:**

    numpy.zeros(2) #it will create an 1D array with 2 elements, both 0
    #Note the parameter of the method is shape, it could be int or a tuple

**3. If you want to create an array with 1s:**

    numpy.ones(2) # this will create 1D array with 2 elements, both 1

**4. Random number generation**

Use the random module of numpy for uniformly distributed numbers:

    np.random.rand(3,2) #3 rows, 2 cols

### Array Creation From Existing Data

**1. If you want to create a Numpy array from a sequence of elements, such as from a list:**

    numpy.asarray([python sequence]) #e.g. numpy.asarray([1,2])

**2. From a buffer in memory:**

We can make a copy of the string in memory:

    x = np.fromstring(‘hi’, dtype=’int8')

Then we can refer to the buffer of the string directly which is memory efficient:

    a = np.frombuffer(x, dtype=’int8')

We can pass in dtype parameter, default is float.

### Array Creation From Numerical Ranges

**1. If you want to create a range of elements:**

    import numpy as np
    array = np.arange(3)
    #array will contain 0,1,2

**2. If you want to create an array with values that are evenly spaced:**

    numpy.arange(first, last, step, type)e.g. to create 0-5, 2 numbers apart
    numpy.arange(0,6,2) will return [0,2,4]

**3. If you want to create an array where the values are linearly spaced between an interval then use:**

    numpy.linspace(first, last, number)
    e.g.
    numpy.linspace(0,10,5) will return [0,2.5,5,7.5,10]

**4. If you want to create an array where the values are log spaced between an interval then use:**

    numpy.logspace(first, end, number)
    e.g.
    a= numpy.logspace(1, 15, 4)#results in [1.00000000e+01 4.64158883e+05 2.15443469e+10 1.00000000e+15]

Any base can be specified, Base10 is the default.

### Resources

- [NumPy.org](https://numpy.org/doc/stable/user/basics.creation.html)

- [Array Creation Routines](https://numpy.org/doc/stable/reference/routines.array-creation.html)

### Continued in Next Week...
