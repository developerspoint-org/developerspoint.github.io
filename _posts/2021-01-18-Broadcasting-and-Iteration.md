---
date: 2021-01-18 23:59:59
layout: post
title: BroadCasting and Itearation
subtitle: "Welcome to our 6th Blog in NumPy Series"
description: >-
  The term broadcasting refers to the ability of NumPy to treat arrays of different shapes during arithmetic operations. Arithmetic operations on arrays are usually done on corresponding elements.
image: >-
  /assets/img/numpy-blog-6.jpg
optimized_image: >-
  /assets/img/numpy-blog-6.jpg
category: numpy
tags:
  - python
  - blog
  - NumPy
author: Sai Charan.M
paginate: true
---

# NumPy Blog - 6

Namaste Readers...🙏🙏
Last week learnt Indexing and Slicing of NumPy Arrays [Here is the Link for the previous week's Blog in case if you have missed it](https://blog.developerspoint.org/Indexing-And-Slicing-of-NumPy-Arrays/). In this Blog we are going to perceive about......

1. Broadcasting
2. Iterating over ndarrays

## Broadcasting

The term **broadcasting** refers to the ability of NumPy to treat arrays of different shapes during arithmetic operations. Arithmetic operations on arrays are usually done on corresponding elements. If two arrays are of exactly the same shape, then these operations are smoothly performed.

    import numpy as np

    a = np.array([1,2,3,4])
    b = np.array([10,20,30,40])
    c = a * b print c

Its output is as follows −

    [10   40   90   160]

If the dimensions of two arrays are dissimilar, element-to-element operations are not possible. However, operations on arrays of non-similar shapes is still possible in NumPy, because of the broadcasting capability. The smaller array is **broadcast** to the size of the larger array so that they have compatible shapes.

Broadcasting is possible if the following rules are satisfied −

- Array with smaller **ndim** than the other is prepended with '1' in its shape.
- Size in each dimension of the output shape is maximum of the input sizes in that dimension.
- An input can be used in calculation, if its size in a particular dimension matches the output size or its value is exactly 1.
- If an input has a dimension size of 1, the first data entry in that dimension is used for all calculations along that dimension.

A set of arrays is said to be **broadcastable** if the above rules produce a valid result and one of the following is true −

- Arrays have exactly the same shape.
- Arrays have the same number of dimensions and the length of each dimension is either a common length or 1.
- Array having too few dimensions can have its shape prepended with a dimension of length 1, so that the above stated property is true.

The following program shows an example of broadcasting.

    import numpy as np
    a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]])
    b = np.array([1.0,2.0,3.0])
    print  'First array:'
    print a
    print  '\n'
    print  'Second array:'
    print b
    print  '\n'
    print  'First Array + Second Array'
     print a + b

The output of this program would be as follows −

    First array:
    [[ 0. 0. 0.]
     [ 10. 10. 10.]
     [ 20. 20. 20.]
     [ 30. 30. 30.]]

    Second array:
    [ 1. 2. 3.]

    First Array + Second Array
    [[ 1. 2. 3.]
     [ 11. 12. 13.]
     [ 21. 22. 23.]
     [ 31. 32. 33.]]

![alt text](/assets/img/broadcasting.jpg)

## Iterating Over Array

NumPy package contains an iterator object **numpy.nditer**. It is an efficient multidimensional iterator object using which it is possible to iterate over an array. Each element of an array is visited using Python’s standard Iterator interface.

    import numpy as np
    a = np.arange(0,60,5)
    a = a.reshape(3,4)

    print  'Original array is:'
    print a
    print  '\n'

    print  'Modified array is:'
    for x in np.nditer(a):
        print x,

The output of this program is as follows −

    Original array is:
    [[ 0 5 10 15]
     [20 25 30 35]
     [40 45 50 55]]

    Modified array is:
    0 5 10 15 20 25 30 35 40 45 50 55

## Iteration Order

If the same elements are stored using F-style order, the iterator chooses the more efficient way of iterating over an array.

    import numpy as np
    a = np.arange(0,60,5)
    a = a.reshape(3,4)

    print  'Original array is:'
    print a
    print  '\n'

    print  'Transpose of the original array is:'
    b = a.T
    print b
    print  '\n'

    print  'Sorted in C-style order:'
    c = b.copy(order =  'C')
    print c

    for x in np.nditer(c):
       print x,

    print  '\n'
    print  'Sorted in F-style order:'

    c = b.copy(order =  'F')
    print c
    for x in np.nditer(c):
       print x,

Its output would be as follows −

    Original array is:
    [[ 0 5 10 15]
     [20 25 30 35]
     [40 45 50 55]]

    Transpose of the original array is:
    [[ 0 20 40]
     [ 5 25 45]
     [10 30 50]
     [15 35 55]]

    Sorted in C-style order:
    [[ 0 20 40]
     [ 5 25 45]
     [10 30 50]
     [15 35 55]]

    0 20 40 5 25 45 10 30 50 15 35 55

    Sorted in F-style order:
    [[ 0 20 40]
     [ 5 25 45]
     [10 30 50]
     [15 35 55]]

    0 5 10 15 20 25 30 35 40 45 50 55

It is possible to force **nditer** object to use a specific order by explicitly mentioning it.

    import numpy as np
    a = np.arange(0,60,5)
    a = a.reshape(3,4)

    print  'Original array is:'
    print a
    print  '\n'
    print  'Sorted in C-style order:'

    for x in np.nditer(a, order =  'C'):
    print x,

    print  '\n'
    print  'Sorted in F-style order:'

    for x in np.nditer(a, order =  'F'):
     print x,

Its output would be −

    Original array is:
    [[ 0 5 10 15]
     [20 25 30 35]
     [40 45 50 55]]

    Sorted in C-style order:
    0 5 10 15 20 25 30 35 40 45 50 55

    Sorted in F-style order:
    0 20 40 5 25 45 10 30 50 15 35 55

### Modifying Array Values

The **nditer** object has another optional parameter called **op_flags**. Its default value is read-only, but can be set to read-write or write-only mode. This will enable modifying array elements using this iterator.

    import numpy as np
    a = np.arange(0,60,5)
    a = a.reshape(3,4)

    print  'Original array is:'
    print a
    print  '\n'

    for x in np.nditer(a, op_flags =  ['readwrite']):
    x[...]  =  2*x

    print  'Modified array is:'
    print a

Its output is as follows −

    Original array is:
    [[ 0 5 10 15]
     [20 25 30 35]
     [40 45 50 55]]

    Modified array is:
    [[ 0 10 20 30]
     [ 40 50 60 70]
     [ 80 90 100 110]]

### Broadcasting Iteration

If two arrays are **broadcastable**, a combined **nditer** object is able to iterate upon them concurrently. Assuming that an array **a** has dimension 3X4, and there is another array **b** of dimension 1X4, the iterator of following type is used (array **b** is broadcast to size of **a**).

    import numpy as np
    a = np.arange(0,60,5)
    a = a.reshape(3,4)

    print  'First array is:'
    print a
    print  '\n'
    print  'Second array is:'

    b = np.array([1,  2,  3,  4], dtype =  int)
    print b

    print  '\n'
    print  'Modified array is:'

    for x,y in np.nditer([a,b]):
      print  "%d:%d"  %  (x,y),

Its output would be as follows −

    First array is:
    [[ 0 5 10 15]
     [20 25 30 35]
     [40 45 50 55]]

    Second array is:
    [1 2 3 4]

    Modified array is:
    0:1 5:2 10:3 15:4 20:1 25:2 30:3 35:4 40:1 45:2 50:3 55:4
