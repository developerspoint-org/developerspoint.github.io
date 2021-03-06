---
date: 2021-03-06 23:59:59
layout: post
title: Array Manipulation Part-2
subtitle: "Welcome to our 8th Blog in NumPy Series"
description: >-
  This Blog will present several examples of using NumPy array manipulation to Change dimensions,Joining Arrays and splitting arrays
image: >-
  /assets/img/numpy-blog-8.jpg
optimized_image: >-
  /assets/img/numpy-blog-8.jpg
category: numpy
tags:
  - python
  - blog
  - NumPy
author: Sai Charan.M
paginate: true
---

# NumPy Blog - 8

In the previous Blog, We learned about Array Manipulation [Here is the Link for the previous week’s Blog in case if you have missed it](http://blog.developerspoint.org/Array-Manipulation-part-1/). In this Blog we are going to perceive about……

# Array Manipulation

## Changing Dimensions

**1. Broadcast:**

As seen earlier, NumPy has in-built support for broadcasting. This function mimics the broadcasting mechanism. It returns an object that encapsulates the result of broadcasting one array against the other.

The function takes two arrays as input parameters. Following example illustrates its use.

**Example**:

    import numpy as np
    x = np.array([[1], [2], [3]])
    y = np.array([4, 5, 6])

    # tobroadcast x against y
    b = np.broadcast(x,y)


    print 'Broadcast x against y:'
    r,c = b.iters
    print r.next(), c.next()
    print r.next(), c.next()
    print '\n'
    # shape attribute returns the shape of broadcast object

    print 'The shape of the broadcast object:'
    print b.shape
    print '\n'
    # to add x and y manually using broadcast
    b = np.broadcast(x,y)
    c = np.empty(b.shape)

    print 'Add x and y manually using broadcast:'
    print c.shape
    print '\n'
    c.flat = [u + v for (u,v) in b]

    print 'After applying the flat function:'
    print c
    print '\n'
    # same result obtained by NumPy's built-in broadcasting support

    print 'The summation of x and y:'
    print x + y

**Output:**

    Broadcast x against y:
    1 4
    1 5

    The shape of the broadcast object:
    (3, 3)

    Add x and y manually using broadcast:
    (3, 3)

    After applying the flat function:
    [[ 5. 6. 7.]
     [ 6. 7. 8.]
     [ 7. 8. 9.]]

    The summation of x and y:
    [[5 6 7]
     [6 7 8]
     [7 8 9]]

**2. Broadcast_to**:

This function broadcasts an array to a new shape. It returns a read-only view on the original array. It is typically not contiguous. The function may throw ValueError if the new shape does not comply with NumPy's broadcasting rules.

The function takes the following parameters.

> numpy.broadcast_to(array, shape, subok)

**Example**

    import numpy as np
    a = np.arange(4).reshape(1,4)

    print 'The original array:'
    print a
    print '\n'

    print 'After applying the broadcast_to function:'
    print np.broadcast_to(a,(4,4))

**It should produce the following output −**

    [[0  1  2  3]
     [0  1  2  3]
     [0  1  2  3]
     [0  1  2  3]]

**3. Expands_Dims**

This function expands the array by inserting a new axis at the specified position. Two parameters are required by this function.

> numpy.expand_dims(arr, axis)

**Example**

    import numpy as np
    x = np.array(([1,2],[3,4]))

    print 'Array x:'
    print x
    print '\n'
    y = np.expand_dims(x, axis = 0)

    print 'Array y:'
    print y
    print '\n'

    print 'The shape of X and Y array:'
    print x.shape, y.shape
    print '\n'
    # insert axis at position 1
    y = np.expand_dims(x, axis = 1)

    print 'Array Y after inserting axis at position 1:'
    print y
    print '\n'

    print 'x.ndim and y.ndim:'
    print x.ndim,y.ndim
    print '\n'

    print 'x.shape and y.shape:'
    print x.shape, y.shape

**The output of the above program would be as follows −**

    Array x:
    [[1 2]
     [3 4]]

    Array y:
    [[[1 2]
     [3 4]]]

    The shape of X and Y array:
    (2, 2) (1, 2, 2)

    Array Y after inserting axis at position 1:
    [[[1 2]]
     [[3 4]]]

    x.ndim and y.ndim:
    2 3

    x.shape and y.shape:
    (2, 2) (2, 1, 2)

## Joining Arrays

**1. Concatenate:**

Concatenation refers to joining. This function is used to join two or more arrays of the same shape along a specified axis. The function takes the following parameters.

> numpy.concatenate((a1, a2, ...), axis)

**Example**

    import numpy as np
    a = np.array([[1,2],[3,4]])

    print 'First array:'
    print a
    print '\n'
    b = np.array([[5,6],[7,8]])

    print 'Second array:'
    print b
    print '\n'
    # both the arrays are of same dimensions

    print 'Joining the two arrays along axis 0:'
    print np.concatenate((a,b))
    print '\n'

    print 'Joining the two arrays along axis 1:'
    print np.concatenate((a,b),axis = 1)

**Its output is as follows −**

    First array:
    [[1 2]
     [3 4]]

    Second array:
    [[5 6]
     [7 8]]

    Joining the two arrays along axis 0:
    [[1 2]
     [3 4]
     [5 6]
     [7 8]]

    Joining the two arrays along axis 1:
    [[1 2 5 6]
     [3 4 7 8]]

**2. Stack:**

This function joins the sequence of arrays along a new axis. This function has been added since NumPy version 1.10.0. Following parameters need to be provided.

> numpy.stack(arrays, axis)

**Example**

    import numpy as np
    a = np.array([[1,2],[3,4]])

    print 'First Array:'
    print a
    print '\n'
    b = np.array([[5,6],[7,8]])

    print 'Second Array:'
    print b
    print '\n'

    print 'Stack the two arrays along axis 0:'
    print np.stack((a,b),0)
    print '\n'

    print 'Stack the two arrays along axis 1:'
    print np.stack((a,b),1)

**It should produce the following output −**

    First array:
    [[1 2]
     [3 4]]

    Second array:
    [[5 6]
     [7 8]]

    Stack the two arrays along axis 0:
    [[[1 2]
     [3 4]]
     [[5 6]
     [7 8]]]

    Stack the two arrays along axis 1:
    [[[1 2]
     [5 6]]
     [[3 4]
     [7 8]]]

## Splitting Arrays

**1. Split**
This function divides the array into subarrays along a specified axis. The function takes three parameters.

> numpy.split(ary, indices_or_sections, axis)

**Example**

    import numpy as np
    a = np.arange(9)

    print 'First array:'
    print a
    print '\n'

    print 'Split the array in 3 equal-sized subarrays:'
    b = np.split(a,3)
    print b
    print '\n'

    print 'Split the array at positions indicated in 1-D array:'
    b = np.split(a,[4,7])
    print b

**Its output is as follows −**

    First array:
    [0 1 2 3 4 5 6 7 8]

    Split the array in 3 equal-sized subarrays:
    [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]

    Split the array at positions indicated in 1-D array:
    [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8])]

**2. Hsplit:**

The numpy.hsplit is a special case of split() function where axis is 1 indicating a horizontal split regardless of the dimension of the input array.

**Example**

    import numpy as np
    a = np.arange(16).reshape(4,4)

    print 'First array:'
    print a
    print '\n'

    print 'Horizontal splitting:'
    b = np.hsplit(a,2)
    print b
    print '\n'

**Its output would be as follows −**

    First array:
    [[ 0 1 2 3]
     [ 4 5 6 7]
     [ 8 9 10 11]
     [12 13 14 15]]

    Horizontal splitting:
    [array([[ 0,  1],
           [ 4,  5],
           [ 8,  9],
           [12, 13]]), array([[ 2,  3],
           [ 6,  7],
           [10, 11],
           [14, 15]])]

**3. Vsplit:**

numpy.vsplit is a special case of split() function where axis is 1 indicating a vertical split regardless of the dimension of the input array. The following example makes this clear.

**Example**

    import numpy as np
    a = np.arange(16).reshape(4,4)

    print 'First array:'
    print a
    print '\n'

    print 'Vertical splitting:'
    b = np.vsplit(a,2)
    print b

**Its output would be as follows −**

    First array:
    [[ 0 1 2 3]
     [ 4 5 6 7]
     [ 8 9 10 11]
     [12 13 14 15]]

    Vertical splitting:
    [array([[0, 1, 2, 3],
           [4, 5, 6, 7]]), array([[ 8,  9, 10, 11],
           [12, 13, 14, 15]])]

### Resources

- [NumPy.org](https://numpy.org/doc/stable/reference/routines.array-manipulation.html)

### Continued in Next Week...
