---
date: 2021-02-05 23:59:59
layout: post
title: Array Manipulation Part-1
subtitle: "Welcome to our 7th Blog in NumPy Series"
description: >-
  This Blog will present several examples of using NumPy array manipulation to access data and subarrays, and to split, reshape, and join the arrays.
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

# NumPy Blog - 7

In the previous Blog, We learned about Broadcasting and Iteration of NumPy Arrays [Here is the Link for the previous week’s Blog in case if you have missed it](http://blog.developerspoint.org/Broadcasting-and-Iteration/). In this Blog we are going to perceive about……

## Array Manipulation

### Adding / Removing Elements

1. **Resize**:

This function returns a new array with the specified size. If the new size is greater than the original, the repeated copies of entries in the original are contained. The function takes the following parameters.

> numpy.resize(arr, shape)

    import numpy as np
    a = np.array([[1,2,3],[4,5,6]])
    print  'First array:'
    print a
    print  '\n'

    print  'The shape of first array:'
    print a.shape
    print  '\n'

     b = np.resize(a,  (3,2))
     print  'Second array:'
     print b
     print  '\n'

     print  'The shape of second array:'
     print b.shape

The above program will produce the following output −

    First array:
    [[1 2 3]
     [4 5 6]]

    The shape of first array:
    (2, 3)

    Second array:
    [[1 2]
     [3 4]
     [5 6]]

    The shape of second array:
    (3, 2)

2. **Append**:

This function adds values at the end of an input array. The append operation is not inplace, a new array is allocated. Also the dimensions of the input arrays must match otherwise ValueError will be generated.

> numpy.append(arr, values, axis)

    import numpy as np
    a = np.array([[1,2,3],[4,5,6]])

    print 'First array:'
    print a
    print '\n'

    print 'Append elements to array:'
    print np.append(a, [7,8,9])
    print '\n'

    print 'Append elements along axis 0:'
    print np.append(a, [[7,8,9]],axis = 0)
    print '\n'

Its output would be as follows −

    First array:
    [[1 2 3]
     [4 5 6]]

    Append elements to array:
    [1 2 3 4 5 6 7 8 9]

    Append elements along axis 0:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]

3. **Insert**:
   This function inserts values in the input array along the given axis and before the given index. If the type of values is converted to be inserted, it is different from the input array. Insertion is not done in place and the function returns a new array. Also, if the axis is not mentioned, the input array is flattened.

> numpy.insert(arr, obj, values, axis)

    import numpy as np
    a = np.array([[1,2],[3,4],[5,6]])

    print 'First array:'
    print a
    print '\n'

    print 'Axis parameter not passed. The input array is flattened before insertion.'
    print np.insert(a,3,[11,12])
    print '\n'
    print 'Axis parameter passed. The values array is broadcast to match input array.'

    print 'Broadcast along axis 0:'
    print np.insert(a,1,[11],axis = 0)
    print '\n'

Its output would be as follows −

    First array:
    [[1 2]
     [3 4]
     [5 6]]

    Axis parameter not passed. The input array is flattened before insertion.
    [ 1 2 3 11 12 4 5 6]

    Axis parameter passed. The values array is broadcast to match input array.
    Broadcast along axis 0:
    [[ 1 2]
     [11 11]
     [ 3 4]
     [ 5 6]]

4. **Delete**:
   This function returns a new array with the specified subarray deleted from the input array. As in case of insert() function, if the axis parameter is not used, the input array is flattened. The function takes the following parameters −

>     Numpy.delete(arr, obj, axis)

    import numpy as np
    a = np.arange(12).reshape(3,4)

    print 'First array:'
    print a
    print '\n'

    print 'Array flattened before delete operation as axis not used:'
    print np.delete(a,5)
    print '\n'

    print 'Column 2 deleted:'
    print np.delete(a,1,axis = 1)
    print '\n'

    print 'A slice containing alternate values from array deleted:'
    a = np.array([1,2,3,4,5,6,7,8,9,10])
    print np.delete(a, np.s_[::2])

Its output would be as follows −

    First array:
    [[ 0 1 2 3]
     [ 4 5 6 7]
     [ 8 9 10 11]]

    Array flattened before delete operation as axis not used:
    [ 0 1 2 3 4 6 7 8 9 10 11]

    Column 2 deleted:
    [[ 0 2 3]
     [ 4 6 7]
     [ 8 10 11]]

    A slice containing alternate values from array deleted:
    [ 2 4 6 8 10]

### Transpose Operations

1.  **Transpose**:
    This function permutes the dimension of the given array. It returns a view wherever possible. The function takes the following parameters.

>     numpy.transpose(arr, axes)

Example:

      import numpy as np
      a = np.arange(12).reshape(3,4)
      print  'The original array is:'
      print a
      print  '\n'

      print  'The transposed array is:'
      print np.transpose(a)

Its output would be as follows −

    The original array is:
    [[ 0 1 2 3]
     [ 4 5 6 7]
     [ 8 9 10 11]]

    The transposed array is:
    [[ 0 4 8]
     [ 1 5 9]
     [ 2 6 10]
     [ 3 7 11]]

2.  **RollAxis**:
    This function rolls the specified axis backwards, until it lies in a specified position. The function takes three parameters.

> numpy.rollaxis(arr, axis, start)

    # It creates 3 dimensional ndarray  import numpy as np
    a = np.arange(8).reshape(2,2,2)
    print  'The original array:'
    print a
    print  '\n'

    # to roll axis-2 to axis-0 (along width to along depth)

    print  'After applying rollaxis function:'
    print np.rollaxis(a,2)

    print  '\n'
    print  'After applying rollaxis function:'
    print np.rollaxis(a,2,1)

Its output is as follows −

    The original array:
    [[[0 1]
     [2 3]]
     [[4 5]
     [6 7]]]

    After applying rollaxis function:
    [[[0 2]
     [4 6]]
     [[1 3]
     [5 7]]]

    After applying rollaxis function:
    [[[0 2]
     [1 3]]
     [[4 6]
     [5 7]]]

3.  **SwapAxis**:
    This function interchanges the two axes of an array. For NumPy versions after 1.10, a view of the swapped array is returned. The function takes the following parameters.

> numpy.swapaxes(arr, axis1, axis2)

    a = np.arange(8).reshape(2,2,2)
    print  'The original array:'
    print a
    print  '\n'

    print  'The array after applying the swapaxes function:'
    print np.swapaxes(a,  2,  0)

Its output would be as follows −

    The original array:
    [[[0 1]
     [2 3]]

     [[4 5]
      [6 7]]]

    The array after applying the swapaxes function:
    [[[0 4]
     [2 6]]

     [[1 5]
      [3 7]]]

### Changing Shape of Array

1. **Reshape**:
   This function gives a new shape to an array without changing the data. It accepts the following parameters −

> numpy.reshape(arr, newshape, order')

    import numpy as np
    a = np.arange(8)
    print  'The original array:'
    print a
    print  '\n'

    b = a.reshape(4,2)
    print  'The modified array:'
    print b

Its output would be as follows −

    The original array:
    [0 1 2 3 4 5 6 7]

    The modified array:
    [[0 1]
     [2 3]
     [4 5]
     [6 7]]

2. **Flatten**:
   This function returns a copy of an array collapsed into one dimension. The function takes the following parameters.

> ndarray.flatten(order)

    import numpy as np
    a = np.arange(8).reshape(2,4)
    print  'The original array is:'
    print a
    print  '\n'

    # default is column-major
    print  'The flattened array is:'
    print a.flatten()
    print  '\n'

    print  'The flattened array in F-style ordering:'
    print a.flatten(order =  'F')

The output of the above program would be as follows −

    The original array is:
    [[0 1 2 3]
     [4 5 6 7]]

    The flattened array is:
    [0 1 2 3 4 5 6 7]

    The flattened array in F-style ordering:
    [0 4 1 5 2 6 3 7]

3. **Ravel**:
   This function returns a flattened one-dimensional array. A copy is made only if needed. The returned array will have the same type as that of the input array. The function takes one parameter.

> numpy.ravel(a, order)

```
 import numpy as np
 a = np.arange(8).reshape(2,4)
 print  'The original array is:'
 print a
 print  '\n'

 print  'After applying ravel function:'
 print a.ravel()
 print  '\n'

 print  'Applying ravel function in F-style ordering:'
 print a.ravel(order =  'F')
```

Its output would be as follows −

    The original array is:
    [[0 1 2 3]
     [4 5 6 7]]

    After applying ravel function:
    [0 1 2 3 4 5 6 7]

    Applying ravel function in F-style ordering:
    [0 4 1 5 2 6 3 7]

**Flatten** always returns a copy. **ravel** returns a view of the original array whenever possible. This isn't visible **in the** printed output, but if you modify the array returned by **ravel**, it may modify the entries **in the** original array. If you modify the entries **in an** array returned from **flatten** this will never happen.

### Resources

- [NumPy.org](https://numpy.org/doc/stable/reference/routines.array-manipulation.html)

### Continued in Next Week...
