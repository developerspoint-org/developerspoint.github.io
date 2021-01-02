---
date: 2021-01-03 23:59:59
layout: post
title: Indexing and Slicing of NumPy Arrays
subtitle: "Welcome to our 5th Blog in NumPy Series"
description: >-
  Contents of ndarray object can be accessed and modified by indexing or slicing, just like Python's in-built container objects.
image: >-
  /assets/img/NumPy-blog-5.jpg
optimized_image: >-
  /assets/img/NumPy-blog-5.jpg
category: numpy
tags:
  - python
  - blog
  - NumPy
author: Sai Charan.M
paginate: true
---

# NumPy Blog - 5

Hello Admirers🙃🙃
Last week learnt how to create arrays and basic operations on arrays[Here is the Link for the previous week's Blog in case if you have missed it](http://blog.developerspoint.org/Array-Creation-In-Numpy/). In this Blog we are going to perceive about......

1. Adding, Sorting, Deleting Elements
2. Indexing and Slicing of NumPy Arrays
3. Advanced Indexing

## Adding/Removing/Sorting Elements

We can perform a number of fast operations on a Numpy array. This makes Numpy a desirable library for the Python users.

**To add elements:**

    a = [0]
    np.append(a, [1,2]) #adds 1,2 at the end
    #insert can also be used if we want to insert along a given indexThis will return [0,1,2]

**To delete elements:**

    np.delete(array, 1) #1 is going to be deleted from the array
    e.g.
    a = np.delete([0,1,2], 1) #results in [0,2]

**Sorting**

    To sort an array, call the sort(array, axis, kind, orderby) function:

    np.sort(array1, axis=1, kind = 'quicksort')
    e.g.
    a = np.sort([[0,3,2],[1,2,3]], axis=1, kind = 'quicksort')
    results in
    [[0 2 3]
     [1 2 3]]

## Indexing and Slicing

Contents of ndarray object can be accessed and modified by indexing or slicing, just like Python's in-built container objects.

Items in ndarray object follows zero-based index. Three types of indexing methods are available − **field access, basic slicing** and **advanced indexing**.

Basic slicing is an extension of Python's basic concept of slicing to n dimensions. A Python slice object is constructed by giving **start, stop**, and **step** parameters to the built-in **slice** function. This slice object is passed to the array to extract a part of array.

    import numpy as np
    a = np.arange(10)
    s = slice(2,7,2)
    print a[s]

Its output is as follows −

`[2 4 6]`

In the above example, an **ndarray** object is prepared by **arange()** function. Then a slice object is defined with start, stop, and step values 2, 7, and 2 respectively. When this slice object is passed to the ndarray, a part of it starting with index 2 up to 7 with a step of 2 is sliced.

The same result can also be obtained by giving the slicing parameters separated by a colon : (start:stop:step) directly to the **ndarray** object.

    import numpy as np
    a = np.arange(10)
    b = a[2:7:2]
    print b

Here, we will get the same output −

    [2  4  6]

If only one parameter is put, a single item corresponding to the index will be returned. If a : is inserted in front of it, all items from that index onwards will be extracted. If two parameters (with : between them) is used, items between the two indexes (not including the stop index) with default step one are sliced.

    # slice single item  import numpy as np
    a = np.arange(10)
    b = a[5]
    print(b)

    # slice items starting from index

    c = a[2:]
    print(c)

    # slice items between indexes

    d = a[2:5]
    print(d)

Its output is as follows −

    5
    [2  3  4  5  6  7  8  9]
    [2  3  4]

The above description applies to multi-dimensional **ndarray** too.

Slicing can also include ellipsis (…) to make a selection tuple of the same length as the dimension of an array. If ellipsis is used at the row position, it will return an ndarray comprising of items in rows.

    # array to begin with
    import numpy as np
    a = np.array([[1,2,3],[3,4,5],[4,5,6]])
    print  'Our array is:'
    print a
    print  '\n'

    # this returns array of items in the second column

    print  'The items in the second column are:'
    print a[...,1]
    print  '\n'

    # Now we will slice all items from the second row

    print  'The items in the second row are:'
    print a[1,...]
    print  '\n'

    # Now we will slice all items from column 1 onwards

    print  'The items column 1 onwards are:'
    print a[...,1:]

The output of this program is as follows −

    Our array is:
    [[1 2 3]
     [3 4 5]
     [4 5 6]]

    The items in the second column are:
    [2 4 5]

    The items in the second row are:
    [3 4 5]

    The items column 1 onwards are:
    [[2 3]
     [4 5]
     [5 6]]

## Advanced Indexing

It is possible to make a selection from ndarray that is a non-tuple sequence, ndarray object of integer or Boolean data type, or a tuple with at least one item being a sequence object. Advanced indexing always returns a copy of the data. As against this, the slicing only presents a view.

There are two types of advanced indexing − **Integer** and **Boolean**.

### Integer Indexing

This mechanism helps in selecting any arbitrary item in an array based on its N dimensional index. Each integer array represents the number of indexes into that dimension. When the index consists of as many integer arrays as the dimensions of the target ndarray, it becomes straightforward.

In the following example, one element of a specified column from each row of ndarray object is selected. Hence, the row index contains all row numbers, and the column index specifies the element to be selected.

import numpy as np

    x = np.array([[1,  2],  [3,  4],  [5,  6]])
    y = x[[0,1,2],  [0,1,0]]
    print y

Its output would be as follows −

    [1  4  5]

The selection includes elements at (0,0), (1,1) and (2,0) from the first array.

In the following example, elements placed at corners of a 4X3 array are selected. The row indices of selection are [0, 0] and [3,3] whereas the column indices are [0,2] and [0,2].

    import numpy as np
    x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])
    print  'Our array is:'
    print x
    print  '\n'
    rows = np.array([[0,0],[3,3]])
    cols = np.array([[0,2],[0,2]])
    y = x[rows,cols]
    print  'The corner elements of this array are:'
    print y

The output of this program is as follows −

    Our array is:
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]

    The corner elements of this array are:
    [[ 0  2]
     [ 9 11]]

The resultant selection is an ndarray object containing corner elements.

Advanced and basic indexing can be combined by using one slice (:) or ellipsis (…) with an index array. The following example uses slice for row and advanced index for column. The result is the same when slice is used for both. But advanced index results in copy and may have different memory layout.

    import numpy as np
    x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])
    print  'Our array is:'
    print x
    print  '\n'
    # slicing z = x[1:4,1:3]
    print  'After slicing, our array becomes:'
    print z
    print  '\n'
    # using advanced index for column
    y = x[1:4,[1,2]]
    print  'Slicing using advanced index for column:'
    print y

The output of this program would be as follows −

    Our array is:
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]

    After slicing, our array becomes:
    [[ 4  5]
     [ 7  8]
     [10 11]]

    Slicing using advanced index for column:
    [[ 4  5]
     [ 7  8]
     [10 11]]

### Boolean Array Indexing

This type of advanced indexing is used when the resultant object is meant to be the result of Boolean operations, such as comparison operators.

In this example, items greater than 5 are returned as a result of Boolean indexing.

    import numpy as np
    x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])
    print  'Our array is:'
    print x
    print  '\n'
    # Now we will print the items greater than 5  print  'The items greater than 5 are:'
    print x[x >  5]

The output of this program would be −

    Our array is:
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]

    The items greater than 5 are:
    [ 6  7  8  9 10 11]

### Resources

- [GeeksForGeeks](https://www.geeksforgeeks.org/indexing-in-numpy/)
- [NumPy.org](https://numpy.org/devdocs/user/basics.indexing.html)

### Continued in Next Week...
