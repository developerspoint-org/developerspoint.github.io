---
date: 2020-11-30 20:00:00
layout: post
title: Introduction To NumPy-Arrays
subtitle: "Welcome to our 3nd Blog in NumPy Series"
description: >-
  Numpy arrays are a commonly used scientific data structure in Python that store data as a grid, or a matrix.
image: >-
  /assets/img/NumPy-Blog-3.jpg
optimized_image: >-
  /assets/img/NumPy-Blog-3.jpg
category: numpy
tags:
  - python
  - blog
  - NumPy
author: Sai Charan.M
paginate: true
---

# NumPy Blog - 3

---

"_Consistency is the hallmark of the unimaginative_ "
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; - _Oscar Wilde_

Bonjour 👋👋Readers...! This Week we are back with an another blog in NumPy Series Keeping the Consistency💪💪 .In this Blog we will Discuss about....

- Numpy Arrays
- Data Types in Numpy

## NumPy Arrays

---

### What are NumPy Arrays...?🤔

**Numpy** arrays are a commonly used scientific data structure in **Python** that store data as a grid, or a matrix.

In **Python**, data structures are objects that provide the ability to organize and manipulate data by defining the relationships between data values stored within the data structure and by providing a set of functionality that can be executed on the data structure.

Like **Python** lists, **NumPy** arrays are also composed of ordered values (called elements) and also use indexing to organize and manipulate the elements in the **NumPy** arrays.

A key characteristic of **NumPy** arrays is that all elements in the array must be the same type of data (i.e. all integers, floats, text strings, etc).

Unlike lists which do not require a specific **Python** package to be defined (or worked with), **NumPy** arrays are defined using the `array()` function from the **NumPy** package.

To this function, you can provide a list of values (i.e. the elements) as the input parameter:

`array = numpy.array([0.7 , 0.75, 1.85])`

The example above creates a **NumPy** array with a simple grid structure along one dimension. However, the grid structure of **NumPy** arrays allow them to store data along multiple dimensions (e.g. rows, columns) that are relative to each other. This dimensionality makes **NumPy** arrays very efficient for storing large amounts of data of the same type and characteristic.

### Key Differences Between Python Lists and Numpy Arrays

While **Python** lists and **NumPy** arrays have similarities in that they are both collections of values that use indexing to help you store and access data, there are a few key differences between these two data structures:

1.  Unlike a **Python** list, all elements in a **NumPy** arrays must be the same data type (i.e. all integers, decimals, text strings, etc).
2.  Because of this requirement, **NumPy** arrays support arithmetic and other mathematical operations that run on each element of the array (e.g. element-by-element multiplication). Recall that lists cannot have these numeric calculations applied directly to them.
3.  Unlike a **Python** list, a **NumPy** array is not edited by adding/removing/replacing elements in the array. Instead, each time that the **NumPy** array is manipulated in some way, it is actually deleted and recreated each time.
4.  **NumPy** arrays can store data along multiple dimensions (e.g. rows, columns) that are relative to each other. This makes **NumPy** arrays a very efficient data structure for large datasets.

### Dimensionality of NumPy Arrays

**NumPy** arrays can be:

- One-dimensional composed of values along one dimension (resembling a **Python** list).
- Two-dimensional composed of rows of individual arrays with one or more columns.
- Multi-dimensional composed of nested arrays with one or more dimensions.

For **NumPy** arrays, brackets `[]` are used to assign and identify the dimensions of the **NumPy** arrays.

This first example below shows how a single set of brackets `[]` are used to define a one-dimensional array.

     #Import numpy with alias np

     import numpy as np


    # Monthly avg precip for Jan through Mar in Boulder, CO

    avg_monthly_precip = np.array([0.70, 0.75, 1.85])

    print(avg_monthly_precip)

```
[0.7  0.75 1.85]

```

Notice that the output of the one-dimensional **NumPy** array is also contained within a single set of brackets `[]`.

To create a two-dimensional array, you need to specify two sets of brackets `[]`, the outer set that defines the entire array structure and inner sets that define the rows of the individual arrays.

     # Monthly precip for Jan through Mar in 2002 and 2013
     precip_2002_2013 = np.array([
         [1.07, 0.44, 1.50],
         [0.27, 1.13, 1.72]
     ])

     print(precip_2002_2013)

```
[[1.07 0.44 1.5 ]
 [0.27 1.13 1.72]]

```

Notice again that the output of the two-dimensional **NumPy** array is contained with two sets of brackets `[]`, which is an easy, visual way to identify whether the **NumPy** array is two-dimensional.

Dimensionality will remain a key concept for working with **NumPy** arrays, as you learn more throughout this chapter including how to use attributes of the **NumPy** arrays to identify the number of dimensions and how to use indexing to slice (i.e. select) data from **NumPy** arrays.

## Data Types In NumPy

---

NumPy has some extra data types, and refer to data types with one character, like `i` for integers, `u` for unsigned integers etc.

Below is a list of all data types in NumPy and the characters used to represent them.

- `i` - integer
- `b` - boolean
- `u` - unsigned integer
- `f` - float
- `c` - complex float
- `m` - timedelta
- `M` - datetime
- `O` - object
- `S` - string
- `U` - unicode string
- `V` - fixed chunk of memory for other type ( void )

### Checking the Data Type of an Array

The NumPy array object has a property called `dtype` that returns the data type of the array.

For Example:

1. Get the data type of an array object:

```
    import numpy as np

    arr = np.array([1, 2, 3, 4])

    print(arr.dtype)
```

```
      #Output

      int64
```

2. Get the data type of an array containing strings:

```
    import numpy as np

    arr = np.array(['apple', 'banana', 'cherry'])

    print(arr.dtype)
```

```
    #Output

    <U6
```

### Creating Arrays With a Defined Data Type

We use the `array()` function to create arrays, this function can take an optional argument: `dtype` that allows us to define the expected data type of the array elements

For Example:

1.  Create an array with data type string:

```
    import  numpy  as  np

    arr = np.array([1,  2,  3,  4],  dtype='S')

    print(arr)
    print(arr.dtype)
```

```
    #Output
    [b'1' b'2' b'3' b'4']
    |S1
```

### Converting Data Type on Existing Arrays

The best way to change the data type of an existing array, is to make a copy of the array with the `astype()` method.

The `astype()` function creates a copy of the array, and allows you to specify the data type as a parameter.

The data type can be specified using a string, like `'f'` for float, `'i'` for integer etc. or you can use the data type directly like `float` for float and `int` for integer.

For Example:

1. Change data type from float to integer by using `'i'` as parameter value:

```
   import numpy as np

   arr = np.array([1.1, 2.1, 3.1])

   newarr = arr.astype('i')

   print(newarr)
   print(newarr.dtype)
```

2. Change data type from float to integer by using `int` as parameter value:

```
   import numpy as np

   arr = np.array([1.1, 2.1, 3.1])

   newarr = arr.astype(int)

   print(newarr)
   print(newarr.dtype)
```

### Resources

- [Data Types](https://www.tutorialspoint.com/numpy/numpy_data_types.htm)
- [NumPy.org](https://numpy.org/doc/stable/user/basics.types.html)

### Continued in Next Week...
