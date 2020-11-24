---
date: 2020-11-15 23:00:00
layout: post
title: Introduction To Numpy
subtitle: "Welcome to our 1st Blog in NumPy Series"
description: >-
  NumPy is a Python package. It stands for Numerical Python. It is a library consisting of multidimensional array objects and a collection of routines for processing of array
image: >-
  /assets/img/numpyintro.png
optimized_image: >-
  /assets/img/numpyintro.png
category: numpy
tags:
  - python
  - blog
  - NumPy
author: Sai Charan.M
paginate: true
---

# Introduction to NumPy

NumPy is a Python package. It stands for **Numerical Python**. It is a library consisting of multidimensional array objects and a collection of routines for processing of array.

![alt text](/assets/img/numpy-blog.png)

NumPy enriches the programming language Python with powerful data structures, implementing multi-dimensional arrays and matrices. These data structures guarantee efficient calculations with matrices and arrays. The implementation is even aiming at huge matrices and arrays, better know under the heading of **_big data_**.

#### At First Glance

- NumPy is an open-source numerical Python library.
- NumPy contains a multi-dimensional array and matrix data structures.
- It can be utilised to perform a number of mathematical operations on arrays such as trigonometric, statistical, and algebraic routines. Therefore, the library contains a large number of mathematical, algebraic, and transformation functions.
- It is an extension of Numeric and Numarray which also contains random number generators.

### Why should we use NumPy..?🤔

If you’re a Python programmer who hasn’t encountered NumPy, you’re potentially missing out. NumPy is an open-source Python library for scientific and numeric computing that lets you work with multi-dimensional arrays far more efficiently than Python alone. It’s probably one of the top five Python packages, and there have been a couple of books written about it

- It’s fast
- Works very well with SciPy and other Libraries
- lets you do matrix arithmetic
- Has lots of built-in functions
- Has universal functions
- Designed for scientific computation

It's free, i.e. it doesn't cost anything and open source. NumPy is an extension on Python rather than a programming language on it's own. NumPy uses Python syntax. Because NumPy is Python, embedding code from other languages like C, C++ and Fortran is very simple.

### Advantages of using NumPy over Python Lists

In Python we have lists that serve the purpose of arrays, but they are slow to process.NumPy aims to provide an array object that is up to 50x faster that traditional Python lists.

The array object in NumPy is called ndarray, it provides a lot of supporting functions that make working with ndarray very easy.Arrays are very frequently used in data science, where speed and resources are very important.

##### 1. Memory usage

---

The most important gain is the memory usage. It also provides a mechanism of specifying the data types of the contents, which allows further optimisation of the code.

As an example, we can create a simple array of six elements using a python list as well as by using `numpy.array(...)` , The difference in amount of memory occupied by it is quite astounding. See the example below

```
import numpy as np
import sys

py_arr = [1,2,3,4,5,6]
numpy_arr = np.array([1,2,3,4,5,6])

sizeof_py_arr = sys.getsizeof(1) * len(py_arr)           # Size = 168
sizeof_numpy_arr = numpy_arr.itemsize * numpy_arr.size   # Size = 48
```

##### 2. Speed

---

Speed is, in fact, a very important property in data structures. Why does it take much less time to use NumPy operations over vanilla python? Let's have a look at the example where we multiply two square matrices.

```
from time import time
import numpy as np
def matmul(A, B):
    N = len(A)
    product = [[0 for x in range(N)] for y in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                product[i][j] += matrix1[i][k] * matrix2[k][j]
    return product
matrix1 = np.random.rand(1000, 1000)
matrix2 = np.random.rand(1000, 1000)
t = time()
prod = matmul(matrix1, matrix1)
print("Normal", time() - t)
t = time()
np_prod = np.matmul(matrix1, matrix2)
print("Numpy", time() - t)
```

The times will be observed as follows:

```
Normal 7.604596138000488
Numpy 0.0007512569427490234
```

We can see that the NumPy implementation is almost 10,000 times faster. Why? Because NumPy uses under-the-hood optimizations such as [transposing](https://numpy.org/doc/stable/reference/generated/numpy.matrix.transpose.html) and [chunked](https://scikit-allel.readthedocs.io/en/stable/model/chunked.html) multiplications. Furthermore, the operations are vectorized so that the looped operations are performed much faster. The NumPy library uses the [BLAS (Basic Linear Algebra Subroutines)](http://www.netlib.org/blas/) library under in its backend. Hence, it is important to install NumPy properly to compile the binaries to fit the hardware architecture.

##### 3.Effect of operations on Numpy array and Python Lists

---

Let us see a example in which the incapability of the Python list to carry out a basic operation is demonstrated. A Python list and a Numpy array having the same elements will be declared and an integer will be added to increment each element of the container by that integer value without looping statements.

```
import numpy as np
ls =[1, 2, 3]
arr = np.array(ls)
try:
    ls = ls + 4
except(TypeError):
    print("Lists don't support list + int")
try:
    arr = arr + 4
    print("Modified Numpy array: ",arr)

except(TypeError):
    print("Numpy arrays don't support list + int")

```

Output:

```
Lists don't support list + int
Modified Numpy array: [5 6 7]
```

#### NumPy – A Replacement for MatLab

NumPy is often used along with packages like SciPy (Scientific Python) and Mat−plotlib (plotting library). This combination is widely used as a replacement for MatLab, a popular platform for technical computing. The combination of NumPy, SciPy and Matplotlib is a free alternative to MATLAB.

Even though MATLAB has a huge number of additional toolboxes available, NumPy has the advantage that Python is a more modern and complete programming language and - as we have said already before - is open source. SciPy adds even more MATLAB-like functionalities to Python. Python is rounded out in the direction of MATLAB with the module Matplotlib, which provides MATLAB-like plotting functionality.

### Resources

- [NumPy documentation](https://numpy.org/doc/)
- [NumPy.Org](https://numpy.org/)
- [NumPy Repository](https://github.com/numpy/numpy)
- [SciPy.Org](https://docs.scipy.org/doc/numpy-1.17.0/user/whatisnumpy.html)

### Continued in next Week....
