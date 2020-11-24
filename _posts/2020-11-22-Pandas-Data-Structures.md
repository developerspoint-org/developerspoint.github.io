---
date: 2020-11-22 23:00:00
layout: post
title: Pandas Data Structures
subtitle: "Welcome to our 2nd Blog in Pandas Series"
description: >-
  Pandas is a Python package providing high-performance, fast. flexible data structures and data analysis tools for the Python programming language. It has its own data structure. In other words, it’s basic use is for data manipulation and data analysis
image: >-
  https://blog.developerspoint.org/assets/img/pandas_week_2.jpg
optimized_image: >-
  https://blog.developerspoint.org/assets/img/pandas_week_2.jpg
category: pandas
tags:
  - python
  - blog
  - pandas
author: Shantanu Mukherjee
paginate: true
---

# What we will learn in this blog?

In this blog we will learn the 3 data structure of pandas namely a series, dataframe and panels. This structures we will work on during whole tutorials.


# Pandas deals with the following three data structures 

1. Series – A 1D array
2. DataFrame – A 2D array
3. Panel – A 3D array

Lets explore it one by one

# Series

Series is an **1D array** and index of the array is also called **labels**.


## Syntax 

```
pandas.Series( data, index, dtype, copy)
```

## Arguments

1. data - It will take a list or array as input and represent it as series.

2. index - For indexing purposes just like array

3. dtype - Type of data which we want to insert

4.	copy - Copy data. Default is False


## Syntax for creating series 
```
import pandas
s = pandas.Series()
print s
```

****Output****
```
Series([], dtype: float64)
```

# Work out practicals

Practical 1) Create a Series from ndarray (numpy array)
```
#import the pandas and naming it as pd
import pandas as pd
import numpy as np
data = np.array(['a','b','c'])
s = pd.Series(data)
print s
```
****Output**** 
```
0   a 
1   b
2   c
```
Practical 2) Create a Series from ndarray (numpy array) using indexes
```
import pandas as pd
import numpy as np
data = np.array(['a','b','c','d'])
s = pd.Series(data,index=[100,101,102,103])
print s
```

****Output****
```
100  a
101  b
102  c
103  d
```

*Observe that we passed the index values here which can be seen as an **Output***

Practical 3) Create a Series from dict

```
import pandas as pd
import numpy as np
data = {'a' : 0., 'b' : 1., 'c' : 2.}
s = pd.Series(data)
print s
```
****Output****
```
a 0.0
b 1.0
c 2.0
```

Observe that Dictionary keys are used as an index.

Practical 4) *What will happen if we create a Series from only a constant number?*

**If an index is provided, the value will be repeated to match the length of index.**
```
import pandas as pd
import numpy as np
s = pd.Series(5, index=[0, 1, 2, 3])
print s
```
**Output** −
```
0  5
1  5
2  5
3  5
```
## Accessing Data from Series

*Data in the series can be accessed similar to that in an array.*

Practical 5)
Retrieving the first element. 
```
import pandas as pd
s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])

#retrieve the first element
print s[0]
```
**Output** −
```
1
```

Practical 6)
Retrieve the first three elements in the Series. If a : is inserted in front of it, all items from that index onwards will be extracted. 
```
import pandas as pd
s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])

#retrieve the first three element
print s[:3]
```
**Output** −
```
a  1
b  2
c  3
```
## Retrieve Data Using Label (index)

Practical 7)
Retrieve an element using label value.
```
import pandas as pd
s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])

#retrieve a single element
print s['a']
```
**Output**
```
1
```

Practical 8)
Retrieve multiple elements using a list of index label values.

```
import pandas as pd
s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])
#retrieve multiple elements
print s[['a','c','d']]
```
**Output** −
```
a  1
c  3
d  4
```


# Dataframe

Similar to series Dataframe is just a **2 D array**. It is the most important structure *careerwise* for pandas.

## Syntax
```
pandas.DataFrame( data, index, columns, dtype, copy)
```

*Data, index, column, dtype, copy are same as of pandas* 

**Columns is index for a column**


## Create an Empty DataFrame
```
import pandas as pd
df = pd.DataFrame()
print df
```
**Output** 
```
Empty DataFrame
Columns: []
Index: []
```
# Worked out practicals

Practical 1)
Creating a DataFrame from Lists

```
import pandas as pd
data = [1,2,3,4,5]
df = pd.DataFrame(data)
print df
```
**Output** −
```
     0
0    1
1    2
2    3
3    4
4    5
```
Practical 2) Making a dataframe
```
import pandas as pd
data = [['A',10],['B',12],['C',13]]
df = pd.DataFrame(data,columns=['Name','Value'])
print df
```
**Output** 
```
      Name     Value
0     A      10
1     B      12
2     C   13
```
Practical 3)
Changing dtype to float
```
import pandas as pd
data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'],dtype=float)
print df
```
**Output** 
```
      Name     Age
0     Alex     10.0
1     Bob      12.0
2     Clarke   13.0
```
**Observe, the dtype parameter changes the type of Age column to floating point.**

## Create a DataFrame from Dict of ndarrays (array of numpy) / Lists


Practical 1)
```
import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data)
print df
```
**Output** −
```
     Age      Name
0     28        Tom
1     34       Jack
2     29      Steve
3     42      Ricky
```
**Observe the values 0,1,2,3. They are the default index assigned to each using the function range(n).**


Practical 2)
Create an indexed DataFrame using arrays.

```
import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data, index=['rank1','rank2','rank3','rank4'])
print df
```
**Output** −
```
         Age    Name
rank1    28      Tom
rank2    34     Jack
rank3    29    Steve
rank4    42    Ricky
```
**Observe, the index parameter assigns an index to each row.**

## Create a DataFrame from List of Dicts


Practical 1)
The following Practical shows how to create a DataFrame by passing a list of dictionaries.

```
import pandas as pd
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
print df
```
**Output** −
```
    a    b      c
0   1   2     NaN
1   5   10   20.0
```
**Observe, NaN (Not a Number) is appended in missing areas.**

Practical 2)Create a DataFrame by passing a list of dictionaries and the row indices.

```
import pandas as pd
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data, index=['first', 'second'])
print df
```
**Output** −
```
        a   b       c
first   1   2     NaN
second  5   10   20.0
```
Practical 3)
Create a DataFrame with a list of dictionaries, row indices, and column indices.

```
import pandas as pd
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]

#With two column indices, values same as dictionary keys
df1 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b'])

#With two column indices with one index with other name
df2 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b1'])
print df1
print df2
```
**Output** −

#df1 **Output**
```
         a  b
first    1  2
second   5  10
```
#df2 **Output**
```
         a  b1
first    1  NaN
second   5  NaN
```

**Observe, df2 DataFrame is created with a column index other than the dictionary key**

## Create a DataFrame from Dict of Series
Dictionary of Series can be passed to form a DataFrame.

Practical
```
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b','c']),
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b' 'c', 'd'])}

df = pd.DataFrame(d)
print df
```
**Output** 
```
      one    two
a     1.0    1
b     2.0    2
c     3.0    3
d     NaN    4
```
**Observe, for the series one, there is no label ‘d’ passed, but in the result, for the d label, NaN is appended with NaN.**

## Let us now understand column selection, addition, and deletion through Practicals.

## Column Selection
We will understand this by selecting a column from the DataFrame.

Practical
```
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print df ['one']
```
**Output** 
```
a     1.0
b     2.0
c     3.0
d     NaN
```
## Column Addition

We will understand this by adding a new column to an existing data frame.

Practical
```
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)

# Adding a new column to an existing DataFrame object with column label by passing new series

print ("Adding a new column by passing as Series:")
df['three']=pd.Series([10,20,30],index=['a','b','c'])
print df

print ("Adding a new column using the existing columns in DataFrame:")
df['four']=df['one']+df['three']

print df
```
**Output** −
```
Adding a new column by passing as Series:
     one   two   three
a    1.0    1    10.0
b    2.0    2    20.0
c    3.0    3    30.0
d    NaN    4    NaN

Adding a new column using the existing columns in DataFrame:
      one   two   three    four
a     1.0    1    10.0     11.0
b     2.0    2    20.0     22.0
c     3.0    3    30.0     33.0
d     NaN    4     NaN     NaN
```
Column Deletion
Columns can be deleted or popped; let us take an Practical to understand how.

Practical

## Using the previous DataFrame, we will delete a column using del function
```
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']), 
   'three' : pd.Series([10,20,30], index=['a','b','c'])}

df = pd.DataFrame(d)
print ("Our dataframe is:")
print df

# using del function
print ("Deleting the first column using DEL function:")
del df['one']
print df

# using pop function
print ("Deleting another column using POP function:")
df.pop('two')
print df
```
**Output** −
```
Our dataframe is:
      one   three  two
a     1.0    10.0   1
b     2.0    20.0   2
c     3.0    30.0   3
d     NaN     NaN   4

Deleting the first column using DEL function:
      three    two
a     10.0     1
b     20.0     2
c     30.0     3
d     NaN      4

Deleting another column using POP function:
   three
a  10.0
b  20.0
c  30.0
d  NaN
```
## Selection.

### Selection by Label

*Rows can be selected by passing row label to a loc function.*

```
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print df.loc['b']
```
**Output** −
```
one 2.0
two 2.0
Name: b, dtype: float64
```
*The result is a series with labels as column names of the DataFrame. And, the Name of the series is the label with which it is retrieved.*

### Selection by integer location
*Rows can be selected by passing integer location to an iloc function.*

```
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print df.iloc[2]
```
**Output** 
```
one   3.0
two   3.0
Name: c, dtype: float64
```
### Slice Rows
*Multiple rows can be selected using ‘ : ’ operator.*

```
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print df[2:4]
```
**Output** 
```
   one  two
c  3.0    3
d  NaN    4
```
### Addition of Rows
*Add new rows to a DataFrame using the append function.* 

```
import pandas as pd

df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])

df = df.append(df2)
print df
```
**Output** 
```
   a  b
0  1  2
1  3  4
0  5  6
1  7  8
```
### Deletion of Rows
*Use index label to delete or drop rows from a DataFrame.* 

**If you observe, in the above Practical, the labels are duplicate. Let us drop a label and will see how many rows will get dropped.**

```
import pandas as pd

df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])

df = df.append(df2)

# Drop rows with label 0
df = df.drop(0)

print df
```
**Output** −
```
  a b
1 3 4
1 7 8
```
In the above Practical, two rows were dropped because those two contain the same label 0.
# Panel

A panel is a **3D array** of data. 

## Let's first understand some basic terms
* items − axis 0, each item corresponds to a DataFrame contained inside.

* major_axis − axis 1, it is the rows of the DataFrames.

* minor_axis − axis 2, it is the columns of the DataFrames.

## Syntax
```
pandas.Panel(data, items, major_axis, minor_axis, dtype, copy)
```
## The parameters are 


1. data	- Data takes various forms like ndarray, series, map, lists, dict, constants and also another DataFrame
2. items	axis=0
3. major_axis	 axis=1
4. minor_axis 	axis=2
5. dtype	Data type of each column
6. copy	Copy data. Default, false
Create Panel
## A Panel can be created using multiple ways like 

* From ndarrays
* From dict of DataFrames
* From 3D ndarray

# creating an empty panel
```
import pandas as pd
import numpy as np

data = np.random.rand(2,4,5)
p = pd.Panel(data)
print p
```
**Output** −
```
<class 'pandas.core.panel.Panel'>
Dimensions: 2 (items) x 4 (major_axis) x 5 (minor_axis)
Items axis: 0 to 1
Major_axis axis: 0 to 3
Minor_axis axis: 0 to 4
```
**Observe the dimensions of the empty panel and the above panel, all the objects are different.**


### **See you in next tutorial !**

# Resources

* [Pandas documentation](https://pandas.pydata.org/docs/)

* [Pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html#user-guide)

* [Scipy](https://www.scipy.org/)
* [Pandas repository ](https://github.com/pandas-dev/pandas)
 
### Continued in next Week....









