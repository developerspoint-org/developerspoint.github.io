---
date: 2020-12-27 23:00:00
layout: post
title: Descriptive statistics
subtitle: "In this tutorial we will learn about some descriptive statistics"
description: >-
  In this tutorial we will learn about some descriptive statistics such as **sum()**, **mean()** , **avg()**, **sumsum()** etc A large number of methods collectively compute descriptive statistics and other related operations on DataFrame. 
image: >-
  https://blog.developerspoint.org/assets/img/stats.jpg
optimized_image: >-
  https://blog.developerspoint.org/assets/img/stats.jpg
category: pandas
tags:
  - python
  - blog
  - pandas
author: Shantanu Mukherjee
paginate: true
---

# Introduction

In this tutorial we will learn about some descriptive statistics such as **sum()**, **mean()** , **avg()**, **sumsum()** etc A large number of methods collectively compute descriptive statistics and other related operations on DataFrame. 

# Creating a dataframe

Let us create a DataFrame and use this object throughout this tutorial for all the operations.

*Example*

```
import pandas as pd
import numpy as np

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
}

#Create a DataFrame
df = pd.DataFrame(d)
print df
```
**Output** −
```
    Age  Name   Rating
0   25   Tom     4.23
1   26   James   3.24
2   25   Ricky   3.98
3   23   Vin     2.56
4   30   Steve   3.20
5   29   Smith   4.60
6   23   Jack    3.80
7   34   Lee     3.78
8   40   David   2.98
9   30   Gasper  4.80
10  51   Betina  4.10
11  46   Andres  3.65
```
 # sum()
*Returns the sum of the values for the requested axis.*

*Example*
```
import pandas as pd
import numpy as np
 
#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
}

#Create a DataFrame
df = pd.DataFrame(d)
print df.sum()
```
**Output** −
```
Age                                                    382
Name     TomJamesRickyVinSteveSmithJackLeeDavidGasperBe...
Rating                                               44.92

```
*Each individual column is added individually (Strings are appended).*

## For axis=1


*Example*
```
import pandas as pd
import numpy as np
 
#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
}
 
#Create a DataFrame
df = pd.DataFrame(d)
print df.sum(1)
```
**Output** −
```
0    29.23
1    29.24
2    28.98
3    25.56
4    33.20
5    33.60
6    26.80
7    37.78
8    42.98
9    34.80
10   55.10
11   49.65
dtype: float64
```

# mean()
*Returns the average value*

*Example*
```
import pandas as pd
import numpy as np

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
}

#Create a DataFrame
df = pd.DataFrame(d)
print df.mean()
```
**Output**−
```
Age       31.833333
Rating     3.743333
dtype: float64
```
# std()
**Returns the standard deviation of the numerical columns.**

*Example*
```
import pandas as pd
import numpy as np

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
}

#Create a DataFrame
df = pd.DataFrame(d)
print df.std()
```
**Output**−
```
Age       9.232682
Rating    0.661628
```


# describe()
*Returns a summary of statistics pertaining to the DataFrame columns.*

*Example*
```
import pandas as pd
import numpy as np

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
}

#Create a DataFrame
df = pd.DataFrame(d)
print df.describe()
```
**Output** −
```
               Age         Rating
count    12.000000      12.000000
mean     31.833333       3.743333
std       9.232682       0.661628
min      23.000000       2.560000
25%      25.000000       3.230000
50%      29.500000       3.790000
75%      35.500000       4.132500
max      51.000000       4.800000
```


1. object − Summarizes String columns
1. number − Summarizes Numeric columns
1. all − Summarizes all columns together



*Example*
```
import pandas as pd
import numpy as np

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
}

#Create a DataFrame
df = pd.DataFrame(d)
print df.describe(include=['object'])
```
**Output** −
```
          Name
count       12
unique      12
top      Ricky
freq         1
```

*Example*

```
import pandas as pd
import numpy as np

#Create a Dictionary of series
d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
}

#Create a DataFrame
df = pd.DataFrame(d)
print df. describe(include='all')
```
**Output** −
```
          Age          Name       Rating
count   12.000000        12    12.000000
unique        NaN        12          NaN
top           NaN     Ricky          NaN
freq          NaN         1          NaN
mean    31.833333       NaN     3.743333
std      9.232682       NaN     0.661628
min     23.000000       NaN     2.560000
25%     25.000000       NaN     3.230000
50%     29.500000       NaN     3.790000
75%     35.500000       NaN     4.132500
max     51.000000       NaN     4.800000
```

### **See you in next tutorial !**

# Resources

* [Pandas documentation](https://pandas.pydata.org/docs/)

* [Pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html#user-guide)

* [Scipy](https://www.scipy.org/)
* [Pandas repository ](https://github.com/pandas-dev/pandas)
 
### Continued in next Week....