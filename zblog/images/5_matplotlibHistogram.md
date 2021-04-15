## Visualize Data Distribution with Histogram

Histogram allows us to visualize the frequency distribution of our data.
It breaks the data into a few smaller bins according to the value of the data, and then count the number of occurences (i.e., the frequency) in each bin.

We can obtain the frequency and bins for a given data using the `histogram()` function from numpy. Let's consider the following example:


```python
import numpy as np

# generate 1000 random numbers
x = np.random.rand(1000, 1)

# count the occurences in each bin in x
frequency, bins = np.histogram(x, bins=10, range=[0, 1])


for b, f in zip(bins[1:], frequency):
    print(f'value: {(round(b,1))} >> frequency: {f}')

```

    value: 0.1 >> frequency: 80
    value: 0.2 >> frequency: 80
    value: 0.3 >> frequency: 98
    value: 0.4 >> frequency: 91
    value: 0.5 >> frequency: 123
    value: 0.6 >> frequency: 97
    value: 0.7 >> frequency: 105
    value: 0.8 >> frequency: 102
    value: 0.9 >> frequency: 117
    value: 1.0 >> frequency: 107
    

Here, we used `numpy.random.rand()` function to generate 1000 uniformly distributed values, ranging from 0 to 1. An array x is defined to store the generated values.
We would like to know how many data is within 0-0.1, how many occurs at 0.1-0.2, and so on.
These can be obtained by calling `numpy.histogram()` function.


## Histogram with Matplotlib
Matplotlib allows us to plot the histogram with `pyplot.hist()` function.
Let's continue with the above example, and use the histogram function in Matplotlib to visualize the data distribution. 


```python
import matplotlib.pyplot as plt 

plt.hist(x, bins = 10)
plt.title("Data Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```


    
![svg](5_matplotlibHistogram_files/5_matplotlibHistogram_4_0.svg)
    


Let's generate another set of random number, but with normal distribution.
Instead of `numpy.random.rand()`, we can use `numpy.random.randn()` to generate a series of values that follow standard normal distribution with zero mean and standard deviation equals to 1.

> Notes:
if we would like to have a normal distribution with specific mean and standard deviation, we can use the following formula:
$$
\sigma * numpy.random.randn() + \mu
$$



```python
# generate 1000 random numbers
x = np.random.randn(1000, 1)

plt.hist(x, bins = 10)
plt.title("Data Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```


    
![svg](5_matplotlibHistogram_files/5_matplotlibHistogram_6_0.svg)
    


## Multiple Histograms in a Single Plot

We can plot multiple histograms for easy comparison. 
Let's create 3 numpy arrays each consists of 1000 normally distributed random numbers based on different mean and standard deviation.


```python
x1 = 3 * np.random.randn(1000,1) + 3
x2 = 2 * np.random.randn(1000,1) + 7
x3 = x 

plt.figure(figsize=(12, 5))
plt.hist(x1, bins = 10, alpha = 0.5, color = 'red', label = 'x1')
plt.hist(x2, bins = 10, alpha = 0.5, color = 'green', label = 'x2')
plt.hist(x3, bins = 10, alpha = 0.5, color = 'blue', label = 'x3')
plt.title("Data Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```


    
![svg](5_matplotlibHistogram_files/5_matplotlibHistogram_8_0.svg)
    


## Density Distribution
Instead of using the number of occurences as the y-axis, we can normalize the occurences frequency by setting density to `True`, as shown below:


```python
plt.figure()
plt.hist(x, bins = 10, density= True)
plt.title("Data Distribution")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.show()
```


    
![svg](5_matplotlibHistogram_files/5_matplotlibHistogram_10_0.svg)
    


## Style the Histogram
We can create spacing between each bin in the histogram using the `set_style()` function from seaborn. 
Note that seaborn is built upon matplotlib, so we can use seaborn and matplotlib together.


```python
import seaborn as sns 
sns.set_style("white")

plt.figure()
plt.hist(x, bins = 10, density= True)
plt.title("Data Distribution")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.show()
```


    
![svg](5_matplotlibHistogram_files/5_matplotlibHistogram_12_0.svg)
    


We can also use the `seaborn.histplot()` function to visualize the histogram and density curve on the same plot.


```python
plt.figure()
# sns.histplot(x, bins = 10, hist_kws={'alpha': 0.5}, kde_kws={'linewidth': 2})
sns.histplot(x, bins = 10, alpha = 0.5, stat="probability", kde = True, legend = False)
plt.title("Data Distribution")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.show()
```


    
![svg](5_matplotlibHistogram_files/5_matplotlibHistogram_14_0.svg)
    



```python

```
