from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#Plotting Normal Distribution
# The code block below samples 1000 numbers from the Normal Distribution with a mean of 100 and standard deviation of 10. The numbers are stored in the nums variable. Use Matplotlib to plot:

nums = np.random.normal(size=1000, loc = 100, scale = 10 )
#Plot a Histogram using 'pyplot.hist'
plt.hist(nums, bins = 100)
#Add a vertical line at the mean.
plt.axvline(nums.mean(), color='red', linestyle='solid', linewidth=2)
#Add a vertical line at one standard deviation above the mean.
plt.axvline(nums.mean() + nums.std(), color='red', linestyle='dashed', linewidth=2)
#Add a vertical line at one standard deviation below the mean.
plt.axvline(nums.mean() - nums.std(), color='red', linestyle='dashed', linewidth=2)
#Print the histogram.
plt.show()

#Plot a Boxplot using 'pyplot.boxplot'
plt.boxplot(nums)
#Print the boxplot.
plt.show()

tele = pd.read_csv('csv/telephone_subs_jordan.csv') #PATH HERE
tele.head()
len(tele) #CHECK LENGTH OF DATASET

# Convert the 'Year' column present in the dataframe into a list (using 'tolist()') and store it in x
list_year = tele['Year'].tolist()
# Convert the 'Telephone Subscriptions' column present in the dataframe into a list (using 'tolist()') and store it in y
list_tele = tele['Telephone Subscriptions'].tolist()

# Plot 'Number of Telephone Subscribers' vs 'Year' Graph (using 'pyplot.plot')


# Plot 'Number of Telephone Subscribers' vs 'Year' Graph (using 'pyplot.plot')
# Use 'pyplot.annotate to place an arrow over the point representing max number of subscribers in the graph.
plt.plot(list_year, list_tele)
plt.annotate('Max', xy=(2005, 1000000), xytext=(2000, 500000),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.show()