import pandas as pd
from statsmodels.stats import weightstats

# Print the null and alternative hypthesis
print("H0: The mean of the two samples are equal")
print("H1: The mean of the two samples are not equal")


df = pd.read_csv('csv/CarPrice.csv') # PATH HERE

# Apply ztest to calculate the p-value 
ztest ,pval1 = weightstats.ztest(df['price'], x2=None, value=15000, alternative='smaller')
print("p-value for ztest is: ",float(pval1))
print("")
print("Since the p-value is less than 0.05, we reject the null hypothesis and accept the alternative hypothesis")
print("The mean of the two samples are not equal")

# Compare p-value with given significance level and print whether the null hypothesis is rejected or cannot be rejected 
comp = pval1 < 0.05
if comp == True:
    print("We reject the null hypothesis")
else:
    print("We cannot reject the null hypothesis")