import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_education = pd.read_csv('csv/Education_Data.csv')

# Filter the rows which has 'Jordan' as entry in the Nationality Column
jordan_filter_row = df_education['Nationality'] == 'Jordan' 
df_jordan = df_education[jordan_filter_row]
print(df_jordan.head())  


## Select the 'VisitedResources' and 'GradeID' Columns
df_jordan_column = df_jordan[['VisitedResources', 'GradeID']]
print(df_jordan_column.head())

#Use Pandas mean() function to find the mean of VisitedResources in Jordan
mean_visited_resources = df_jordan_column['VisitedResources'].mean()
print(mean_visited_resources,'mean')

# Use Pandas median() function to find the median of VisitedResources in Jordan
median_visited_resources = df_jordan_column['VisitedResources'].median()
print(median_visited_resources,'median')

# Use Pandas mode() function to find the mode of GradeID in Jordan
mode_grade_id = df_jordan_column['GradeID'].mode()
print(mode_grade_id,'mode')