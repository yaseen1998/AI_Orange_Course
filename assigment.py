import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import json
import seaborn as sns
def Convert_y_to_numeric(y):
    unique_value = y.unique()
    mapping = {}
    for i in range(len(unique_value)):
        mapping[unique_value[i]] = i+1
    y = y.map(mapping)
    return y
    
def clean_data(x, y):
    get_all_columns = x.columns
    result = {"column":{}}
    for column in get_all_columns:
        try:
            column_data = x[column]
            if column_data.dtype == "object":
                if "string_column" not in result:
                    result["string_column"] = []
                result["string_column"].append(column)
                continue
            Hypothesis_testing(column_data, y, result,column)
            
        except Exception as e:
            continue
    with open("result.json", "w") as f:
        json.dump(result, f)


def Hypothesis_testing(column_data, y,result,column):
    p_value = pearsonr(column_data, y)[1]
    if p_value > 0.05:
        if "remove_colmn" not in result:
            result["remove_colmn"] = {
                "column": [],
                "reason": "p_value > 0.04"
            }
        result["remove_colmn"]["column"].append(column)
    else:
        Column_Description(column_data,result,column)
    
    
    


def Column_Description(column_data,result,column):
    description = column_data.describe()
    description_dict = description.to_dict()
    print(column_data.isnull().sum())
    description_dict['null'] = str(column_data.isnull().sum())
    result['column'][column]=Outliar(column_data,description_dict)

def Outliar(column_data,description_dict):
    Q1 = column_data.quantile(0.25)
    Q3 = column_data.quantile(0.75)
    IQR = Q3 - Q1
    lowerBand = Q1 - 1.5 * IQR
    upperBand = Q3 + 1.5 * IQR
    outlier_lower = column_data[(column_data < lowerBand)]
    outlier_upper = column_data[(column_data > upperBand)]
    description_dict["outlier"] = {
        "lower": len(outlier_lower),
        "upper": len(outlier_upper),
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "total": len(outlier_lower) + len(outlier_upper),
    }
    
    return description_dict

df = pd.read_csv("csv/credit_score.csv")
# x all columns except credit score
x = df.iloc[:, 0:-1]
# y credit score
# y  = df.iloc[:, -1]
# y = Convert_y_to_numeric(y)
# clean_data(x, y)

# drop string column

        
number_of_columns = len(x.columns)
divide_number = 3
total_row = number_of_columns // divide_number

#every colum has whole fig but in same column has three fig show all in one figfig, axes = plt.subplots(number_of_columns, 1, figsize=(8, 6 * number_of_columns))
fig, axes = plt.subplots(total_row+1, divide_number, figsize=(15, 20 ))
ax_row = 0
ax_col = 0
for i in x.columns:
    if x[i].dtype == "object":
        continue
    sns.boxplot(x[i], ax=axes[ax_row, ax_col])
    # title
    axes[ax_row, ax_col].set_title(i)
    ax_col += 1
    if ax_col == divide_number:
        ax_row += 1
        ax_col = 0

plt.show()
