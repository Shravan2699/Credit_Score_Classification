import pandas as pd
import numpy as np
import os

data = pd.read_csv('./Desktop/Business_Analytics_Fundamentals(Group 9)/test.csv')

# print(data)

#Computing Mean for columns

average_EMI = data["Total_EMI_per_month"].mean()


print(data.info())
print(data.describe())

print(data['Credit_Mix'].value_counts())

# average_age = data["Age"].mean()
print(average_EMI)
# print(average_age)

current_directory = os.getcwd()
print(f"This is the cwd {current_directory}") 
