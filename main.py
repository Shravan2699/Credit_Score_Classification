#Test data is used to evalute the model's accuracy after training the data
from ast import increment_lineno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.metrics import classification_report


sns.set_style("whitegrid")
sns.set_context("paper",font_scale=1.0)

# %matplotlib inline

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 100
pd.options.display.float_format = '{:.2f}'.format

import warnings
warnings.filterwarnings('ignore')

##Above code imports all the necessary libraries and helps with the presentation of data


#Loading the dataset
df = pd.read_csv('./data/train.csv')
print(df.head())

df.columns = [x.lower() for x in df.columns]

print(df.columns)

#Checking data quality

df.shape


#Dropping not needed columns
df.drop(['id','customer_id','month','name','ssn','type_of_loan','credit_history_age'],axis=1,inplace=True)

print(df.info())
print(df.duplicated().value_counts()) 

#Checking for null values in the dataset next
print(df.isnull().sum().reset_index().rename(columns={'index': 'feature_name',0: 'null_counts'}))
print(df.shape[0])


#Dropping null values
size_before_cleaning = df.shape
df = df[df.isnull().sum(axis=1) < 3]
print(f"{size_before_cleaning[0] - df.shape[0]} records dropped")

# print(df.head())
# print(df.isnull().sum().reset_index().rename(columns={'index': 'feature_name',0: 'null_counts'}))

def amount_invested_monthly(col):
    if "__" in str(col):
        return str(col).split("_")[1]
    else:
        return str(col)
    
df["amount_invested_monthly"] = df["amount_invested_monthly"].apply(amount_invested_monthly)

df["amount_invested_monthly"].replace('', np.nan, inplace=True)

# Drop rows with NaN in the "amount_invested_monthly" column
df.dropna(subset=["amount_invested_monthly"], inplace=True)
# Convert the column to float
df["amount_invested_monthly"] = df["amount_invested_monthly"].astype(float)

df["amount_invested_monthly"] = df["amount_invested_monthly"].astype("float")

def filter_delayed_payments(value):
    if "_" in str(value):
        return str(value).split("_")[1]
    elif "-" in str(value):
        return str(value).replace("-","")
    elif str(value) == "_":
        return str(value)
    else:
        return str(value)

df["num_of_delayed_payment"] = df["num_of_delayed_payment"].apply(filter_delayed_payments)
df["num_of_delayed_payment"].replace('', np.nan, inplace=True)

# Drop rows with NaN in the "amount_invested_monthly" column
df.dropna(subset=["num_of_delayed_payment"], inplace=True)
# Convert the column to float

def filter_general(value):
    if '-' in str(value):
        return str(value).split('-')[1]
    elif '_' in str(value):
        return str(value).split('_')[0]
    else:
        return str(value)
    
df.drop(df[df["monthly_balance"]=="__-333333333333333333333333333__"].index,inplace=True)
for i in ['age','annual_income','num_of_loan','outstanding_debt','monthly_balance']:
    df[i] = df[i].apply(filter_general)
    df[i] = df[i].astype(np.float64)
    print(i + " Successfully Cleaned")


df['occupation'] = df['occupation'].replace('______',np.nan)
df["occupation"] = df["occupation"].fillna(np.random.choice(pd.Series(['Scientist','Teacher','Engineer','Entrepreneur','Developer','Lawyer','Media_Manager','Doctor','Journalist','Manager','Accountant','Musician','Mechanic','Writer','Architect'])))

df['credit_mix'] = df['credit_mix'].replace('_',np.nan)
df['credit_mix'] = df['credit_mix'].fillna(np.random.choice(pd.Series(['Standard','Good','Bad'])))


df['payment_of_min_amount'] = df['payment_of_min_amount'].replace('NM',np.nan)
df['payment_of_min_amount'] = df['payment_of_min_amount'].fillna(np.random.choice(pd.Series(['Yes','No'])))

# df['payment_behaviour'] = df['payment_behaviour'].replace() 
df['payment_behaviour'] = df['payment_behaviour'].fillna(np.random.choice(pd.Series(['High_spent_Small_value_payments','Low_spent_Large_value_payments','High_spent_Large_value_payments','High_spent_Medium_value_payments','Low_spent_Large_value_payments','Low_spent_Small_value_payments','Low_spent_Medium_value_payments'])))

# print(df.head())
# df["changed_credit_limit"] = df["changed_credit_limit"].apply(lambda x:x.split("-"))[-1]
# df.drop(df[df["changed_credit_limit"]=="_".index],inplace=True)
# df["changed_credit_limit"] = df["changed_credit_limit"].astype("float")
