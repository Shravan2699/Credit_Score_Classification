#Test data is used to evalute the model's accuracy after training the data
from ast import increment_lineno
from xml.etree.ElementInclude import include
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
from sklearn.metrics import confusion_matrix
# from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


sns.set_style("whitegrid")
sns.set_context("paper",font_scale=1.0)
#This is just for styling

# %matplotlib inline

pd.options.display.max_rows = 1000 #Truncates the rows if they are over 100
pd.options.display.max_columns = 100 #Truncates the columns if they are over 100
pd.options.display.float_format = '{:.2f}'.format #Only displays upto 2 decimal points for numericals

import warnings
warnings.filterwarnings('ignore')

##Above code imports all the necessary libraries and helps with the presentation of data


#Loading the dataset
df = pd.read_csv('./data/train.csv')
# print(df.head())

df.columns = [x.lower() for x in df.columns]

# print(df.columns)

#Checking data quality

print(df.shape)

#Dropping not needed columns
df.drop(['id','customer_id','month','name','ssn','type_of_loan','credit_history_age'],axis=1,inplace=True)

print(df.info())
print(df.duplicated().value_counts()) 


# #Checking for null values in the dataset next
print(df.isnull().sum().reset_index().rename(columns={'index': 'feature_name',0: 'null_counts'}))
print(df.shape[0])


# #Dropping null values
# #This line filters the DataFrame df by keeping only those rows where the sum of NaN values across columns is less than 3. It drops rows that have more than 2 NaN values.
size_before_cleaning = df.shape
df = df[df.isnull().sum(axis=1) < 3]
print(f"{size_before_cleaning[0] - df.shape[0]} records dropped")



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
df["num_of_delayed_payment"] = df["num_of_delayed_payment"].astype("float")

df["num_of_delayed_payment"].fillna(df["num_of_delayed_payment"].median(), inplace=True)



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

df = df[df['age'] <= 120]
df = df[df['num_bank_accounts'] <= 20]
df = df[df['num_credit_card'] <= 20]
df = df[df['num_of_loan'] <= 20]


df['occupation'] = df['occupation'].replace('_______',np.nan)
df["occupation"] = df["occupation"].fillna(np.random.choice(pd.Series(['Scientist','Teacher','Engineer','Entrepreneur','Developer','Lawyer','Media_Manager','Doctor','Journalist','Manager','Accountant','Musician','Mechanic','Writer','Architect'])))


df['credit_mix'] = df['credit_mix'].replace('_',np.nan)
df['credit_mix'] = df['credit_mix'].fillna(np.random.choice(pd.Series(['Standard','Good','Bad'])))


df['payment_of_min_amount'] = df['payment_of_min_amount'].replace('NM',np.nan)
df['payment_of_min_amount'] = df['payment_of_min_amount'].fillna(np.random.choice(pd.Series(['Yes','No'])))

# df['payment_behaviour'] = df['payment_behaviour'].replace() 

df['payment_behaviour'] = df['payment_behaviour'].replace('!@9#%8',np.nan)
df['payment_behaviour'] = df['payment_behaviour'].fillna(np.random.choice(pd.Series(['High_spent_Small_value_payments','Low_spent_Large_value_payments','High_spent_Large_value_payments','High_spent_Medium_value_payments','Low_spent_Large_value_payments','Low_spent_Small_value_payments','Low_spent_Medium_value_payments'])))
# print(df['payment_behaviour'].isnull().sum())
df["changed_credit_limit"] = df["changed_credit_limit"].apply(lambda x:x.split("-")[-1])
#So, the entire line is transforming each element in the 'changed_credit_limit' column by splitting it at '-' and keeping only the last part. The result is a new column with the extracted values.
df.drop(df[df["changed_credit_limit"].str.contains("_")].index, inplace=True)
#Next,we drop the rows with values "_"
df["changed_credit_limit"] = df["changed_credit_limit"].astype("float")
#Then,we convert the data type of the column to float

for i in ['monthly_inhand_salary',"amount_invested_monthly",'num_credit_inquiries','monthly_balance']:
    df[i] = df[i].fillna(df[i].median())





# #######################################################################################
#EDA
print(df.info())

df_cat = df.select_dtypes(include='object')
df_num = df.select_dtypes(include='number')
print(df_num.info())
print(df_cat.info())


#Categorical features:

#The below code gives the count of unique values in each column of the categorical columns that we have
for col in df_cat:
    print(df_cat[col].value_counts())
    print('\n---------------------------')



# #Occupation
plt.figure(figsize=(10,5))
ax = sns.countplot(x='occupation',data=df_cat,order=df['occupation'].value_counts().index)
plt.title('Counts of Unique Occupations')
plt.xlabel('Occupation')
plt.ylabel('Counts')
for p in ax.patches:
    ax.annotate('{}'.format(p.get_height()),(p.get_x()+0.1,p.get_height()+50))

plt.xticks(rotation=90)
plt.show()


# # #Credit Mix
# plt.figure(figsize=(10,5))
# ax = sns.countplot(x='credit_mix',data=df_cat,order=df['credit_mix'].value_counts().index)
# plt.title('Credit Mix Counts')
# plt.xlabel('Credit Mix')
# plt.ylabel('Counts')
# for p in ax.patches:
#     ax.annotate('{}'.format(p.get_height()),(p.get_x()+0.1,p.get_height()+50))

# # plt.xticks(rotation=90)
# plt.show()

# print(df_cat.info())


# # #Payment of Minimum Amount
# plt.figure(figsize=(10,5))
# ax = sns.countplot(x='payment_of_min_amount',data=df_cat,order=df['payment_of_min_amount'].value_counts().index)
# plt.title('Payment of min_amount Counts')
# plt.xlabel('Minimum Amount Paid?')
# plt.ylabel('Counts')
# for p in ax.patches:
#     ax.annotate('{}'.format(p.get_height()),(p.get_x()+0.1,p.get_height()+50))

# # plt.xticks(rotation=90)
# plt.show()

# # #Payment Behaviour
# plt.figure(figsize=(10,5))
# ax = sns.countplot(x='payment_behaviour',data=df_cat,order=df['payment_behaviour'].value_counts().index)
# plt.title('Payment behaviour Counts')
# plt.xlabel('Payment Behaviour')
# plt.ylabel('Counts')
# for p in ax.patches:
#     ax.annotate('{}'.format(p.get_height()),(p.get_x()+0.1,p.get_height()+50))
# plt.xticks(rotation=45)
# plt.show()

# # #Credit Score
# plt.figure(figsize=(10,5))
# ax = sns.countplot(x='credit_score',data=df_cat,order=df['credit_score'].value_counts().index)
# plt.title('Credit_score Counts')
# plt.xlabel('Credit Score')
# plt.ylabel('Counts')
# for p in ax.patches:
#     ax.annotate('{}'.format(p.get_height()),(p.get_x()+0.1,p.get_height()+50))
# # plt.xticks(rotation=45)
# plt.show()


# #NUMERICAL DATA COLUMNS


#Correlation Heatmap
corr = df_num.corr()


plt.figure(figsize=(10,5))
sns.heatmap(corr.corr(),cmap="GnBu",center=0,annot=True)
plt.show()

numeric_cols =  df.select_dtypes(exclude="object").columns
# cat_cols = df.select_dtypes(include="object").columns
# print(numeric_cols)
# print(cat_cols)

# #Credit score correlation with all the other features

# #VIF
#We will only consider the numeric columns to compute this
vif_df = df[numeric_cols]
X = sm.add_constant(vif_df)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# Calculate VIF
# sm.OLS(X[col], X.drop(col, axis=1)): sm refers to the statsmodels library, and OLS stands for Ordinary Least Squares, which is a method for estimating the parameters of a linear regression model.
# For each iteration, this part of the code fits a simple linear regression model where the current column (col) is the dependent variable, and all other columns (excluding the current one) are independent variables.
# fit(): This method fits the linear regression model to the data.
# .rsquared: The rsquared attribute retrieves the coefficient of determination for the fitted regression model.It measures the proportion of the variance in the dependent variable explained by the independent variables.
# The entire formula for VIF is 1/1-R^2
# VIF scores for each variables indicate Athe extent to which the variance of each variable is inflated due to potential multicollinearity with other independent variables. Higher VIF values suggest a higher degree of multicollinearity for the corresponding variable.
vif_values = [1 / (1 - sm.OLS(X[col], X.drop(col, axis=1)).fit().rsquared) for col in X.columns[1:]]

# Create DataFrame
vif_data = pd.DataFrame({'feature': X.columns[1:], 'VIF': vif_values})

print(vif_data)

#We will now remove the features with vif greater than 2 as values above 2 are seen as a signal that there may be multicollinearity issues.
df = df.drop(['monthly_inhand_salary','amount_invested_monthly','monthly_balance'],axis=1)
# df = df.dropna(subset=df.columns.values)


#Data Preparation
X = df.drop('credit_score', axis=1)  # Features
y = df['credit_score']  # Target variable

# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#We will take 20% data as test data

numeric_cols_train = X_train.select_dtypes(include=['number']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col in numeric_cols_train]

X_train_encoded = pd.get_dummies(X_train, columns=['occupation', 'credit_mix', 'payment_of_min_amount', 'payment_behaviour'])
X_test_encoded = pd.get_dummies(X_test, columns=['occupation', 'credit_mix', 'payment_of_min_amount', 'payment_behaviour'])
# # #We use one-hot encoding for our categorical features to make them binary 


scaler = StandardScaler()
X_train_encoded[numeric_cols] = scaler.fit_transform(X_train_encoded[numeric_cols])
X_test_encoded[numeric_cols] = scaler.transform(X_test_encoded[numeric_cols])



# # #Model-Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_encoded, y_train)

y_pred = logreg.predict(X_test_encoded)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# print(df_cat.info())
print(numeric_cols)
print(X_train.columns)


# #Age-outlier values-needds to be 

# print(df.describe())

# print(X_train_encoded.columns)
# print(X_test_encoded.columns)

# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=logreg.classes_, yticklabels=logreg.classes_)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()