import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from pandas import ExcelFile
from pandas import ExcelWriter
import xgboost as xgb

dftrain = pd.read_csv(r"C:\Users\Firoz Jaipuri\Downloads\train_oSwQCTC\train.csv")
dftest = pd.read_csv(r"C:\Users\Firoz Jaipuri\Downloads\test_HujdGe7\test.csv")


dftrain.isnull().sum()

#sns.set(rc={'figure.figsize':(11.7,8.27)})
#sns.distplot(df['Purchase'],bins=30)
#plt.show()

#change Age from Range to numeric 
dftrain.loc[dftrain['Age'] == "0-17", 'Age'] = 15 
dftrain.loc[dftrain['Age'] == '18-25', 'Age'] = 21
dftrain.loc[dftrain['Age'] == '26-35', 'Age'] = 30
dftrain.loc[dftrain['Age'] == '36-45', 'Age'] = 40
dftrain.loc[dftrain['Age'] == '46-50', 'Age'] = 48
dftrain.loc[dftrain['Age'] == '51-55', 'Age'] = 53
dftrain.loc[dftrain['Age'] == '55+', 'Age'] = 60

dftest.loc[dftest['Age'] == "0-17", 'Age'] = 15 
dftest.loc[dftest['Age'] == '18-25', 'Age'] = 21
dftest.loc[dftest['Age'] == '26-35', 'Age'] = 30
dftest.loc[dftest['Age'] == '36-45', 'Age'] = 40
dftest.loc[dftest['Age'] == '46-50', 'Age'] = 48
dftest.loc[dftest['Age'] == '51-55', 'Age'] = 53
dftest.loc[dftest['Age'] == '55+', 'Age'] = 60


# Change stayInCity from 4+ to 4 then change this column to have integer values 
dftrain.loc[dftrain['Stay_In_Current_City_Years'] == '4+', 'Stay_In_Current_City_Years'] = '4' 
TrainIntStay = dftrain.iloc[:,6] 
TrainStayInCity = pd.to_numeric(TrainIntStay, downcast='signed')
dftrain.drop(['Stay_In_Current_City_Years'],axis=1, inplace=True)
dftrain = pd.concat([dftrain, TrainStayInCity], axis=1)

dftest.loc[dftest['Stay_In_Current_City_Years'] == '4+', 'Stay_In_Current_City_Years'] = '4'
IntStay = dftest.iloc[:,6] 
TestStayInCity = pd.to_numeric(IntStay, downcast='signed')
dftest.drop(['Stay_In_Current_City_Years'],axis=1, inplace=True)
dftest = pd.concat([dftest, TestStayInCity], axis=1)


# Change Gender to Binary
dftrain.loc[dftrain['Gender'] == 'M', 'Gender'] = 0 
dftrain.loc[dftrain['Gender'] == 'F', 'Gender'] = 1

dftest.loc[dftest['Gender'] == 'M', 'Gender'] = 0 
dftest.loc[dftest['Gender'] == 'F', 'Gender'] = 1

#Remove rows with product category >19
dftrain2 = dftrain.loc[dftrain['Product_Category_1'] < 19]
dftest2 = dftest.loc[dftest['Product_Category_1'] < 19]

# OneHot encoding for City_Category
dftrain2 = pd.concat([dftrain2,pd.get_dummies(dftrain2['City_Category'], prefix='City_Category')],axis=1)
dftrain2.drop(['City_Category'],axis=1, inplace=True)

dftest2 = pd.concat([dftest2,pd.get_dummies(dftest2['City_Category'], prefix='City_Category')],axis=1)
dftest2.drop(['City_Category'],axis=1, inplace=True)

# remove NANs from Product_Category 2 and 3 
dftest2['Product_Category_2'].fillna(0, inplace=True)
dftest2['Product_Category_3'].fillna(0, inplace=True)
dftrain2['Product_Category_2'].fillna(0, inplace=True)
dftrain2['Product_Category_3'].fillna(0, inplace=True)

#Count by UserIds
Usercounttraindf = dftrain2.groupby('User_ID')['User_ID'].transform('count')
Usercounttrain = pd.DataFrame(Usercounttraindf.values, columns = ['UserCount'])
dftrain2 = pd.concat([dftrain2, Usercounttrain], axis=1)

Usercounttestdf = dftest2.groupby('User_ID')['User_ID'].transform('count')
Usercounttest = pd.DataFrame(Usercounttestdf.values, columns = ['UserCount'])
dftest2 = pd.concat([dftest2, Usercounttest], axis=1)


#Count by Product Ids
Prodcounttraindf = dftrain2.groupby('Product_ID')['Product_ID'].transform('count')
Prodcounttrain = pd.DataFrame(Prodcounttraindf.values, columns = ['ProdCount'])
dftrain2 = pd.concat([dftrain2, Prodcounttrain], axis=1)

Prodcounttestdf = dftest2.groupby('Product_ID')['Product_ID'].transform('count')
Prodcounttest = pd.DataFrame(Prodcounttestdf.values, columns = ['ProdCount'])
dftest2 = pd.concat([dftest2, Prodcounttest], axis=1)
 
#Prodmeantraindf = dftrain2.groupby('Product_ID')['Purchase'].transform('mean')
#Prodmeantrain = pd.DataFrame(Prodmeantraindf.values, columns = ['ProdMean'])
#dftrain2 = pd.concat([dftrain2, Prodmeantrain], axis=1)

#Prodmeantestdf = dftest2.groupby('Product_ID')['ProdMean'].transform('mean')
#Prodmeantest = pd.DataFrame(Prodmeantraindf.values, columns = ['ProdMean'])
#dftest2 = pd.concat([dftest2, Prodmeantrain[0:50,:]], axis=1)


#Create X by removing Purchase and ProductId columns
XUserId = pd.DataFrame(dftrain2.iloc[:,0])
X = pd.DataFrame(dftrain2.iloc[:,2:9])
Ytrain = dftrain2.iloc[:,9]
Z = pd.DataFrame(dftrain2.iloc[:,10:17])
X = pd.concat([XUserId, X, Z], axis=1)
Xtrain = X.iloc[:,0:17]


#Create Xtest
XtestUI = pd.DataFrame(dftest2.iloc[:,0])
XtestPI = pd.DataFrame(dftest2.iloc[:,1])
Xtest = pd.DataFrame(dftest2.iloc[:,2:16])
Xtest = pd.concat([XtestUI, Xtest], axis=1)
Xtesttest = Xtest.iloc[:,0:16]
Xtesttest.head

data_dmatrix = xgb.DMatrix(data=Xtrain,label=Ytrain)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
               max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(Xtrain,Ytrain)
preds = xg_reg.predict(Xtesttest)

Ypred = pd.DataFrame(preds, columns = ['Purchase'])
Result = pd.concat([XtestUI, XtestPI, Ypred], axis = 1)
Result.to_csv(r'C:\Users\Firoz Jaipuri\Downloads\SampleSubmission6.csv')
#dftrain2.to_csv(r'C:\Users\Firoz Jaipuri\Downloads\train_oSwQCTC\Xtrain.csv')

