import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns

# Importin dataset
dataset=pd.read_csv('2015.csv')
req_dataset=dataset.iloc[:,[0,3]]
req_dataset['year']=pd.Series(2015,index=req_dataset.index)
dataset1=pd.read_csv('2016.csv')
req_dataset1=dataset1.iloc[:,[0,3]]
req_dataset1['year']=pd.Series(2016,index=req_dataset1.index)
dataset2=pd.read_csv('2017.csv')
req_dataset2=dataset2.iloc[:,[0,2]]
req_dataset2['year']=pd.Series(2017,index=req_dataset2.index)
req_dataset2=req_dataset2.rename(columns={'Happiness.Score':'Happiness Score' })
fin_dataset=[req_dataset,req_dataset1,req_dataset2]
result=pd.concat(fin_dataset)
indicators=pd.read_csv('Data1.csv')
col_names=['Country']
col_names.append('Year')
col_names.append('Score')

#ty_col contain type of all colomns
ty_col={}
ty_col['Country']=str
ty_col['Year']=int
ty_col['Score']=float


indi=pd.read_csv('Data2.csv')


#getting all indicators name and type
for i in range(0,indi.shape[0]):
    ty_col[indi.iloc[i][0]]=float
    col_names.append(indi.iloc[i][0])

#create dataframe for indicatrs
temp_result=result.copy()
temp_result.set_index("Country",inplace=True)
result1  = pd.DataFrame(columns = col_names)
indicators.set_index("Country Name",inplace=True)
k=0

for i in range(0,158):
    country=result.iloc[i]["Country"]
    score_details=temp_result.loc[country]
    data_2015={}
    data_2016={}
    data_2017={}
    try:
        data_2015['Year']=2015
        data_2015['Country']=[country]
        data_2015['Score']=score_details.iloc[0,0]
        data_2016['Year']=2016
        data_2016['Country']=[country]
        data_2016['Score']=score_details.iloc[1,0]
        data_2017['Year']=2017
        data_2017['Country']=[country]
        data_2017['Score']=score_details.iloc[2,0]
        c_list=indicators.loc[country]
        for j in range(c_list.shape[0]):
            data_2015[c_list.iloc[j]['Series Code']]=c_list.iloc[j]['2015 [YR2015]']
            data_2016[c_list.iloc[j]['Series Code']]=c_list.iloc[j]['2016 [YR2016]']
            data_2017[c_list.iloc[j]['Series Code']]=c_list.iloc[j]['2017 [YR2017]']
        temp_frame = pd.DataFrame.from_dict(data_2015)
        temp_frame1 = pd.DataFrame.from_dict(data_2016)
        temp_frame2 = pd.DataFrame.from_dict(data_2017)
        frames=[result1,temp_frame,temp_frame1,temp_frame2]
        result1=pd.concat(frames,ignore_index=True)
    except:
        print("Country "+country+" is not found")
        
        
#replacing .. will nan values
result1=result1.replace("..", np.nan)

#converting the type of dataframe to required type
result1 = result1.astype(ty_col)

#removing unwanted clomns. Colomns having lot of null values (>200)
del result1["FP.WPI.TOTL"]
del result1["SE.ADT.1524.LT.ZS"]
del result1["SH.XPD.EHEX.CH.ZS"]
del result1["per_allsp.cov_pop_tot"]
del result1["per_sa_allsa.cov_pop_tot"]
del result1["per_si_allsi.cov_q1_tot"]
del result1["per_si_allsi.cov_q5_tot"]
del result1["SE.XPD.PRIM.ZS"]
del result1["SE.XPD.SECO.ZS"]
ty_col.pop("FP.WPI.TOTL",None)
ty_col.pop("SE.ADT.1524.LT.ZS",None)
ty_col.pop("SH.XPD.EHEX.CH.ZS",None)
ty_col.pop("per_allsp.cov_pop_tot",None)
ty_col.pop("per_sa_allsa.cov_pop_tot",None)
ty_col.pop("per_si_allsi.cov_q1_tot",None)
ty_col.pop("per_si_allsi.cov_q5_tot",None)
ty_col.pop("SE.XPD.PRIM.ZS",None)
ty_col.pop("SE.XPD.SECO.ZS",None)


#filling missing value with prev year values or the incremental value from previous years
i=0
while(i<result1.shape[0]):
    j=0
    #print(i)
    
    while(j<result1.shape[1]):
        #print(j)
        if(result1.columns.values[j]=="Country"):
            j=j+1
            continue
        if(math.isnan(result1.loc[i+2,result1.columns.values[j]])):
            if(math.isnan(result1.loc[i+1,result1.columns.values[j]])==False and math.isnan(result1.loc[i,result1.columns.values[j]])==False):
                result1.loc[i+2,result1.columns.values[j]]=result1.loc[i+1,result1.columns.values[j]]+(result1.loc[i+1,result1.columns.values[j]] - result1.loc[i,result1.columns.values[j]])
            elif(math.isnan(result1.loc[i,result1.columns.values[j]])==False):
                result1.loc[i+2,result1.columns.values[j]]=result1.loc[i+1,result1.columns.values[j]]=result1.loc[i,result1.columns.values[j]]
        j=j+1
    i=i+3
        

#filling null values with mean of the colomns
#result1.isnull().sum()
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN")
country_result=result1.iloc[:,1]
del result1["Country"]
imputer=imputer.fit(result1.iloc[:,:])
result1.iloc[:,:]=imputer.transform(result1.iloc[:,:])


#Removing outliers (since the dataset is small, the percentile taken is only 5%
for i in range(result1.shape[1]):
    if(result1.columns.values[i]=="Country" or result1.columns.values[i]=="Year" or result1.columns.values[i]=="Score"):
        continue
    mean=np.mean(result1.iloc[:,i])
    low_val=np.percentile(result1.iloc[:,i], 5, interpolation='lower')
    high_val=np.percentile(result1.iloc[:,i], 95, interpolation='lower')
    j=0
    while(j<result1.shape[0]):
        if(result1.loc[j,result1.columns.values[i]]> high_val or result1.loc[j,result1.columns.values[i]]< low_val):
            result1.loc[j,result1.columns.values[i]]=mean
        j=j+1

result1["Country"]=country_result

#plotting variations of all indicators with the score to get an idea of their relation
y_val=result1.loc[:,"Score"]
for i in range(result1.shape[1]):
    if(result1.columns.values[i]=="Country" or result1.columns.values[i]=="Year" or result1.columns.values[i]=="Score"):
        continue
    x_val=result1.iloc[:,i]
    plt.scatter(x_val,y_val,c='b',marker='o')
    plt.xlabel(result1.columns.values[i], fontsize=16)
    plt.ylim([0,10])
    plt.xlim([x_val.min(),x_val.max()])
    plt.ylabel('score', fontsize=16)
    plt.title('scatter plot - indi score',fontsize=20)
    plt.show()

# Train test split
x=result1.copy()
del x["Country"]
del x["Score"]
del x["Year"]
y=result1.loc[:,"Score"]
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)


# fitting multiple linear model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#prediction of test data
y_pred=regressor.predict(x_test)

#plotting a scatterplot
sns.jointplot(y_test,y_pred)

#imporving model
#model selection through backward elimination and a significance level of .1%
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((393,1)).astype(int),values=x,axis=1)   #add a colomn for b0(constant)
x_opt=x[:,:]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#removing x16(16) as its sig value is very high
attr_list=[]
removed=[16]
for i in range(25):
    if(not(i in removed)):
        attr_list.append(i)
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()


#removing x1(1) as its sig value is very high
attr_list=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24]
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#removing x10(11)(not that x10 !=10) as its sig value is very high
attr_list=[0,2,3,4,5,6,7,8,9,10,12,13,14,15,17,18,19,20,21,22,23,24]
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#removing x11(13) as its sig value is very high
attr_list=[0,2,3,4,5,6,7,8,9,10,12,14,15,17,18,19,20,21,22,23,24]
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#removing x13(17) as its sig value is very high
attr_list=[0,2,3,4,5,6,7,8,9,10,12,14,15,18,19,20,21,22,23,24]
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#removing x2(3) as its sig value is very high
attr_list=[0,2,4,5,6,7,8,9,10,12,14,15,18,19,20,21,22,23,24]
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#removing x7(9) as its sig value is very high
attr_list=[0,2,4,5,6,7,8,10,12,14,15,18,19,20,21,22,23,24]
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#removing x17(24) as its sig value is very high
attr_list=[0,2,4,5,6,7,8,10,12,14,15,18,19,20,21,22,23]
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#removing x2(4) as its sig value is very high
attr_list=[0,2,5,6,7,8,10,12,14,15,18,19,20,21,22,23]
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#removing x1(2) as its sig value is very high
attr_list=[0,5,6,7,8,10,12,14,15,18,19,20,21,22,23]
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#removing x5(12) as its sig value is very high
attr_list=[0,5,6,7,10,14,15,18,19,20,21,22,23]
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#removing x12(23) as its sig value is very high
attr_list=[0,5,6,7,10,12,14,15,18,19,20,21,22]
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#removing x2(6) as its sig value is very high
attr_list=[0,5,7,10,12,14,15,18,19,20,21,22]
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#removing x3(10) as its sig value is very high
attr_list=[0,5,7,12,14,15,18,19,20,21,22]
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#removing x5(15) as its sig value is very high
attr_list=[0,5,7,12,14,18,19,20,21,22]
x_opt=x[:,attr_list]
regressor_OLS =sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x=result1.copy()
del x["Country"]
del x["Score"]
del x["Year"]
x_new=x[:,attr_list]
y=result1.loc[:,"Score"]
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_opt,y,test_size=.2,random_state=0)

# fitting multiple linear model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#prediction of test data
y_pred=regressor.predict(x_test)
import seaborn as sns
sns.jointplot(y_test,y_pred)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
np.sqrt(((y_pred - y_test) ** 2).mean())

######################End of linear regression

######################Random Tree regression

from sklearn.ensemble import RandomForestRegressor
R_regressor= RandomForestRegressor(n_estimators=10,random_state=0)
R_regressor.fit(x_train,y_train)

R_y_pred=R_regressor.predict(x_test)
sns.jointplot(y_test,R_y_pred)
r2_score(y_test,R_y_pred)
np.sqrt(((R_y_pred - y_test) ** 2).mean())

#########################    end of program        #########################
