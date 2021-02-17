#!/usr/bin/env python
# coding: utf-8

# In[259]:


#import packages
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from mlxtend.regressor import StackingCVRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from xgboost import XGBRegressor


#read in data and view data
house_train = pd.read_csv("train.csv")
house_test=pd.read_csv("test.csv")
house=[house_train,house_test]
house_data=pd.concat(house,sort=True)
house_data.head()


#check the demision of the entire data set
house_data.shape


#drop id
house_data = house_data.drop(['Id'],axis=1)
#find missing value
house_data.isnull().sum().sort_values(ascending=False)[0:37]


#drop the variable with lots of mising variables
#drop PoolQC MiscFeature  Alley Fence  
house_data = house_data.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)


#view LotFrontage
print(house_data["LotFrontage"].mode())
print(house_data["LotFrontage"].mean())
print(house_data["LotFrontage"].median())
plt.hist(house_data['LotFrontage'],15)
plt.show()
house_data['LotFrontage'] = house_data['LotFrontage'].fillna(house_data["LotFrontage"].mean())

# MasVnrType & MasVnrArea & GarageYrBlt
house_data["MasVnrType"] = house_data["MasVnrType"].fillna("None")
house_data["MasVnrArea"] = house_data["MasVnrArea"].fillna(0)
house_data["GarageYrBlt"] = house_data["GarageYrBlt"].fillna(0)
# fill NA 
house_data['FireplaceQu'] = house_data['FireplaceQu'].fillna('None')
house_data['GarageQual'] = house_data['GarageQual'].fillna('None')
house_data['GarageCond'] = house_data['GarageCond'].fillna('None')
house_data['BsmtExposure'] = house_data['BsmtExposure'].fillna('None')
house_data['BsmtFinType2'] = house_data['BsmtFinType2'].fillna('None')
house_data['BsmtFinType1'] = house_data['BsmtFinType1'].fillna('None')
house_data['BsmtCond'] = house_data['BsmtCond'].fillna('None')
house_data['BsmtQual'] = house_data['BsmtQual'].fillna('None')
house_data['GarageType'] = house_data['GarageType'].fillna('NA')
house_data['GarageFinish'] = house_data['GarageFinish'].fillna('NA')
house_data['FireplaceQu'] = house_data['FireplaceQu'].fillna('None')


#mapping ordinal data
#create mapping 
value_trans={'TA':2,'Gd':3, 'Fa':1,'Ex':4,'Po':1,'None':0,'Y':1,'N':0,'Reg':3,'IR1':2,'IR2':1,'IR3':0,"None" : 0,
            "No" : 2, "Mn" : 2, "Av": 3,"Gd" : 4,"Unf" : 1, "LwQ": 2, "Rec" : 3,"BLQ" : 4, "ALQ" : 5, "GLQ" : 6
            ,"P" : 1, "Y" : 2}
# ExterQual, ExterCond,
#BsmtQual BsmtCond  BsmtExposure 
# BsmtFinType1 BsmtFinType2 
# HeatingQC #KitchenQual
# FireplaceQu 
#GarageQual GarageCond PavedDrive 

#mapping ordinal variable to number 
house_data['ExterQual'] = house_data['ExterQual'].map(value_trans)
house_data['ExterCond'] = house_data['ExterCond'].map(value_trans)
house_data['BsmtQual'] = house_data['BsmtQual'].map(value_trans)
house_data['BsmtCond'] = house_data['BsmtCond'].map(value_trans)
house_data['BsmtExposure'] = house_data['BsmtExposure'].map(value_trans)
house_data['BsmtFinType1'] = house_data['BsmtFinType1'].map(value_trans)
house_data['BsmtFinType2'] = house_data['BsmtFinType2'].map(value_trans)
house_data['HeatingQC'] = house_data['HeatingQC'].map(value_trans)
house_data['KitchenQual'] = house_data['KitchenQual'].map(value_trans)
house_data['FireplaceQu'] = house_data['FireplaceQu'].map(value_trans)
house_data['GarageQual'] = house_data['GarageQual'].map(value_trans)
house_data['GarageCond'] = house_data['GarageCond'].map(value_trans)
house_data['PavedDrive'] = house_data['PavedDrive'].map(value_trans)


#check again the missing vlaue 
house_data.isnull().sum().sort_values(ascending=False)[0:20]

#combine some variables that describe the same features
#exterior overall =(ExterQual+ExterCond)/2
house_data['Exterior']=(house_data['ExterQual']+house_data['ExterCond'])/2
#basement overall= ()/5
house_data['Basement']=(house_data['BsmtQual']+house_data['BsmtCond']+house_data['BsmtExposure']+house_data['BsmtFinType1']+house_data['BsmtFinType2'])/5
# 1stFlrSF: First Floor square feet, 2ndFlrSF: Second floor square feet,TotalBsmtSF
#total square feet 
house_data['Total_SF']=house_data['1stFlrSF']+house_data['2ndFlrSF']+house_data['TotalBsmtSF']
# BsmtFullBath: Basement full bathrooms,BsmtHalfBath: Basement half bathrooms,FullBath: Full bathrooms above grade,HalfBath: Half baths above grade
house_data['Total_bath']=(house_data['BsmtFullBath']+0.5*house_data['BsmtHalfBath']+house_data['FullBath']+0.5*house_data['HalfBath'])
#OpenPorchSF: Open porch area in square feet,EnclosedPorch: Enclosed porch area in square feet,3SsnPorch: Three season porch area in square feet
#ScreenPorch: Screen porch area in square feet
house_data['Total_Porch']=house_data['OpenPorchSF']+house_data['EnclosedPorch']+house_data['3SsnPorch']+house_data['3SsnPorch']


#drop redundant variables above
house_data = house_data.drop(['ExterQual','ExterCond','BsmtQual','BsmtCond',
                                'BsmtExposure','BsmtFinType1','BsmtFinType2','1stFlrSF','2ndFlrSF',
                               'TotalBsmtSF','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
                               'OpenPorchSF','EnclosedPorch','3SsnPorch','3SsnPorch'],axis=1)

#fill other NA
house_data['Total_bath'] = house_data['Total_bath'].fillna(house_data["Total_bath"].median())
house_data['KitchenQual'] = house_data['KitchenQual'].fillna(house_data["KitchenQual"].median())
house_data['GarageArea'] = house_data['GarageArea'].fillna(house_data["GarageArea"].median())
house_data['GarageCars'] = house_data['GarageCars'].fillna(house_data["GarageCars"].median())
house_data['Total_SF'] = house_data['Total_SF'].fillna(house_data["Total_SF"].median())
house_data['BsmtUnfSF'] = house_data['BsmtUnfSF'].fillna(house_data["BsmtUnfSF"].median())
house_data['BsmtFinSF2'] = house_data['BsmtFinSF2'].fillna(house_data["BsmtFinSF2"].median())
house_data['BsmtFinSF1'] = house_data['BsmtFinSF1'].fillna(house_data["BsmtFinSF1"].median())

#general information about dataset
plt.hist(house_data['SalePrice'],60)
plt.show()
# sale price is right skew, so i decided to a log-transformation to normailzed the data
plt.hist(np.log(house_data['SalePrice']),60)
plt.show()
# use normalized sale peice as the price for model building and prediction
house_data['log_SalePrice']=np.log(house_data['SalePrice'])

#look at the correlation between all the numberic variables
corr_matrix = house_data.corr()
f, ax = plt.subplots(figsize=(35,24))
cmap = sn.diverging_palette(220, 10, as_cmap=True)
mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))
sn.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot = True)

#Encode categorical features
house_data = pd.get_dummies(house_data).reset_index(drop=True)
house_data.head()

#check the missing value again
house_data.isnull().sum().sort_values(ascending=False)[0:5]

#split the unknown sale price data with the know training set
house_train=house_data[:1460]
house_test=house_data[1460:]
#split the data for training and testing
Y= house_train['log_SalePrice']
X = house_train.drop(['SalePrice','log_SalePrice'],axis=1)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,shuffle=False)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

#Multiple Linear Regression model 
mlr_data=x_train[['OverallQual','GrLivArea','GarageCars','Total_SF','Total_bath']]
mlr_test=x_test[['OverallQual','GrLivArea','GarageCars','Total_SF','Total_bath']]
model_Mlr = LinearRegression().fit(mlr_data, y_train)
predict_Mlr = model_Mlr.predict(mlr_test)
print('Root Mean Square Error = ' + str(math.sqrt(mean_squared_error(y_test, predict_Mlr))))
print("R^2 Score: "+str(model_Mlr.score(mlr_test, y_test)))

# summarize feature importance
feature_importances = pd.DataFrame(model_Mlr.coef_,index = mlr_data.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances

# Extreme Gradient Boosting
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=3500,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=5)

# Light Gradient Boosting
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=6000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=5)

# Ridge regression
kfolds = KFold(n_splits=10, shuffle=True, random_state=5)
ridge_alphas = [1e-15, 1e-10, 1e-8,0.0005,0.0001, 0.001, 0.05, 0.01, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=Kf))

# Random Forest 
rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=5)

# Gradient Boosting regression 
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=5)  
#stack model 
stack = StackingCVRegressor(regressors=(xgboost, lightgbm, ridge,rf,gbr),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)

# fit every model
xgboost_model=xgboost.fit(x_train, y_train)
predict_xgboost = xgboost_model.predict(x_test)
print('xgboost_model:Root Mean Square Error = ' + str(math.sqrt(mean_squared_error(y_test, predict_xgboost))))

lightgbm_model=lightgbm.fit(x_train,y_train)
predict_lightgbm = lightgbm_model.predict(x_test)
print('lightgbm_model:Root Mean Square Error = ' + str(math.sqrt(mean_squared_error(y_test, predict_lightgbm))))

ridge_model=ridge.fit(x_train,y_train)
predict_ridge = ridge_model.predict(x_test)
print('ridge_model: Root Mean Square Error = ' + str(math.sqrt(mean_squared_error(y_test, predict_ridge))))

rf_model=rf.fit(x_train,y_train)
predict_rf = rf_model.predict(x_test)
print('rf_model:Root Mean Square Error = ' + str(math.sqrt(mean_squared_error(y_test, predict_rf))))

gbr_model=gbr.fit(x_train,y_train)
predict_gbr = gbr_model.predict(x_test)
print('gbr_model: Root Mean Square Error = ' + str(math.sqrt(mean_squared_error(y_test, predict_gbr))))

stack_model = stack.fit(np.array(x_train), np.array(y_train))
predict_stack = stack_model.predict(np.array(x_test))
print('stack model: Root Mean Square Error = ' + str(math.sqrt(mean_squared_error(y_test, predict_stack))))

#important feature XGBoost Regression
feature_importances_xgbost = pd.DataFrame(xgboost_model.feature_importances_,index = x_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_xgbost[:10]
#importantce feature of light Gradient Boosting
feature_importances_lightgbm = pd.DataFrame(lightgbm_model.feature_importances_,index = x_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_lightgbm[:10]
#importantce feature of random forest rf_model
feature_importances_rf= pd.DataFrame(rf_model.feature_importances_,index = x_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_rf[:10]
#importantce feature of Gradient Boosting
feature_importances_gbr = pd.DataFrame(gbr_model.feature_importances_,index = x_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_gbr[:10]
#submission for the kaggle
house_t=pd.read_csv("test.csv")
predict_s = stack_model.predict(np.array(house_test))
submission = pd.DataFrame({
        "Id": house_t["Id"],
        "SalePrice": np.exp(predict_s)
    })
submission.to_csv('submission.csv', index=False)

