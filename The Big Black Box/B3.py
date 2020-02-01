#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import lightgbm as lgbm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc, mean_absolute_error, f1_score
from sklearn.tree import DecisionTreeRegressor as dt
import datetime
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import random
import category_encoders as ce
import math
import statistics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ## Encoder Wrapper 
# #### (Inspired from the work of Will McGinnis on category_encoders - https://github.com/wdm0006)

# In[2]:


def encode_all(df,dfv,dfk,encoder_to_use,target_col,handle_missing='return_nan'):
    
    encoders_used = {}
    
    for col in encoder_to_use:

        if encoder_to_use[col]=='BackwardDifferenceEncoder':
            encoder=ce.BackwardDifferenceEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing)
            encoder.fit(X=df,y=df[target_col])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='BaseNEncoder':
            encoder=ce.BaseNEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing,base=3) 
            encoder.fit(X=df,y=df[target_col])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='BinaryEncoder':
            encoder=ce.BinaryEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing)
            encoder.fit(X=df,y=df[target_col])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='CatBoostEncoder':
            encoder=ce.CatBoostEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing,sigma=None,a=2)
            encoder.fit(X=df,y=df[target_col])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

    #     if encoder_to_use[col]=='HashingEncoder':
    #         encoder=ce.HashingEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing)
    #         encoder.fit(X=df,y=df[target_col])
    #         df=encoder.transform(df)
    #         encoders_used[col]=encoder

        if encoder_to_use[col]=='HelmertEncoder':
            encoder=ce.HelmertEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing)
            encoder.fit(X=df,y=df[target_col])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='JamesSteinEncoder':
            encoder=ce.JamesSteinEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing, model='binary')
            encoder.fit(X=df,y=df[target_col])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='LeaveOneOutEncoder':
            encoder=ce.LeaveOneOutEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing,sigma=None)
            encoder.fit(X=df,y=df[target_col])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='MEstimateEncoder':
            encoder=ce.MEstimateEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing,randomized=True,sigma=None,m=2)
            encoder.fit(X=df,y=df[target_col])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='OneHotEncoder':
            encoder=ce.OneHotEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing,use_cat_names=True)
            encoder.fit(X=df,y=df[target_col])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='OrdinalEncoder':
            encoder=ce.OrdinalEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing)
            encoder.fit(X=df,y=df[target_col])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='SumEncoder':
            encoder=ce.SumEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing)
            encoder.fit(X=df,y=df[target_col])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='PolynomialEncoder':
            encoder=ce.PolynomialEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing)
            encoder.fit(X=df,y=df[target_col])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder

        if encoder_to_use[col]=='TargetEncoder':
            encoder=ce.TargetEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing,min_samples_leaf=10, smoothing=5)
            encoder.fit(X=df,y=df[target_col])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder


        if encoder_to_use[col]=='WOEEncoder':
            encoder=ce.WOEEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing=handle_missing,randomized=True,sigma=None)
            encoder.fit(X=df,y=df[target_col])
            df=encoder.transform(df)
            dfv=encoder.transform(dfv)
            dfk=encoder.transform(dfk)
            encoders_used[col]=encoder
            
#         print("Encoding done for - ",col)
    
    print("Completed encoder - ",datetime.datetime.now())
    
    return df, dfv, dfk, encoders_used


# ### Imputer Function

# In[3]:


def imputer(df, dfv, dfk, target_col, imputer_dict):
    
    result = {}
    
    for i in imputer_dict:
        
        if imputer_dict[i]['Indicator'] == 'deleterows':
            if df[i].isna().sum() > 0:
                df = df[df[i].isfinite()]
                dfv = dfv[dfv[i].isfinite()]
                dfk = dfk[dfk[i].isfinite()]
            
        if imputer_dict[i]['Indicator'] == True:
            if df[i].isna().sum() > 0:
                df[i+'_null_ind'] = np.where(df[i].isna(),1,0)
                dfv[i+'_null_ind'] = np.where(dfv[i].isna(),1,0)
                dfk[i+'_null_ind'] = np.where(dfk[i].isna(),1,0)
        
        if imputer_dict[i]['mvi'] in ['mean','median','most_frequent']:
            imp = SimpleImputer(missing_values = np.nan
                                , strategy = imputer_dict[i]['mvi']
                                , verbose = True
                                , add_indicator = False
                                , fill_value = None
                               )
            imp.fit(df[[i]])
            result[i] = imp
            df.loc[:,i] = result[i].transform(df[[i]])
            dfv.loc[:,i] = result[i].transform(dfv[[i]])
            dfk.loc[:,i] = result[i].transform(dfk[[i]])
        
        if imputer_dict[i]['mvi'] == 'far_val':
            result[i] = df[i].max()*100
            df[i] = np.where(df[i].isna(),result[i],df[i])
            dfv[i] = np.where(dfv[i].isna(),result[i],dfv[i])
            dfk[i] = np.where(dfk[i].isna(),result[i],dfk[i])
        
        
    ##### interativeimputer (if none of the above then this) ######
    
    imp = IterativeImputer(
                        max_iter = 3
                       , estimator = ExtraTreesRegressor() #### hyperparameter, alternatively beysian, knn etc.
                       , n_nearest_features = 5 ##### Change value for maximum columns considered to predict missing value
                      )
                      
    dfvc = dfv.copy()
    dfv[target_col] = np.nan
    
    dfkc = dfk.copy()
    dfk[target_col] = np.nan
    
    dfcolumns = df.columns
    imp.fit(df)
    df = pd.DataFrame(imp.transform(df))
    df.columns = dfcolumns
    dfv = pd.DataFrame(imp.transform(dfv))
    dfv.columns = dfcolumns
    dfk = pd.DataFrame(imp.transform(dfk))
    dfk.columns = dfcolumns
    
    dfv[target_col] = np.array(dfvc[target_col])
    dfk[target_col] = np.nan
    
    for i in imputer_dict:
        if imputer_dict[i]['mvi'] == 'iterativeimputer':
            result[i] = imp
    
    print("Completed imputer - ",datetime.datetime.now())
    
    return df, dfv, dfk, result


# ## Encoder Combinations
# Using random selection for encoder against each column. Final output is a dictionary with key as column name and encoder name as the value along with a few other important features for that column such as null handling.
# Note: The number of columns in the dataset does not matter as the encoders selected are done dynamically using dynamic codes.

# In[4]:


def make_encoder_combinations(df, n_samples):
    
    n_samples = str(n_samples)
    
    #### Creating categorical column List
    columns = []
    for i in df:
        if df[i].dtype.kind not in 'bifucM':
            columns.append(i)

    #### Finding dsitinct categories within a column
    d_cnt = {}
    for i in columns:
        d_cnt[i] = df[i].nunique()

    #### List of encoders to be selected from
    encoders = ['BackwardDifferenceEncoder'
                , 'BinaryEncoder'
                # , 'HashingEncoder'
                , 'HelmertEncoder'
#                 , 'JamesSteinEncoder'
                , 'LeaveOneOutEncoder'
                , 'MEstimateEncoder'
                , 'OneHotEncoder'
                , 'BaseNEncoder'
                , 'CatBoostEncoder'
                , 'OrdinalEncoder'
                , 'SumEncoder'
                , 'PolynomialEncoder'
                , 'TargetEncoder'
                # , 'WOEEncoder' #### ignoring for regression problem. Can be used in classification
               ]

    #### List of encoders
    target_encoders = [
                        'BaseNEncoder'
                        , 'BinaryEncoder'
                        , 'CatBoostEncoder'
                        , 'LeaveOneOutEncoder'
                        , 'MEstimateEncoder'
                        , 'OrdinalEncoder'
                        , 'TargetEncoder'
                        , 'WOEEncoder'
                       ]

    #### Creating code strings
    s, s1, s2, s4 = '', '', '', ''

    for i in range(len(columns)):
        s1 = s1 + "sam"+str(i)+" = np.char.mod('%d', np.random.uniform(low = 0, high = len(encoders), size = "+n_samples+"))" + "\n"
        s2 = s2 + "sam"+str(i) +"[i] +"
        s4 = s4 + "d[columns["+str(i)+"]] = encoders[int(i["+str(i)+"])]" + "\n"

    s2 = s2[:-1]
    exec(s1)
    val = []
    s2 = "for i in range("+n_samples+"): val.append("+s2+")"
    exec(s2)

    #### Run dynamic code
    l_encs = list(np.unique(np.array(val)))
    encs_final_dicts = []
    for i in l_encs:
        d = {}
        exec(s4)
        encs_final_dicts.append(d)

    #### Filtering out encoder combinations resulting in large number of features
    cardinality_check = 8 #### can be used as a hyperparameter
    cnt = 0
    encs_final_dicts_rm = []
    for i in encs_final_dicts:
        flag = 0
        for j in i: 
            if d_cnt[j] > cardinality_check and i[j] not in target_encoders:
                flag = 1
        if flag == 1:
            encs_final_dicts_rm.append(cnt)
        cnt += 1

    encs_final_dicts2 = []
    for i in range(len(encs_final_dicts)):
        if i not in encs_final_dicts_rm:
            encs_final_dicts2.append(encs_final_dicts[i])
    
    print("Made encoder combinations - ",datetime.datetime.now())
    
    return encs_final_dicts2


# ## Imputer Combinations

# In[30]:


def make_imputer_combinations(df, target_col, n_samples = 100):

    n_samples = str(n_samples) #### hyperparameter
    
    #### Creating column list
    columns = []
    for i in df:
        if i != target_col:
            columns.append(i)
    
    #### Listing available imputers
    imputers = ['iterativeimputer', 'median', 'mean','most_frequent','deleterows']
    Indicators = [False,True]

    #### Creating dynamic python strings
    s, s1, s2, s4 = '', '', '', ''
        
    for i in range(len(columns)):
        s1 = s1 + "sam"+str(i)+" = np.char.mod('%d', np.random.uniform(low = 0, high = len(imputers), size = "+n_samples+"))" + "\n"
        s2 = s2 + "sam"+str(i) +"[i] +"
        s4 = s4 + "d[columns["+str(i)+"]] = {'mvi':imputers[int(i["+str(i)+"])], 'Indicator':Indicators[int(i["+str(len(columns))+"])]}" + "\n"

    #### Running dynamic strigs
    s2 = s2[:-1]
    exec(s1)
    val = []
    s2 = "for i in range("+n_samples+"): val.append("+s2+")"
    exec(s2)
    
    l_imps = list(np.unique(np.array(val)))
    
    #### Setting value for the true-false indicator for the imputer null handling
    l_imps2 = []
    for i in l_imps:
        for j in [0]: ### range(1): ### change for true and false
            l_imps2.append(i + str(j))
            
    imps_final_dicts = []
    for i in l_imps2:
        d = {}
        exec(s4)
        imps_final_dicts.append(d)
    
    print("Completed making imputer combinations - ",datetime.datetime.now())
    
    return imps_final_dicts


# #### Function for unique values of list

# In[6]:


def unique(list1): 
    
    # intilize a null list 
    unique_list = []       
    
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list


# ## Creating function to reduce cardinality

# In[7]:


def cat_cleaner(df, dfkaggle, cat_col, target_col, stretch = 0.8):
    
    #### Filtering out nulls
    dfx = df[df[cat_col].notnull()]
    
    #### Subsetting only cat_col and the target column to reduce data
    dfx = dfx[[cat_col,target_col]]

    #### lower case for categories cleaning
    dfx[cat_col] = dfx[cat_col].str.lower()

    #### identifying special characters
    chars = []
    for i in list(dfx[cat_col]):
        for j in i:
            if (ord(j) in list(range(65,91))) or (ord(j) in list(range(97,123))):
                chars = chars
            else:
                chars.append(j)

    #### Cleaning special characters
    for i in unique(chars):
        if i != ' ':
            dfx[cat_col] = dfx[cat_col].str.replace(i,' ')

    #### Splitting the categories by spaces and creating a newer dataframe
    df_prof = dfx[cat_col].str.split(' ', 10, expand=True)
    df_prof[target_col] = dfx[target_col]

    #### Creating list of unique words
    l = []
    val = []
    for i in df_prof.columns[:-1]:
        df_prof2 = df_prof[df_prof[i].notnull()]
        m = df_prof2.iloc[:,i].tolist()
        n = df_prof2[target_col].tolist()
        for j in m:
            l.append(j)
        for k in n:
            val.append(k)

    #### Creating frequency matrix to find most common words
    df_prof3 = pd.DataFrame({'Words':l,target_col:val})
    df_prof4 = df_prof3.groupby('Words',as_index = False).agg({target_col:['mean','var','count']})
    df_prof4.columns = df_prof4.columns.droplevel(level=0)
    df_prof4['var_sum'] = df_prof4['var'] * df_prof4['count']
    df_prof4 = df_prof4.drop(columns = ['count'])
    df_prof4.columns = ['Words','mean','var','count']

    #### Removing other insignificant terms in the categories
    df_prof4 = df_prof4[df_prof4['Words'] != ' ']
    df_prof4 = df_prof4[df_prof4['Words'] != '']
    df_prof4 = df_prof4[df_prof4['Words'] != 'and']

    #### top percentile through stretch parameter
    df_prof5 = df_prof4.copy()
    df_prof5['total_word_count'] = df_prof5['count'].sum()
    df_prof5 = df_prof5.sort_values(by = 'count', ascending = False)
    df_prof5['cnt_cumsum'] = df_prof5['count'].cumsum()
    df_prof5['cum_perc'] = df_prof5['cnt_cumsum']/df_prof5['total_word_count']
    df_prof5 = df_prof5[df_prof5['cum_perc'] <= stretch]
    
    #### Creating a list of significant words
    sig_words = list(df_prof5['Words'])

    #### Creating columns for each term
    for i in sig_words:
        df['is_'+i] = np.where(df[cat_col].str.contains(i),1,0)
        dfkaggle['is_'+i] = np.where(dfkaggle[cat_col].str.contains(i),1,0)
    
    return df, dfkaggle


# ### Train test split function outputting dataframe instead of series objects

# In[8]:


def split_val(df, fraction = 0.2):
    df3 = df.copy()
    df3 = df3.reset_index().drop(columns = 'index')
    s = np.random.uniform(low = 0, high = df3.shape[0]-1, size = round(df3.shape[0]*fraction))
    s = s.round()
    dfv = df3.iloc[s,:]
    dft = df3.drop(s)
    print("Completed split val - ",datetime.datetime.now())
    return dfv, dft


# ### Data scaler function to output dataframe rather than series objects

# In[9]:


def scale(dft,dfv,dfk,target_col):
    
    cols = dft4.drop(columns = target_col).columns
    scaler = StandardScaler()
    scaler.fit(dft.drop(columns = target_col))
    
    dftc = dft.copy()
    dft = pd.DataFrame(scaler.transform(dft.drop(columns = target_col)))
    dft.columns = cols
    dft[target_col] = np.array(dftc[target_col])
    
    dfvc = dfv.copy()
    dfv = pd.DataFrame(scaler.transform(dfv.drop(columns = target_col)))
    dfv.columns = cols
    dfv[target_col] = np.array(dfvc[target_col])
    
    dfkc = dfk.copy()
    dfk = pd.DataFrame(scaler.transform(dfk.drop(columns = target_col)))
    dfk.columns = cols
    dfk[target_col] = np.array(dfkc[target_col])
    
    print("Completed Scaling - ",datetime.datetime.now())
    return dft, dfv, dfk


# ### Outliar detection function
# Using PyOD

# In[10]:


def remove_outliars(dft, target_col):
    
    ol_model = IForest() #### can be used as a hyperparameter
    ol_model.fit(dft.drop(columns = target_col))
    dft['is_outliar'] = ol_model.labels_
    dft = dft[dft['is_outliar'] != 1]
    dft = dft.drop(columns = 'is_outliar')
    print("Completed Outliar Detection - ",datetime.datetime.now())
    
    return dft


# ### The model function wrapping 3 separate techniques
# 1. XG Boost
# 2. LightGBM
# 3. CatBoost
# Choice of model is also a hyperparameter used by the process to find the best algorithm with a group of encoders and imputers

# In[47]:


def model_and_predict(df4, df_val, dfk, target_col):
    
    X_train, X_test, Y_train, Y_test = train_test_split(df4.drop(columns = [target_col])
                                                        , df4[target_col], test_size = 0.20 ##### can be used as a hyperparameter
                                                       )

    rand = random.choice(['XGB','LGB', 'Catboost'])
    print("Chosen Algorithm : ",rand)
    
#################################### XGBoost
    
    if rand == 'XGB':
    
        bst = xgb.XGBRegressor(
                                base_score=0.5
                                , colsample_bylevel=1
                                , colsample_bytree=1
                                , gamma=0
                                , learning_rate=0.1
                                , max_delta_step=0
                                , max_depth = 10
                                , min_child_weight=1
                                , missing = None
                                , n_estimators = 500
                                , nthread=-1
                                , objective = 'reg:squarederror'
                                , eval_metric = 'mae'
                                , reg_alpha = 0
                                , reg_lambda = 1
                                , scale_pos_weight = 1
                                # , seed = 0
                                , silent = False
                                , subsample=1
                                , verbose = 1
                                # , tree_method = 'gpu_hist' # to be used if you have a CUDA supported GPU
                                # , gpu_id = 0 # to be used if you have a CUDA supported GPU
                                )

        bst.fit(X_train, Y_train, eval_set =[(X_train, Y_train), (X_test, Y_test)], eval_metric='mae', verbose = True)

#################################### XGBoost

#################################### LightGBM

    elif rand == 'LGB':
        bst = lgbm.LGBMRegressor(
                                  boosting_type='gbdt'
                                    , max_depth=9
                                    , learning_rate=0.05
                                    , n_estimators=500
                                    , objective='regression'
                                    , min_split_gain=0.0
                                    , min_child_weight=0.001
                                    , min_child_samples=20
                                    , num_leaves = 150
                                    , n_jobs=-1
                                    , silent=True
                                    , importance_type='split'
                                    # , device = 'gpu' # to be used if you have a CUDA supported GPU
                                    # , gpu_platform_id = 0 # to be used if you have a CUDA supported GPU
                                    # , gpu_device_id = 0 # to be used if you have a CUDA supported GPU
                                )
                                
        bst.fit(X_train, Y_train, eval_set =(X_test, Y_test), eval_metric='mae', verbose = True)
        
#################################### CatBoost
    
    elif rand == 'Catboost':
        
        bst = CatBoostRegressor(eval_metric='MAE'
                                , use_best_model=True
                                , metric_period = 100
                                , depth = 9
                                # , task_type = "GPU" # to be used if you have a CUDA supported GPU
                                # , devices = '0:1' # to be used if you have a CUDA supported GPU
                                , num_boost_round = 500
                               )
                                
        bst.fit(X_train,Y_train,eval_set=(X_test,Y_test), verbose = True)

#################################### CatBoost
    
#################################### 10 fold Crossvalidation

    results = []

    for i in range(10):
        df_val2 = shuffle(df_val)
        df_val3 = df_val2[0:int(df_val2.shape[0]*0.3)]
        rkf = bst.predict(df_val3.drop(columns = [target_col]))
        results.append(mean_absolute_error(df_val3[target_col], rkf))
        
#################################### 10 fold Crossvalidation
    
    r = bst.predict(df_val.drop(columns = [target_col]))
    acc_score = mean_absolute_error(df_val[target_col], r)

    rk = bst.predict(dfk.drop(columns = [target_col]))
    dfk[target_col] = rk
    
    print("Completed Modelling - ",datetime.datetime.now())
    
    return acc_score, X_train.columns, bst, dfk, results, rand


# ### Reader Function
# Must output a training(known dependent vbariable values) and a test(unknown dependent vbariable values) data. 
# 
# This function is meant for the user to make any specific changes before feedind the data for regression.

# In[12]:


def read_data(train_data_path, test_data_path):
    
    df = pd.read_csv(train_data_path
                     , na_values = ['nA','#N/A','#NUM!']
                    )
    
    drop_cols = ['Instance','Hair Color','Body Height [cm]','Wears Glasses'] # ,'Crime Level in the City of Employement']
    df['Yearly Income in addition to Salary (e.g. Rental Income)'] = df['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace(" EUR","").astype(float)
    df = df.drop(columns = drop_cols)
    
    dfkaggle = pd.read_csv(test_data_path
                     , na_values = ['nA','#N/A','#NUM!']
                    )
    
    dfkaggle['Yearly Income in addition to Salary (e.g. Rental Income)'] = dfkaggle['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace(" EUR","").astype(float)
    dfkaggle = dfkaggle.drop(columns = drop_cols)

    df = shuffle(df)
    df = df[0:min(df.shape[0],10000)]
    
    df2 = df.copy()
    dfkaggle2 = dfkaggle.copy()
    
    
    clean_cols = ['Gender','Country','Housing Situation','University Degree'] #, 'Crime Level in the City of Employement']
    
    for i in clean_cols:
        df2[i] = np.where(df2[i] == '0', np.nan, df2[i])
        dfkaggle2[i] = np.where(dfkaggle2[i] == '0', np.nan, dfkaggle2[i])
        
    df2.columns = [i.replace("]","").replace("[","").replace(" ","_").replace(")","").replace("(","") for i in df2.columns]
    dfkaggle2.columns = df2.columns
    
    df2['Profession2'] = df2['Profession'].str.slice(0,5)
    dfkaggle2['Profession2'] = dfkaggle2['Profession'].str.slice(0,5)
    
#     df2['Total_Yearly_Income_EUR'] = np.log(df2['Total_Yearly_Income_EUR'])
    
    for i in df2:
        if df2[i].dtype.kind not in 'bifuc':
            df2[i] = df2[i].str.lower()
            df2[i] = df2[i].str.strip()
            dfkaggle2[i] = dfkaggle2[i].str.lower()
            dfkaggle2[i] = dfkaggle2[i].str.strip()

    df2['is_senior'] = np.where(df2['Profession'].str.contains("senior"),1,0)
    dfkaggle2['is_senior'] = np.where(dfkaggle2['Profession'].str.contains("senior"),1,0)
 
    df2['is_manager'] = np.where(df2['Profession'].str.contains("manager"),1,0)
    dfkaggle2['is_manager'] = np.where(dfkaggle2['Profession'].str.contains("manager"),1,0)

    l1 = list(df2.dropna(axis = 1, how = 'all').columns)
    l2 = list(dfkaggle2.dropna(axis = 1, how = 'all').columns)
    l = [i for i in l1 if i in l2]
    l.append('Total_Yearly_Income_EUR')
    
    print("Completed Reader - ",datetime.datetime.now())
    return df2[l], dfkaggle2[l]


# ### The hustle
# We are ready to use all the functions we created to get random combinations of encoders and imputers to start giving the result in terms of validation errors

# In[45]:


def big_black_box(min_cv
                   , target_col
                   , train_data_path
                   , test_data_path
                   , model_summary_file = "model results.csv" # path for final summary file
                   , trials = 100 # Number of experiments with encoder and imputer combinations
                 ):
    cnt = 0
    r = []

    df, df_kaggle = read_data(train_data_path = train_data_path, test_data_path = test_data_path)

    for i in range(1):    

        seed = random.randint(0,1000)
        print("Seed: ",seed)

        np.random.seed(seed)

        dfv2, dft2 = split_val(df)

        encs_final_dicts = make_encoder_combinations(dft2, n_samples = trials*100)

        for i in encs_final_dicts: ### unlist and take all elements for full run

            dft3, dfv3, dfkaggle3, encoders_used = encode_all(df = dft2, dfv = dfv2, dfk = df_kaggle, target_col = target_col, encoder_to_use = i)

            dft3.dropna(axis=1, how='all', inplace = True)
            dfv3 = dfv3[dft3.columns]
            dfkaggle3 = dfkaggle3[dft3.columns]
            dfc = dfv3.copy()

            imps_final_dict = make_imputer_combinations(dft3, n_samples = 2, target_col = target_col)
            for j in imps_final_dict: ### unlist and take all elements for full run
                
                cnt += 1
                
                if cnt > trials:
                    print('Summary File has been saved to: ',model_summary_file)
                    break
                    
                if dfv3[dfv3[target_col].isna()].shape[0] > 0:
                    dfv3 = dfc.copy()

                dft4, dfv4, dfkaggle4, result = imputer(df = dft3, dfv = dfv3, dfk = dfkaggle3, target_col = target_col, imputer_dict = j)

                acc_score, X_train_columns, bst, dfk, cv, rand = model_and_predict(dft4, dfv4, dfkaggle4, target_col = target_col)

                if statistics.mean(cv) < min_cv: 
                    min_cv = statistics.mean(cv)
                    dfk.to_csv("kaggle submission data.csv")

                rl = [i,j,acc_score, rand, statistics.mean(cv), statistics.variance(cv), cv, datetime.datetime.now(), ''] ## bst.get_booster().get_score(importance_type="cover")

                r.append(rl)
                dfr = pd.DataFrame(r)
                dfr.columns = ['encoder_combination', 'imputer combination', 'validation_mae', 'model', 'cv_val_mean_mae', 'cv_variance', 'cv_values', 'time', 'comment']
                dfr = dfr[['encoder_combination', 'imputer combination', 'validation_mae', 'model', 'cv_val_mean_mae', 'cv_variance', 'cv_values', 'time', 'comment']]
                dfr.to_csv(model_summary_file, index = False)
            
            if cnt > trials:
                break

# In[46]:


big_black_box(min_cv = 150000 # any large value as per your RMSE
                   , target_col = 'Total_Yearly_Income_EUR' # to be changed by the user for 
                   , train_data_path = "tcd-ml-1920-group-income-train.csv" # known target
                   , test_data_path = "tcd-ml-1920-group-income-test.csv" # unknown target
                   , model_summary_file = "model results.csv" # path for final summary file
                   , trials = 100 # Number of experiments with encoder and imputer combinations
                 )

