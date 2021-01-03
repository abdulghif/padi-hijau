import pandas as pd
import numpy as np
import pandas_profiling

# library untuk eksplorasi data
import matplotlib.pyplot as plt
import seaborn as sns

# library untuk membagi data
from sklearn.model_selection import train_test_split

# library untuk evaluasi hasil prediksi
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score

# library untuk bebrapa model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

from sklearn.feature_selection import SelectFromModel

# library untuk tunning parameter
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# library untuk menggabungkan beberapa algoritme
from imblearn.pipeline import Pipeline
# library untuk menyeimbangkan data
from imblearn.over_sampling import SMOTE

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
import category_encoders as ce

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.calibration import CalibratedClassifierCV

df_train = pd.read_csv('bri-data-hackathon-people-analytic/train.csv')
df_test = pd.read_csv('bri-data-hackathon-people-analytic/test.csv')

def Pipelining(df_train_prep1,df_test_prep1):
    ################################ Preprocess and Data Cleaning ############################

    def achievement_target1(x):
        if x == 'Pencapaian 50%-100%':
            return 'achiev_50%-100%'
        elif x == 'Pencapaian 100%-150%':
            return 'achiev_100%-150%'
        elif x == 'Pencapaian < 50%':
            return 'achiev_< 50%'
        elif x == 'Pencapaian > 1.5':
            return 'achiev_> 1.5'
        else:
            return x

    def achievement_target2(x):
        if x == 'Pencapaian 50%-100%':
            return 'achiev_50%-100%'
        elif x == 'Pencapaian 100%-150%':
            return 'achiev_100%-150%'
        elif x == 'Pencapaian < 50%':
            return 'achiev_< 50%'
        elif x == 'Pencapaian > 1.5':
            return 'achiev_> 1.5'
        else:
            return x

    def achievement_target3(x):
        if x == 'not reached':
            return 'not_reached'
        else:
            return x

    year_grad = []

    x = 1960

    while(x < 2020):
        year_grad.append(str(x))
        x += 1

    # PREPROCESS VARIABEL    

    df_train_prep1['achievement_target_1'] = df_train_prep1['achievement_target_1'].apply(achievement_target1)
    df_train_prep1['achievement_target_2'] = df_train_prep1['achievement_target_2'].apply(achievement_target2)
    df_train_prep1['achievement_target_3'] = df_train_prep1['achievement_target_3'].apply(achievement_target3)

    df_train_prep1['Achievement_above_100%_during3quartal'].fillna(0, inplace = True)

    df_train_prep1['year_graduated'] = np.where(df_train_prep1['year_graduated'].isin(year_grad),df_train_prep1['year_graduated'],3000)
    df_train_prep1['year_graduated'] = df_train_prep1['year_graduated'].astype(int)

    df_train_prep1['Education_level'] = np.where(df_train_prep1['Education_level'].isnull() & df_train_prep1['GPA'].isnull(),'other b',df_train_prep1['Education_level'])
    df_train_prep1['Education_level'] = np.where(df_train_prep1['Education_level'].isnull() & (df_train_prep1['GPA'] == 0),'other b',df_train_prep1['Education_level'])
    df_train_prep1['Education_level'] = np.where(df_train_prep1['Education_level'].isnull(),'other a',df_train_prep1['Education_level'])

    df_train_prep1['GPA'] = np.where(df_train_prep1['GPA']>100,df_train_prep1['GPA']/100,df_train_prep1['GPA'])
    df_train_prep1['GPA'] = np.where(df_train_prep1['Education_level'].isin(['level_0','level_1']),0,df_train_prep1['GPA'])
    df_train_prep1['GPA'] = np.where(df_train_prep1['GPA'].isnull(),0,df_train_prep1['GPA'])
    df_train_prep1['GPA'] = np.where((df_train_prep1['GPA'] == 0)&(df_train_prep1['Education_level'].isin(['level_3','level_4','level_5'])),3,df_train_prep1['GPA'])

    df_train_prep1['job_duration_as_permanent_worker'].fillna(0, inplace = True)


    df_test_prep1['achievement_target_1'] = df_test['achievement_target_1'].apply(achievement_target1)
    df_test_prep1['achievement_target_2'] = df_test['achievement_target_2'].apply(achievement_target2)
    df_test_prep1['achievement_target_3'] = df_test['achievement_target_3'].apply(achievement_target3)

    df_test_prep1['Achievement_above_100%_during3quartal'].fillna(0, inplace = True)

    df_test_prep1['year_graduated'] = np.where(df_test['year_graduated'].isin(year_grad),df_test['year_graduated'],3000)
    df_test_prep1['year_graduated'] = df_test_prep1['year_graduated'].astype(int)

    df_test_prep1['Education_level'] = np.where(df_test['Education_level'].isnull() & df_test['GPA'].isnull(),'other b',df_test['Education_level'])
    df_test_prep1['Education_level'] = np.where(df_test_prep1['Education_level'].isnull() & (df_test_prep1['GPA'] == 0),'other b',df_test_prep1['Education_level'])
    df_test_prep1['Education_level'] = np.where(df_test_prep1['Education_level'].isnull(),'other a',df_test_prep1['Education_level'])

    df_test_prep1['GPA'] = np.where(df_test['GPA']>100,df_test['GPA']/100,df_test['GPA'])
    df_test_prep1['GPA'] = np.where(df_test_prep1['Education_level'].isin(['level_0','level_1']),0,df_test_prep1['GPA'])
    df_test_prep1['GPA'] = np.where(df_test_prep1['GPA'].isnull(),0,df_test_prep1['GPA'])
    df_test_prep1['GPA'] = np.where((df_test_prep1['GPA'] == 0)&(df_test_prep1['Education_level'].isin(['level_3','level_4','level_5'])),3,df_test_prep1['GPA'])

    df_test_prep1['job_duration_as_permanent_worker'].fillna(0, inplace = True)

    ################################ Feature Engineering ############################

    def age_2(x):
        if x <= 35:
            return 'ambis'
        elif x > 35:
            return 'non-ambis'

    def age_3(x):
        if x <= 25:
            return 1
        elif x <= 35:
            return 2
        elif x <= 45:
            return 3
        elif x <= 55:
            return 4
        elif x <= 65:
            return 5

    def year_graduated_2(x):
        if x >= 2015:
            return 'young'
        elif x < 2015:
            return 'old'

    def job_duration_2(x):
        if x < 6:
            return 'new'
    #     elif x < 24:
    #         return 'med'
        else:
            return 'old'

    # TOTAL LEAVE 

    df_train_prep1['total leave'] = df_train_prep1['annual leave'] + df_train_prep1['sick_leaves']
    df_test_prep1['total leave'] = df_test_prep1['annual leave'] + df_test_prep1['sick_leaves']

    # AGE

    df_train_prep1['age2'] = 2020 - df_train_prep1['age']
    df_train_prep1['age2'] = df_train_prep1['age2'].apply(age_2)

    df_test_prep1['age2'] = 2020 - df_test_prep1['age']
    df_test_prep1['age2'] = df_test_prep1['age2'].apply(age_2)

    df_train_prep1['age3'] = 2020 - df_train_prep1['age']
    df_train_prep1['age3'] = df_train_prep1['age3'].apply(age_3)

    df_test_prep1['age3'] = 2020 - df_test_prep1['age']
    df_test_prep1['age3'] = df_test_prep1['age3'].apply(age_3)

    # YEAR GRADUATED

    df_train_prep1['year_graduated2'] = df_train_prep1['year_graduated'].apply(year_graduated_2)
    df_test_prep1['year_graduated2'] = df_test_prep1['year_graduated'].apply(year_graduated_2)

    # JOB DURATION

    df_train_prep1['job_duration_as_permanent_worker2'] = df_train_prep1['job_duration_as_permanent_worker'].apply(job_duration_2)
    df_test_prep1['job_duration_as_permanent_worker2'] = df_test_prep1['job_duration_as_permanent_worker'].apply(job_duration_2)

    # TOTAL ROTATION

    df_train_prep1['total_rotation'] = df_train_prep1['branch_rotation'] + df_train_prep1['job_rotation'] + df_train_prep1['assign_of_otherposition']
    df_test_prep1['total_rotation'] = df_test_prep1['branch_rotation'] + df_test_prep1['job_rotation'] + df_test_prep1['assign_of_otherposition']

    # IPK O

    df_train_prep1['in study'] = np.where((df_train_prep1['GPA'] == 0)&(df_train_prep1['Education_level'].isin(['level_3','level_4','level_5'])),'study','finish')
    df_test_prep1['in study'] = np.where((df_test_prep1['GPA'] == 0)&(df_test_prep1['Education_level'].isin(['level_3','level_4','level_5'])),'study','finish')

    # LEVEL UP PERSON

    df_train_prep1['person level up'] = np.where(df_train_prep1['job_duration_in_current_job_level']>df_train_prep1['job_duration_in_current_person_level'],'up','stay')
    df_test_prep1['person level up'] = np.where(df_test_prep1['job_duration_in_current_job_level']>df_test_prep1['job_duration_in_current_person_level'],'up','stay')

    # TRAINING DURATION

    df_train_prep1['training - duration'] = df_train_prep1['job_duration_from_training'] - df_train_prep1['job_duration_as_permanent_worker']
    df_test_prep1['training - duration'] = df_test_prep1['job_duration_from_training'] - df_test_prep1['job_duration_as_permanent_worker']

    # NEW VAR FOR ONE HOT

    df_train_prep1['job_level_oh'] = df_train_prep1['job_level'] 
    df_train_prep1['person_level_oh'] = df_train_prep1['person_level']

    df_test_prep1['job_level_oh'] = df_test_prep1['job_level'] 
    df_test_prep1['person_level_oh'] = df_test_prep1['person_level']

    # FILTER

    df_train_prep1 = df_train_prep1[df_train_prep1['Employee_status'] == 'Permanent']

    ############################### Feature Engineering Scikit ############################

    # MODELING

    new_var_one_hot = ['age2','year_graduated2','job_duration_as_permanent_worker2','in study','person level up']

    one_hot_var = [
        'job_level_oh', 
        'person_level_oh', 
        'Employee_type', 
    #     'Employee_status',   
        'gender', 
        'marital_status_maried(Y/N)', 
    #     'Education_level',
        'achievement_target_1', 
        'achievement_target_2',
        'achievement_target_3'] + new_var_one_hot

    ordinal_val = [
        'job_level',
        'person_level',
        'Education_level'
    #     'achievement_target_1', 
    #     'achievement_target_2'
    ]

    new_var_num = ['year_graduated','total leave','age3','total_rotation','training - duration']

    numeric_var = [
        'job_duration_in_current_job_level',   
        'job_duration_in_current_person_level',
        'job_duration_in_current_branch', 
        'age', 
        'number_of_dependences',
        'number_of_dependences (male)', 
        'number_of_dependences (female)', 
        'GPA',   
        'job_duration_as_permanent_worker', 
        'job_duration_from_training',   
        'branch_rotation', 
        'job_rotation', 
        'assign_of_otherposition',   
        'annual leave', 
        'sick_leaves',
        'Avg_achievement_%',
        'Last_achievement_%', 
        'Achievement_above_100%_during3quartal'] + new_var_num

    ordinal_mapping = [
        {'col':'job_level',
        'mapping':{None:0,'JG04':1,'JG05':2,'JG03':3,'JG06':4}},
        {'col':'person_level',
        'mapping':{None:0,'PG01':1,'PG02':2,'PG03':3,'PG04':4, 'PG05':5, 'PG06':6, 'PG07':7, 'PG08':8}},
        {'col':'Education_level',
        'mapping':{None:0,'other a':1, 'level_0':2,'level_1':3,'other b':5,'level_2':4, 'level_3':6, 'level_4':7, 'level_5':8}}    
        ]

    ordinal_encoder = ce.OrdinalEncoder(cols = ['job_level','person_level','Education_level'],mapping = ordinal_mapping)

    one_hot_encoder_pipeline = Pipeline([
                                        ('imputer',SimpleImputer(strategy = 'most_frequent')),
                                        ('one hot encoder',OneHotEncoder(handle_unknown = 'ignore'))
    ])

    numerical_pipeline = SimpleImputer(strategy = 'median')

    transformer = ColumnTransformer([
        ('one hot encoder',one_hot_encoder_pipeline,one_hot_var),
        ('ordinal encoder',ordinal_encoder,ordinal_val),
        ('numerical_pipeline',numerical_pipeline,numeric_var)
    ])

    ################################ Daata Splitting ############################

    var_x = numeric_var + one_hot_var + ordinal_val

    X = df_train_prep1[var_x]
    X_sub = df_test_prep1[var_x]
    y = df_train_prep1['Best Performance']

    X_trainval, X_test, y_trainval, y_test = train_test_split(X,y, stratify = y, test_size = 2000, random_state = 200)
#     X_train, X_val, y_train, y_val = train_test_split(X_trainval,y_trainval, stratify = y_trainval, test_size = 2000, random_state = 200)

    ################################ Sklearn Prepreocess Fitting ############################

    transformer.fit(X)

    one_hot_result = list(transformer.transformers_[0][1]['one hot encoder'].get_feature_names())
    features = one_hot_result + ordinal_val + numeric_var
    
    data_full = [X, X_sub, y]
    data_ml = [X_trainval, X_test, y_trainval, y_test]
    
    return transformer, features, data_full, data_ml

# lanjut modeling