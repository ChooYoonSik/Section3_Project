import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
import pickle

df_eng = pd.read_csv('project3_data.csv', encoding='utf-8')

train, test = train_test_split(df_eng, test_size=0.2, random_state=42)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

X_train = train.iloc[:,:-1]
y_train = train.iloc[:, -1]

X_test = test.iloc[:,:-1]
y_test = test.iloc[:, -1]

pipe_tuning = Pipeline([
                ('Encoding', TargetEncoder()),
                ('Scaling', StandardScaler()),
                ('XGB', XGBRegressor(XGB__reg_lambda=5, 
                                     XGB__reg_alpha=0.5, 
                                     XGB__n_estimators=400, 
                                     XGB__min_child_weight=15, 
                                     XGB__max_depth=20, 
                                     XGB__learning_rate=0.1, 
                                     XGB__gamma=2, 
                                     XGB__colsample_bytree=0.9, 
                                     random_state=42))
                ])

pipe_tuning.fit(X_train, y_train)

with open('pipe.pkl','wb') as pickle_file:
    pickle.dump(pipe_tuning, pickle_file)

