'''
Machine learning model
@author: Timothy Wu, Jasper Wu
'''

import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import sklearn.metrics as metrics
from sklearn.feature_selection import SelectKBest, f_classif
import eli5
from eli5.sklearn import PermutationImportance


months = ['Oct', 'Nov', 'Dec-Jan']


os.getcwd()
def finalizedData():
    df = pd.DataFrame()
    for year in range(2009, 2020):
        schedule = pd.read_csv('Schedule/schedule{}.csv'.format(year))
        if year == 2010:
            months.insert(0, 'Aug-Sep')
        for month in months:
            rawdata = pd.read_csv('AggregateData/{}-{}-Predictions-Per-Game.csv'.format(month, year))
            index = (schedule['Month'] == month)
            temp = schedule.loc[index]

            temp.set_index(['Home'], drop=True, inplace=True)
            rawdata.set_index(['Name'], drop=True, inplace=True)
            temp = temp.join(rawdata)

            temp.set_index(['Away'], drop=True, inplace=True)
            temp = temp.join(rawdata, rsuffix='_Away')
            df = df.append(temp, ignore_index=True)
    df.to_csv('AllData.csv', encoding='utf-8')

finalizedData()
df = pd.read_csv('AllData.csv')
y = df.HomeWin
features = list(df.columns[6:])
X = df[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
model = XGBClassifier(n_estimators=1000, learning_rate=0.05)
model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(val_X, val_y)],verbose=False)
prediction = model.predict(val_X)
accuracy = accuracy_score(val_y, prediction)
print(accuracy)

'''
perm = PermutationImportance(model, random_state=1).fit(val_X, val_y)
print(eli5.format_as_text(eli5.explain_weights(perm, feature_names=val_X.columns.tolist())))
'''


'''
train = list(df.columns[5:])
data = df[train]

valid_fraction = 0.1
valid_size = int(len(X) * valid_fraction)
train = data[:-2 * valid_size]
valid = data[-2 * valid_size:-valid_size]
test = data[-valid_size:]

dtrain = lgb.Dataset(train[features], label=train['HomeWin'])
dvalid = lgb.Dataset(valid[features], label=valid['HomeWin'])

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)

ypred = bst.predict(test[features])
score = metrics.roc_auc_score(test['HomeWin'],ypred)
print(score)
'''


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

d_train = lgb.Dataset(x_train, label=y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['num_boost_round'] = 100
params['num_leaves'] = 31
params['metric'] = 'binary_logloss'
params['max_depth'] = 10
params['sub_features'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50

model = lgb.train(params, d_train, 100)
ypred = model.predict(x_test)
accuracy = accuracy_score(y_test, ypred.round())
print(accuracy)

