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

months = ['Aug-Sep', 'Oct', 'Nov', 'Dec-Jan']


os.getcwd()
def finalizedData():
    df = pd.DataFrame()
    for year in range(2009, 2020):
        schedule = pd.read_csv('Schedule/schedule{}.csv'.format(year))
        months = ['Oct', 'Nov', 'Dec-Jan']
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

#finalizedData()

df = pd.read_csv('AllData.csv')
y = df.HomeWin
features = list(df.columns[6:])
X = df[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
model = XGBClassifier(n_estimators=1000, learning_rate=0.05)
model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(val_X, val_y)],verbose=False)
prediction = model.predict(val_X)
accuracy = accuracy_score(val_y, prediction)

'''
prediction = cross_val_predict(model, X, y, cv=5)
accuracy = accuracy_score(y, prediction.round())
'''
print(accuracy)

model2 = RandomForestClassifier(n_estimators=100)
cv_scores = cross_val_score(model2, X, y, cv=5, scoring='accuracy')
print(cv_scores)


