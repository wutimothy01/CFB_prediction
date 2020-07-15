import pandas as pd
import numpy as np
import os
import datetime
import xgboost as xgb
import sklearn
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

os.getcwd()
months = ['Oct', 'Nov', 'Dec-Jan']
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
y = df['HomeWin']
'''
df = df.drop(columns=['Fum. Gain_turnover_offense', 'Int. Gain_turnover_offense', 'Fum. Lost_turnover_offense', 'Int. Lost_turnover_offense', 'Margin_turnover_offense',
                        '2XP_score_offense', 'Safety_score_offense', '2XP_score_defense', 'Safety_score_defense', 'Fum. Gain_turnover_offense_Away', 
                        'Int. Gain_turnover_offense_Away', 'Fum. Lost_turnover_offense_Away', 'Int. Lost_turnover_offense_Away', 'Margin_turnover_offense_Away',
                        '2XP_score_offense_Away', 'Safety_score_offense_Away', '2XP_score_defense_Away', 'Safety_score_defense_Away'])
'''

#features = list(df.columns[6:])
features = ['Margin_turnover_offense', 'Points_score_offense', 'Points_score_defense', 'Att_rush_offense', 'Yards_rush_offense', 'Yards_rush_defense',
            'Yards_pass_offense', 'Yards_pass_defense', 'Pen_penalties_offense', 'Pen_penalties_defense', 'Margin_turnover_offense_Away', 'Points_score_offense_Away', 
            'Points_score_defense_Away', 'Att_rush_offense_Away', 'Yards_rush_offense_Away', 'Yards_rush_defense_Away',
            'Yards_pass_offense_Away', 'Yards_pass_defense_Away', 'Pen_penalties_offense_Away', 'Pen_penalties_defense_Away']
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=812)

def modelfit(model, X_train, X_test, y_train, y_test, cv_folds=5, early_stopping_rounds=50, featureimportance=False):
    xgb_param = model.get_xgb_params()
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds, metrics='auc',
                        early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
    model.set_params(n_estimators=cvresult.shape[0])
    model.fit(X_train, y_train, eval_metric='auc')
    trainpred = model.predict(X_train)
    testpred = model.predict(X_test)
    trainprob = model.predict_proba(X_train)[:,1]
    testprob = model.predict_proba(X_test)[:,1]
    if not featureimportance:
        print("\nModel Report")
        print(f"Parameters: {model.get_params()}")
        print(f"Accuracy (Train): {metrics.accuracy_score(y_train, trainpred) * 100} %")
        print(f"AUC Score (Train): {metrics.roc_auc_score(y_train, trainprob)}")
        print(f"Accuracy (Test): {metrics.accuracy_score(y_test, testpred) * 100} %")
        print(f"AUC Score (Test): {metrics.roc_auc_score(y_test, testprob)}")

    return model
print("started", end="")
print(datetime.datetime.now())
initialmodel = XGBClassifier(
            learning_rate=0.1,
            n_estimators=1000,
            #default 5
            max_depth=5,
            #default 1
            min_child_weight=1,
            #default 0
            gamma=0,
            #default 0.8
            subsample=0.8,
            #default 0.8
            colsample_bytree=0.8,
            #default 0
            reg_alpha=0,
            #default 1
            reg_lambda=1,
            objective= 'binary:logistic',
            scale_pos_weight=1,
            seed=812)

model = modelfit(initialmodel, X_train, X_test, y_train, y_test)

def optimizeparams(initialmodel, params):
    gsearch = GridSearchCV(initialmodel, param_grid = params, scoring='roc_auc', n_jobs=4, cv=5)
    gsearch.fit(X_train, y_train)
    print(gsearch.best_params_)
    print(gsearch.best_score_)
    return gsearch.best_estimator_

from numpy import sort
from sklearn.feature_selection import SelectFromModel
def featureimportance(model, X_train, X_test, y_train, y_test):
    thresholds = sort(model.feature_importances_)
    bestthresh = 0
    bestN = 0
    bestaccuracy = 0
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        select_X_test = selection.transform(X_test)
        # train model
        selection_model = XGBClassifier()
        selection_model = modelfit(selection_model, select_X_train, select_X_test, y_train, y_test, featureimportance=True)
        # eval model
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = metrics.accuracy_score(y_test, predictions)
        print(f"Thresh={thresh}, n={select_X_train.shape[1]}, Accuracy: {accuracy*100}%")
        if accuracy > bestaccuracy:
            bestthresh = thresh
            bestN = select_X_train.shape[1]
            bestaccuracy = accuracy

    print(f"Best Run: Thresh={bestthresh}, n={bestN}, Accuracy: {bestaccuracy*100}%")
    xgb.plot_importance(model, height=2)
    plt.tick_params(axis='y', which='major', labelsize=5)
    plt.show()

params = {'max_depth':[3,5,7,9], 'min_child_weight':[1,3,5]}
model = optimizeparams(model, params)
bestmaxdepth = model.get_params()['max_depth']
bestminchildweight = model.get_params()['min_child_weight']
params = {'max_depth':[bestmaxdepth-1, bestmaxdepth, bestmaxdepth+1], 'min_child_weight':[bestminchildweight-1,bestminchildweight,bestminchildweight+1]}
model = optimizeparams(model, params)
model.set_params(n_estimators=1000)
model = modelfit(model, X_train, X_test, y_train, y_test)

params = {'gamma':[i/10.0 for i in range(0,5)]}
model = optimizeparams(model, params)
bestgamma = model.get_params()['gamma']
if bestgamma == 0:
    bestgamma = 0.05
params = {'gamma':[i/100.0 for i in range(int(bestgamma*100-5.0), int(bestgamma*100+10.0), 5)]}
model = optimizeparams(model, params)
model.set_params(n_estimators=1000)
model = modelfit(model, X_train, X_test, y_train, y_test)

params = {'subsample':[i/10.0 for i in range(6,10)], 'colsample_bytree':[i/10.0 for i in range(6,10)]}
model = optimizeparams(model, params)
bestsub = model.get_params()['subsample']
bestcol = model.get_params()['colsample_bytree']
params = {'subsample':[i/100.0 for i in range(int(bestsub*100-5.0), int(bestsub*100+10.0), 5)], 
        'colsample_bytree':[i/100.0 for i in range(int(bestcol*100-5.0), int(bestcol*100+10.0), 5)]}
model = optimizeparams(model, params)
model.set_params(n_estimators=1000)
model = modelfit(model, X_train, X_test, y_train, y_test)

params = {'reg_alpha':[1e-5, 1e-2, 0, 1, 100], 'reg_lambda':[1e-5, 1e-2, 0, 1, 100]}
model = optimizeparams(model, params)
bestalpha = model.get_params()['reg_alpha']
bestlambda = model.get_params()['reg_lambda']
alp = bestalpha
lamb = bestlambda
if bestalpha == 0:
    alp = 1
if bestlambda == 0:
    lamb = 1
params = {'reg_alpha':[alp/100.0, alp/10.0, bestalpha, alp*10, alp*100], 
            'reg_lambda':[lamb/100.0, lamb/10.0, bestlambda, lamb*10, lamb*100]}
model = optimizeparams(model, params)
model.set_params(n_estimators=1000)
model = modelfit(model, X_train, X_test, y_train, y_test)

model.set_params(n_estimators=5000, learning_rate=0.01)
model = modelfit(model, X_train, X_test, y_train, y_test)

featureimportance(model, X_train, X_test, y_train, y_test)
print("finished", end="")
print(datetime.datetime.now())
