'''
Manage the data scraped and prepare it to insert into the model
@author: Timothy Wu, Jasper Wu
'''

import numpy as np
import pandas as pd
import csv
import os

months = ['August-September', 'October', 'November', 'December']
convertedMonths = ['Aug/Sep', 'Oct', 'Nov', 'Dec/Jan']
sides = ['offense', 'defense']
modes = ['score', 'rush', 'pass', 'penalties', '3rd', 'red']

os.getcwd()
if not os.path.exists('Modified Data'):
    os.mkdir('Modified Data')
os.chdir('Modified Data')

def joinData(year, index):
    final = pd.read_csv('../Data/{}-{}-{}-{}.csv'.format(year, months[index], 'offense', 'turnover'))
    final['Month'] = convertedMonths[index]
    final['Year'] = year
    final.set_index(['Name', 'Month', 'Year'], drop=True, inplace=True)
    final.drop(columns=['Rank'], inplace=True)
    final = final.add_suffix('_turnover')
    for mode in modes:
        for side in sides:
            temp = pd.read_csv('../Data/{}-{}-{}-{}.csv'.format(year, months[index], side, mode))
            temp['Month'] = convertedMonths[index]
            temp['Year'] = year
            temp.set_index(['Name', 'Month', 'Year'], drop=True, inplace=True)
            temp.drop(columns=['Rank'], inplace=True)
            temp = temp.add_suffix('_{}_{}'.format(mode, side))
            final = final.join(temp)
            print('joined {} {} {} {}'.format(year, months[index], side, mode))
    final.to_csv('Modified_Data_{}-{}.csv'.format(year, months[index]), encoding = 'utf-8')


for year in range(2009, 2020):
    for month in range(0,4):
        joinData(year, month)





