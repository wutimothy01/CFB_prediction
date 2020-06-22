'''
A web scraper to scrape for College Football Stats
@author: Timothy Wu, Jasper Wu
'''

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import csv
import re
from datetime import datetime
import os

'''
Scrape the following stats:
    Scoring Offense
    Scoring Defense
    Rushing Offense
    Rushing Defense
    Passing Offense
    Passing Defense
    Penalties
    Opponent Penalties
    Turnover Margin
    3rd Down Conversions
    Opponent 3rd Down Conversions
    Redzone Conversions
    Opponent Redzone Conversions
'''

score_header = ['Rank', 'Name', 'G', 'TD', 'FG', '1XP', '2XP', 'Safety', 'Points', 'Points/G']

rush_header = ['Rank', 'Name', 'G', 'Att', 'Yards', 'Avg', 'TD', 'Att/G', 'Yards/G']

pass_header = ['Rank', 'Name', 'G', 'Att', 'Comp', 'Pct', 'Yards', 'Yards/Att', 'TD', 'Int', 'Rating', 'Att/G', 'Yards/G']

penalties_header = ['Rank', 'Name', 'G', 'Pen', 'Yards', 'Pen/G', 'Yards/G']

turnover_header = ['Rank', 'Name', 'G', 'Fum. Gain', 'Int. Gain', 'Total Gain', 'Fum. Lost', 'Int. Lost', 'Total Lost', 'Margin', 'Margin/G']

third_conversions_header = ['Rank', 'Name', 'G', 'Attempts', 'Conversions', 'Conversion %']

redzone_header = ['Rank', 'Name', 'G', 'Attempts', 'Scores', 'Score %', 'TD', 'TD %', 'FG', 'FG %']

months = {15: 'August-September', 16: 'October', 17: 'November', 18: 'December'}
modes = {'score': ('09', score_header), 'rush': ('01', rush_header), 'pass': ('02', pass_header), 'turnover': ('12', turnover_header), 
    'penalties': ('14', penalties_header), '3rd': ('25', third_conversions_header), 'red': ('27', redzone_header)}
sides = ['offense', 'defense']


def getData(key, year, month, side):
    url = 'http://www.cfbstats.com/' + str(year) + '/leader/national/team/' + side + '/split' + str(month) + '/category' + modes[key][0] + '/sort01.html'
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "html.parser")
    table = soup.find('table')
    all_rows = []
    for trs in table.find_all('tr'):
        tds = trs.find_all('td')
        row = [cell.text.strip() for cell in tds]
        if row:
            all_rows.append(row)
    df = pd.DataFrame(data=all_rows, columns = modes[key][1])
    df.set_index(['Rank'], drop=True, inplace=True)
    df.to_csv('data-' + str(year) + '-' + months[month] + '-' + side + '-' + key + '.csv', encoding = 'utf-8')
    print('data-' + str(year) + '-' + months[month] + '-' + side + '-' + key + '.csv done')

os.getcwd()
os.chdir('Data')

for year in range(2009, 2020):
    for key in modes:
        for month in months:
            if key == 'turnover':
                getData(key, year, month, sides[0])              
            else:
                for side in sides:
                    getData(key, year, month, side)

# test the scraper with just one scrape
#getData('score', 2019, 15, 'offense')

