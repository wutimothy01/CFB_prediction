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

score_off = 'Score_Off.csv' 
score_def = 'Score_Def.csv'
score_header = ['Rank', 'Name', 'G', 'TD', 'FG', '1XP', '2XP', 'Safety', 'Points', 'Points/G']
score_off_url = 'http://www.cfbstats.com/2019/leader/national/team/offense/split01/category09/sort01.html'

rush_off = 'Rush_Off.csv'
rush_def = 'Rush_Def.csv'
rush_header = ['Rank', 'Name', 'G', 'Att', 'Yards', 'Avg', 'TD', 'Att/G', 'Yards/G']

pass_off = 'Pass_Off.csv'
pass_def = 'Pass_Def.csv'
pass_header = ['Rank', 'Name', 'G', 'Att', 'Comp', 'Pct', 'Yards', 'Yards/Att', 'TD', 'Int', 'Rating', 'Att/G', 'Yards/G']

penalties = 'Penalties.csv'
opp_penalties = 'Opp_Penalties.csv'
penalties_header = ['Rank', 'Name', 'G', 'Pen', 'Yards', 'Pen/G', 'Yards/G']

turnover = 'Turnover.csv'
turnover_header = ['Rank', 'Name', 'G', 'Fum. Gain', 'Int. Gain', 'Total Gain', 'Fum. Lost', 'Int. Lost', 'Total Lost', 'Margin', 'Margin/G']

third_conversions = 'Third_Conversions.csv'
opp_third_conversions = 'Opp_Third_Conversions.csv'
third_conversions_header = ['Rank', 'Name', 'G', 'Attempts', 'Conversions', 'Conversion %']

redzone = 'Redzone.csv'
opp_redzone = 'Opp_Redzone.csv'
redzone_header = ['Rank', 'Name', 'G', 'Attempts', 'Scores', 'Score %', 'TD', 'TD %', 'FG', 'FG %']

url = 'http://www.cfbstats.com/2019/leader/national/team/offense/split15/category09/sort01.html'
r = requests.get(url)
data = r.text
soup = BeautifulSoup(data, "html.parser")
table = soup.find('table')
all_rows = []
for trs in table.find_all('tr'):
    tds = trs.find_all('td')
    row = [cell.text.strip() for cell in tds]
    if len(row) != 0 and (len(all_rows) == 0 or all_rows[-1][0] != row[0]):
        all_rows.append(row)

df = pd.DataFrame(data=all_rows, columns = header)
df.set_index(['Rank'], drop=True, inplace=True)
print("Scoring Offense Done")