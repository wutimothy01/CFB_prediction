'''
A web scraper to scrape for College Football Stats
@author: Timothy Wu, Jasper Wu
'''

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import csv
import os
import datetime

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
#headers for corresponding stats (offense and defense have same headers)
score_header = ['Rank', 'Name', 'G', 'TD', 'FG', '1XP', '2XP', 'Safety', 'Points', 'Points/G']

rush_header = ['Rank', 'Name', 'G', 'Att', 'Yards', 'Avg', 'TD', 'Att/G', 'Yards/G']

pass_header = ['Rank', 'Name', 'G', 'Att', 'Comp', 'Pct', 'Yards', 'Yards/Att', 'TD', 'Int', 'Rating', 'Att/G', 'Yards/G']

penalties_header = ['Rank', 'Name', 'G', 'Pen', 'Yards', 'Pen/G', 'Yards/G']

turnover_header = ['Rank', 'Name', 'G', 'Fum. Gain', 'Int. Gain', 'Total Gain', 'Fum. Lost', 'Int. Lost', 'Total Lost', 'Margin', 'Margin/G']

third_conversions_header = ['Rank', 'Name', 'G', 'Attempts', 'Conversions', 'Conversion %']

redzone_header = ['Rank', 'Name', 'G', 'Attempts', 'Scores', 'Score %', 'TD', 'TD %', 'FG', 'FG %']

# dictionary that converts number in url to month want to scrape
months = {15: 'August-September', 16: 'October', 17: 'November', 18: 'December'}

# dictionary that converts the type of stat into the number corresponding in the url as well as the corresponding header for the dataframe
modes = {'score': ('09', score_header), 'rush': ('01', rush_header), 'pass': ('02', pass_header), 'turnover': ('12', turnover_header), 
    'penalties': ('14', penalties_header), '3rd': ('25', third_conversions_header), 'red': ('27', redzone_header)}

# array that corresponds to either offense or defense side
sides = ['offense', 'defense']

'''
takes the type of stat, the year, the month, and offense or defense and scrapes the stats on the corresponding page, saving it as a csv 
with corresponding name

key: the key to the dictionary that lists all the types of stats to track
year: the specific year we are looking at
month: the specific month we are looking at
side: either offense or defense (team's vs opponent's) for most stats except turnovers

csv file is in the form: data-year-month-side-key.csv
'''
def getRawData(key, year, month, side):
    # url based on the parameters
    url = 'http://www.cfbstats.com/' + str(year) + '/leader/national/team/' + side + '/split' + str(month) + '/category' + modes[key][0] + '/sort01.html'
    
    # transform the url into soup
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "html.parser")
    table = soup.find('table')
    all_rows = []
    
    # scrape the stats into the array all_rows
    for trs in table.find_all('tr'):
        tds = trs.find_all('td')
        row = [cell.text.strip() for cell in tds]
        # only add the row if its not empty
        if row:
            all_rows.append(row)
    
    # convert the array into a dataframe
    df = pd.DataFrame(data=all_rows, columns = modes[key][1])
    # set the index to the Rank
    df.set_index(['Rank'], drop=True, inplace=True)
    # import the stats to a csv file and encode it
    df.to_csv('{}-{}-{}-{}.csv'.format(year, months[month], side, key), encoding = 'utf-8')
    # print to say we are done
    print('{}-{}-{}-{}.csv done'.format(year, months[month], side, key))
    

def scrape_raw_data():
    # for each year from 2009 to 2020
    for year in range(2009, 2020):
        # for each type of stat want to check
        for key in modes:
            # for each month from August to December
            for month in months:
                # if its turnover only do offense otherwise do both offense and defense
                if key == 'turnover':
                    # get the data with the specified parameters
                    getRawData(key, year, month, sides[0])              
                else:
                    for side in sides:
                        getRawData(key, year, month, side)

schedule_header_modern = ['Week', 'Date', 'Time', 'Day', 'Home', 'HomePts', 'Where', 'Away', 'AwayPts', 'Notes']
schedule_header_old = ['Week', 'Date', 'Day', 'Home', 'HomePts', 'Where', 'Away', 'AwayPts', 'Notes']
def scrape_schedule(year):
    url = 'https://www.sports-reference.com/cfb/years/' + str(year) + '-schedule.html'
    
    # transform the url into soup
    r = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data, "html.parser")
    table = soup.find('table')
    all_rows = []

    # scrape the schedule into the array all_rows
    for trs in table.find_all('tr'):
        tds = trs.find_all('td')
        row = [cell.text.strip() for cell in tds]
        # only add the row if its not empty
        if row:
            all_rows.append(row)
    
    # convert the array into a dataframe
    if year < 2013:
        df = pd.DataFrame(data=all_rows, columns = schedule_header_old)
    else:
        df = pd.DataFrame(data=all_rows, columns = schedule_header_modern)
    # parse out the month and year
    df['Month'] = df['Date'].str.slice(stop=3)
    df['Year'] = df['Date'].str.slice(start=7)

    #take out ranking (8) from ranked teams 
    for elem in range(df['Home'].size):
        if '(' in df.at[elem, 'Home'][0:1]:
            df.at[elem, 'Home'] = df.at[elem, 'Home'].split(')', 1)[1].strip()
    for elem in range(df['Away'].size):
        if '(' in df.at[elem, 'Away'][0:1]:
            df.at[elem, 'Away'] = df.at[elem, 'Away'].split(')', 1)[1].strip()

    #swap home and away if in wrong order
    idx = (df['Where'] == '@')
    df.loc[idx, ['Home', 'Away']] = df.loc[idx, ['Away', 'Home']].values

    #set the index to the month and year
    df.set_index(['Month', 'Year'], drop=True, inplace=True)

    #drop unnecessary columns
    if year < 2013: 
        df.drop(columns=['Week', 'Date', 'Day', 'Where', 'Notes'], inplace=True)
    else:
        df.drop(columns=['Week', 'Date', 'Time', 'Day', 'Where', 'Notes'], inplace=True)
    # import the stats to a csv file and encode it
    df.to_csv('schedule' + str(year) + '.csv', encoding = 'utf-8')
    # print to say we are done
    print('schedule' + str(year) + '.csv done')

def scrape_raw_schedule():
    for year in range(2009, 2020):
        scrape_schedule(year)


# changes path to current working directory and if the Data folder doesn't exist, make it and change to data folder to store data
os.getcwd()
if not os.path.exists('Data'):
    os.mkdir('Data')
if not os.path.exists('Schedule'):
    os.mkdir('Schedule')
os.chdir('Data')
#comment this to not have it scrape 10 years worth of data again
#scrape_raw_data()

# test the scraper with just one possibility rather than all
#getData('score', 2019, 15, 'offense')

#change path to schedule to store data
os.chdir('../Schedule')

#comment this to not have it scrape 10 years worth of schedules again
scrape_raw_schedule()

#test the schedule scraper with just one possibility
#scrape_schedule(2019)


