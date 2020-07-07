'''
Scrape and manipulate data into final aggregate datasets for the model
Scrape and manipulate schedule data into final dataset for the model
@author: Timothy Wu, Jasper Wu
'''

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import csv
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
#headers for corresponding stats (offense and defense have same headers)
score_header = ['Rank', 'Name', 'G', 'TD', 'FG', '1XP', '2XP', 'Safety', 'Points', 'Points/G']

rush_header = ['Rank', 'Name', 'G', 'Att', 'Yards', 'Avg', 'TD', 'Att/G', 'Yards/G']

pass_header = ['Rank', 'Name', 'G', 'Att', 'Comp', 'Pct', 'Yards', 'Yards/Att', 'TD', 'Int', 'Rating', 'Att/G', 'Yards/G']

penalties_header = ['Rank', 'Name', 'G', 'Pen', 'Yards', 'Pen/G', 'Yards/G']

turnover_header = ['Rank', 'Name', 'G', 'Fum. Gain', 'Int. Gain', 'Total Gain', 'Fum. Lost', 'Int. Lost', 'Total Lost', 'Margin', 'Margin/G']

third_conversions_header = ['Rank', 'Name', 'G', 'Attempts', 'Conversions', 'Conversion %']

redzone_header = ['Rank', 'Name', 'G', 'Attempts', 'Scores', 'Score %', 'TD', 'TD %', 'FG', 'FG %']

# dictionary that converts number in url to month want to scrape
months = {15: 'Aug-Sep', 16: 'Oct', 17: 'Nov', 18: 'Dec-Jan'}

# dictionary that converts the type of stat into the number corresponding in the url as well as the corresponding header for the dataframe
modes = {'turnover': ('12', turnover_header), 'score': ('09', score_header), 'rush': ('01', rush_header), 'pass': ('02', pass_header),
    'penalties': ('14', penalties_header), '3rd': ('25', third_conversions_header), 'red': ('27', redzone_header)}

# array that corresponds to either offense or defense side
sides = ['offense', 'defense']

'''
takes the type of stat, the year, the month, and offense or defense and scrapes the stats on the corresponding page,
returning a dataframe with the stats

key: the key to the dictionary that lists all the types of stats to track
year: the specific year we are looking at
month: the specific month we are looking at
side: either offense or defense (team's vs opponent's) for most stats except turnovers

returns a dataframe for the corresponding stat for the corresponding side of offense for the corresponding month and year
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
    # save the names to use for schedule scrape
    if month == 17 and key == 'score' and side == 'offense':
        df.to_csv('../Names/{}.csv'.format(year))
    # set the index to the names
    df.set_index(['Name'], drop=True, inplace=True)
    # drop the columns of rank as well as the per game and percentage stats since they cannot be aggregated
    df.drop(columns=['Rank'], inplace=True)
    df.drop(columns=[col for col in df.columns if '%' in col or '/G' in col], inplace=True)
    # add an ending to the column headers to be able to distinguish
    df = df.add_suffix('_{}_{}'.format(key, side))
    # print to say we are done
    print('Scraped {}-{}-{}-{}'.format(year, months[month], key, side))
    # return the dataframe
    return df
    
#fix bad names in schedule to match
def fix_names(name):
    if name == 'Louisiana':
        return 'Louisiana-Lafayette'
    if name == 'Middle Tennessee State':
        return 'Middle Tennessee'
    if name == 'Nevada-Las Vegas':
        return 'UNLV'
    if name == 'Brigham Young':
        return 'BYU'
    if name == 'Miami (FL)':
        return 'Miami (Florida)'
    if name == 'Miami (OH)':
        return 'Miami (Ohio)'
    if name == 'Southern Methodist':
        return 'SMU'
    if name == 'Louisiana State':
        return 'LSU'
    if name == 'Alabama-Birmingham':
        return 'UAB'
    if name == 'Hawaii':
        return "Hawai'i"
    if name == 'Central Florida':
        return 'UCF'
    if name == 'Texas Christian':
        return 'TCU'
    if name == 'Texas-San Antonio':
        return 'UTSA'
    if name == 'Texas-El Paso':
        return 'UTEP'
    if name == 'Southern California':
        return 'USC'
    if name == 'Bowling Green State':
        return 'Bowling Green'
    return name

schedule_header_modern = ['Week', 'Date', 'Time', 'Day', 'Home', 'HomePts', 'Where', 'Away', 'AwayPts', 'Notes']
schedule_header_old = ['Week', 'Date', 'Day', 'Home', 'HomePts', 'Where', 'Away', 'AwayPts', 'Notes']

'''
scrapes the schedule for the year
takes out irrelevant teams and edits the file to make sure home and away are correct and who won
return dataframe with info
'''
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

    # aggregate August and September together and December and January together
    for index, row in df.iterrows():
        if 'Sep' in row['Month'] or 'Aug' in row['Month']:
            row['Month'] = 'Aug-Sep'
        if 'Dec' in row['Month'] or 'Jan' in row['Month']:
            row['Month'] = 'Dec-Jan'
    
    #take out ranking (8) from ranked teams 
    for index, row in df.iterrows():
        if '(' in row['Home'][0:1]:
            row['Home'] = row['Home'].split(')', 1)[1].strip()
        if '(' in row['Away'][0:1]:
            row['Away'] = row['Away'].split(')', 1)[1].strip()

    #fix schedule names to match with raw stats names
    df['Home'] = df['Home'].apply(fix_names)
    df['Away'] = df['Away'].apply(fix_names)

    #drop schools that suck
    names = pd.read_csv('../Names/{}.csv'.format(year))
    for index, row in df.iterrows():
        if (row['Home'] not in names['Name'].unique()) or (row['Away'] not in names['Name'].unique()):
            df.drop(index, inplace=True)

    #swap home and away if in wrong order
    
    idx = (df['Where'] == '@')
    df.loc[idx, ['Home', 'Away']] = df.loc[idx, ['Away', 'Home']].values
    df.loc[idx, ['HomePts', 'AwayPts']] = df.loc[idx, ['AwayPts', 'HomePts']].values

    #set the index to the month and year
    df.set_index(['Month', 'Year'], drop=True, inplace=True)
    #check who wins (1 = home won, 0 = away won)
    df['HomeWin'] = 0
    index = (df['HomePts'] > df['AwayPts'])
    df.loc[index, ['HomeWin']] = 1

    #drop unnecessary columns
    if year < 2013: 
        df.drop(columns=['Week', 'Date', 'Day', 'Where', 'Notes'], inplace=True)
    else:
        df.drop(columns=['Week', 'Date', 'Time', 'Day', 'Where', 'Notes'], inplace=True)
    #change path to schedule to store data
    os.chdir('../Schedule')
    # import the stats to a csv file and encode it
    df.to_csv('schedule' + str(year) + '.csv', encoding = 'utf-8')
    # print to say we are done
    print('schedule' + str(year) + '.csv done')
    os.chdir('../AggregateData')

#scrape schedule from 2009 to 2020
def scrape_raw_schedule():
    for year in range(2009, 2020):
        scrape_schedule(year)

def scrape_raw_data():
    # for each year from 2009 to 2020
    for year in range(2009, 2020):
        aggregate = pd.DataFrame()
        # for each month
        for month in months:
            # for each type of stat want to check
            oneMonth = pd.DataFrame()
            for key in modes:
                # if its turnover only do offense otherwise do both offense and defense
                if key == 'turnover':
                    # start with turnover and start the joined dataframe for the month
                    oneMonth = getRawData(key, year, month, sides[0])        
                else:
                    for side in sides:
                        temp = getRawData(key, year, month, side)
                        #join the data to the ongoing dataframe of all stats for the month
                        oneMonth = oneMonth.join(temp)
            #change to numeric numbers so we can add
            oneMonth = oneMonth.apply(pd.to_numeric)     
            #aggregate it to set up for predictions
            aggregate = aggregate.add(oneMonth, fill_value=0)
            #save to a csv file with the corresponding predictions for the future
            if month < 18:
                aggregate.to_csv('{}-{}-Predictions.csv'.format(months[month+1], year), encoding = 'utf-8')
                print('aggregate-data-{}-{}-Predictions.csv'.format(months[month+1], year))
            else:
                aggregate.to_csv('Aug-Sep-{}-Predictions.csv'.format(year+1), encoding = 'utf-8')
                print('aggregate-data-Aug-Sep-{}-Predictions.csv'.format(year+1))

            #edit the dataframe to be per game by dividing by how many games played
            aggregatepergame = aggregate.div(aggregate['G_turnover_offense'], axis='index')
            aggregatepergame.drop(columns=[col for col in aggregatepergame.columns if 'G_' in col[0:2]], inplace=True)
            #save to a csv file for corresponding prediction month and year for per game data
            if month < 18:
                aggregatepergame.to_csv('{}-{}-Predictions-Per-Game.csv'.format(months[month+1], year), encoding = 'utf-8')
                print('aggregate-data-{}-{}-Predictions-Per-Game.csv'.format(months[month+1], year))
            else:
                aggregatepergame.to_csv('Aug-Sep-{}-Predictions-Per-Game.csv'.format(year+1), encoding = 'utf-8')
                print('aggregate-data-Aug-Sep-{}-Predictions-Per-Game.csv'.format(year+1))



# changes path to current working directory and if the Data folder doesn't exist, make it and change to data folder to store data
os.getcwd()
if not os.path.exists('AggregateData'):
    os.mkdir('AggregateData')
if not os.path.exists('Schedule'):
    os.mkdir('Schedule')
if not os.path.exists('Names'):
    os.mkdir('Names')
os.chdir('AggregateData')

#run the program, comment out what to run and what not to run
scrape_raw_data()
scrape_raw_schedule()
