import json
import time

import pandas as pd
from sportsipy.nfl.schedule import Schedule
from sportsipy.nfl.teams import Teams

with open('resources/config.json', 'r') as config_file:
    config = json.load(config_file)


def get_division(team_name):
    name_to_division = config.get('team_divisions')

    return name_to_division.get(team_name)


def get_name_from_abbrev(abbrev):
    abbrev_to_name = config.get('team_abbreviations')

    return abbrev_to_name.get(abbrev, abbrev)


def load_schedule():
    with open(config.get('resource_locations').get('schedule'), 'r') as f:
        schedule = json.load(f)
        return schedule


def get_games_before_week(week, use_persisted=True):
    if use_persisted:
        week_results = pd.read_csv(config.get('resource_locations').get('games'))
        week_results = week_results.dropna()
    else:
        teams = Teams()
        games_in_week = list()
        for abbrev in teams.dataframes['abbreviation']:
            sch = Schedule(abbrev).dataframe
            sch['team'] = abbrev
            game = sch.loc[sch['week'] <= week]
            game = game.reset_index(drop=True)
            if not game.empty and game.loc[0]['points_scored'] is not None:
                games_in_week.append(game)
            time.sleep(5)
        if games_in_week:
            week_results = pd.concat(games_in_week)
        else:
            week_results = pd.DataFrame()
    return week_results
