import statistics

import maya
import requests
from datetime import datetime, timedelta, timezone
import json

domain = 'https://api.the-odds-api.com'
with open('resources/config.json', 'r') as f:
    config = json.load(f)


def get_odds():
    # Request the odds for the games
    now = datetime.now(timezone.utc).replace(microsecond=0)
    next_week = now + timedelta(days=8)
    params = {'apiKey': config.get('betting_constants').get('odds_api_key'),
              'regions': 'us',
              'commenceTimeFrom': now.isoformat().replace('+00:00', 'Z'),
              'commenceTimeTo': next_week.isoformat().replace('+00:00', 'Z'),
              'markets': 'spreads'}
    req = requests.get(domain + '/v4/sports/americanfootball_nfl/odds/', params=params)

    # Get info on request usages remaining
    requests_remaining = req.headers.get('x-requests-remaining')
    if float(requests_remaining) < 5:
        print(requests_remaining + ' Requests remaining')

    # Get the odds data
    data = req.json()

    # Verify the status of getting the spread data
    if not data:
        raise Exception('Unable to fetch spread data')

    # For each game in the odds data
    games = list()
    for game in data:
        # Get the teams
        # teams = game.get('teams')
        # first_team = teams[0]
        home_team = game.get('home_team')
        away_team = game.get('away_team')

        # Get the time of the game
        game_time = game.get('commence_time')
        game_time = maya.when(str(game_time), timezone='US/Mountain')

        # Skip games that are after the week end date
        # if game_time > week_end_date:
        #     continue

        # For each odds site
        bookmakers = game.get('bookmakers')
        home_spreads = list()
        for bookmaker in bookmakers:
            # Get when the odds were last updated
            odds_updated_time = bookmaker.get('last_update')
            odds_updated_time = maya.when(str(odds_updated_time), timezone='US/Mountain')

            # If the odds weren't updated in the last day
            if odds_updated_time < maya.now().add(days=-1):
                # Skip it
                continue

            markets = bookmaker.get('markets')
            for market in markets:
                if market.get('key') == 'spreads':
                    outcomes = market.get('outcomes')
                    outcomes = [outcome for outcome in outcomes if outcome.get('name') == home_team]
                    if outcomes:
                        home_spreads.append(float(outcomes[0].get('point')))

        # Get the average home spread
        if home_spreads:
            home_spread = statistics.mean(home_spreads)
        else:
            home_spread = 0

        games.append(((home_team, away_team), game_time, home_spread))
    return games


def get_fanduel_odds(future_days=7, bet_type='spreads'):
    # Request the odds for the games
    # params = {'apiKey': '60cf84589508c25d9471beafbf3c3201',
    params = {'apiKey': '96f87732ce15ffb4a049122ce389599c',
              'sport': 'americanfootball_nfl',
              'region': 'us',
              'mkt': bet_type}
    req = requests.get(domain + '/v3/odds/', params=params)

    # Get info on request usages remaining
    requests_remaining = req.headers.get('x-requests-remaining')
    if float(requests_remaining) < 5:
        print(requests_remaining + ' Requests remaining')

    # Get the odds data
    resp = req.json()

    # Verify the status of getting the spread data
    if not resp.get('success'):
        raise Exception('Unable to fetch spread data')

    data = resp.get('data')

    # For each game in the odds data
    games = list()
    for game in data:
        # Get the teams
        teams = game.get('teams')
        first_team = teams[0]
        home_team = game.get('home_team')
        away_team = [team for team in teams if team != home_team][0]

        # Get the time of the game
        game_time = game.get('commence_time')
        game_time = maya.when(str(game_time), timezone='US/Central')

        current_time = maya.now()

        if current_time.add(days=future_days) < game_time:
            continue

        home_spread = 0.0
        home_odds = 100
        away_spread = 0.0
        away_odds = 100

        for site in game.get('sites'):
            if site.get('site_key') != 'fanduel':
                continue

            # Get when the odds were last updated
            odds_updated_time = site.get('last_update')
            odds_updated_time = maya.when(str(odds_updated_time), timezone='US/Central')

            # If the odds weren't updated in the last day
            if odds_updated_time < maya.now().add(days=-1):
                # Skip it
                continue

            # Get the point spreads for the game
            odds = site.get('odds')

            if bet_type == 'spreads':
                spreads = odds.get('spreads')
                points = spreads.get('points')
                odds = spreads.get('odds')

                # Get the spread from the home team perspective
                if first_team == home_team:
                    home_spread = points[0]
                    home_odds = odds[0]
                    away_spread = points[1]
                    away_odds = odds[1]
                else:
                    home_spread = points[1]
                    home_odds = odds[1]
                    away_spread = points[0]
                    away_odds = odds[0]
            elif bet_type == 'h2h':
                h2h = odds.get('h2h')
                if first_team == home_team:
                    home_odds = h2h[0]
                    away_odds = h2h[1]
                else:
                    home_odds = h2h[1]
                    away_odds = h2h[0]

        home_probability = 1 / home_odds
        away_probability = 1 / away_odds

        home_american = convert_probability_to_american(home_probability)
        away_american = convert_probability_to_american(away_probability)

        games.append((home_team, away_team, home_spread, away_spread, home_american, away_american))
    return games


def get_fanduel_ou_odds(config, future_days=7):
    # Request the odds for the games
    # params = {'apiKey': '60cf84589508c25d9471beafbf3c3201',
    params = {'apiKey': '96f87732ce15ffb4a049122ce389599c',
              'sport': 'americanfootball_nfl',
              'region': 'us',
              'mkt': 'totals'}
    req = requests.get(domain + '/v3/odds/', params=params)

    # Get info on request usages remaining
    requests_remaining = req.headers.get('x-requests-remaining')
    if float(requests_remaining) < 5:
        print(requests_remaining + ' Requests remaining')

    # Get the odds data
    resp = req.json()

    # Verify the status of getting the spread data
    if not resp.get('success'):
        raise Exception('Unable to fetch spread data')

    data = resp.get('data')

    # For each game in the odds data
    games = list()
    for game in data:
        # Get the teams
        teams = game.get('teams')
        first_team = teams[0]
        home_team = game.get('home_team')
        away_team = [team for team in teams if team != home_team][0]

        all_teams = config.get('teams')
        home_team = [t for t in all_teams if t in home_team][0]
        away_team = [t for t in all_teams if t in away_team][0]

        # Get the time of the game
        game_time = game.get('commence_time')
        game_time = maya.when(str(game_time), timezone='US/Central')

        current_time = maya.now()

        if current_time.add(days=future_days) < game_time:
            continue

        ou = 0.0
        over_odds = 100
        under_odds = 100

        for site in game.get('sites'):
            if site.get('site_key') != 'fanduel':
                continue

            # Get when the odds were last updated
            odds_updated_time = site.get('last_update')
            odds_updated_time = maya.when(str(odds_updated_time), timezone='US/Central')

            # If the odds weren't updated in the last day
            if odds_updated_time < maya.now().add(days=-1):
                # Skip it
                continue

            # Get the point spreads for the game
            odds = site.get('odds')

            totals = odds.get('totals')
            over_pos = 0 if totals.get('position') == 'over' else 1
            under_pos = 1 if totals.get('position') == 'over' else 0

            over_odds = totals.get('odds')[over_pos]
            under_odds = totals.get('odds')[under_pos]
            ou = float(totals.get('points')[over_pos])

        over_probability = 1 / over_odds
        under_probability = 1 / under_odds

        over_american = convert_probability_to_american(over_probability)
        under_american = convert_probability_to_american(under_probability)

        games.append((home_team, away_team, ou, over_american, under_american))
    return games


def convert_american_to_probability(american):
    if american < 0:
        probability = american / (american - 100)
    else:
        probability = 100 / (american + 100)
    return probability


def convert_probability_to_american(probability):
    if probability < .5:
        american = (100 / probability) - 100
    else:
        try:
            american = 100 * probability / (probability - 1)
        except ZeroDivisionError:
            american = 9900
    return round(american)
