import itertools
import json
import math

import pandas as pd
import numpy as np
from prettytable import PrettyTable
from scipy.optimize import minimize_scalar

from app import league_structure
from app import odds_helper
from app.helper import Helper


class Bettor:
    def __init__(self, team_df, individual_df, graph, gen_poisson_model):
        self.team_df = team_df
        self.individual_df = individual_df
        self.graph = graph
        self.gen_poisson_model = gen_poisson_model
        with open('resources/config.json', 'r') as f:
            self.config = json.load(f)

    def get_vegas_line(self, home_name, away_name, odds):
        matching_odds = [odd for odd in odds if (away_name in odd[0][0] or home_name in odd[0][0]) and
                         (away_name in odd[0][1] or home_name in odd[0][1])]
        if len(matching_odds) < 1:
            print('Odds not found for', away_name.ljust(self.config.get('justify_width')), '@', home_name)
            return 0
        else:
            matching_odds = matching_odds[0]
            return matching_odds[-1]

    def get_spread_chance(self, favorite, underdog, spread):
        if spread > 0:
            return

        helper = Helper(self.team_df, self.individual_df, self.graph, self.gen_poisson_model)
        favorite_dist = helper.get_dist_from_gen_poisson_model(favorite, underdog)
        underdog_dist = helper.get_dist_from_gen_poisson_model(underdog, favorite)

        max_points = self.config.get('betting_constants').get('max_possible_points')

        underdog_chances = underdog_dist.pmf([score for score in range(max_points)])
        favorite_chances = favorite_dist.pmf([score - spread for score in range(max_points)])
        favorite_cover_chances = favorite_dist.sf([score - spread for score in range(max_points)])
        # favorite_fail_chances = favorite_dist.cdf([score - spread for score in range(max_points)])

        cover_chances = underdog_chances * favorite_cover_chances
        push_chances = underdog_chances * favorite_chances
        # fail_chances = underdog_chances * (favorite_fail_chances - favorite_chances)

        cover_chance = sum(cover_chances)
        push_chance = sum(push_chances)
        fail_chance = 1 - cover_chance - push_chance
        return cover_chance, push_chance, fail_chance

    def get_over_chance(self, team1, team2, ou):
        helper = Helper(self.team_df, self.individual_df, self.graph, self.gen_poisson_model)
        team1_dist = helper.get_dist_from_gen_poisson_model(team1, team2)
        team2_dist = helper.get_dist_from_gen_poisson_model(team2, team1)

        team1_chances = team1_dist.pmf([score for score in range(math.ceil(ou))])
        team2_chances = team2_dist.pmf([score for score in range(math.ceil(ou))])

        combos = pd.DataFrame(itertools.product(range(math.ceil(ou)), range(math.ceil(ou))), columns=['Team1 Scores',
                                                                                                      'Team2 Scores'])
        combos['Chance'] = combos.apply(lambda r: team1_chances[r['Team1 Scores']] *
                                                  team2_chances[r['Team2 Scores']], axis=1)

        combo_grid = pd.pivot_table(data=combos, values='Chance', index='Team1 Scores', columns='Team2 Scores')
        combo_grid = np.flip(combo_grid.values, axis=0)

        under_chance = np.tril(combo_grid).sum()
        return 1 - under_chance

    def get_h2h_odds(self, team1, team2):
        h2h_odds = odds_helper.get_fanduel_odds(future_days=7, bet_type='h2h')
        moneyline = [game for game in h2h_odds if (team1 in game[0] and team2 in game[1]) or
                     (team2 in game[0] and team1 in game[1])]
        if moneyline:
            game = moneyline[0]
            if team1 in game[0]:
                team1_ml = game[-2]
                team2_ml = game[-1]
                team1_ml_chance = odds_helper.convert_american_to_probability(team1_ml)
                team2_ml_chance = odds_helper.convert_american_to_probability(team2_ml)
                return team1_ml_chance, team2_ml_chance
            else:
                team1_ml = game[-1]
                team2_ml = game[-2]
                team1_ml_chance = odds_helper.convert_american_to_probability(team1_ml)
                team2_ml_chance = odds_helper.convert_american_to_probability(team2_ml)
                return team1_ml_chance, team2_ml_chance
        return .5, .5

    def all_bets(self, pot, week):
        ats_odds = odds_helper.get_fanduel_odds(future_days=7)
        h2h_odds = odds_helper.get_fanduel_odds(future_days=7, bet_type='h2h')
        o_u_odds = odds_helper.get_fanduel_ou_odds(self.config, future_days=7)

        schedule = league_structure.load_schedule()
        week_games = schedule.get('weeks')[week - 1]

        bets = list()
        for game in ats_odds:
            home_team, away_team, home_spread, away_spread, home_american, away_american = game
            if not any(week_game.get('home') in home_team and
                       week_game.get('away') in away_team for week_game in week_games):
                continue

            home_spread = float(home_spread)
            away_spread = float(away_spread)

            home_team = home_team.split()[-1]
            away_team = away_team.split()[-1]

            favorite = home_team if home_spread < 0.0 else away_team
            underdog = away_team if home_spread < 0.0 else home_team

            favorite_spread = home_spread if home_spread < 0 else away_spread
            underdog_spread = away_spread if home_spread < 0 else home_spread

            favorite_american = home_american if home_spread < 0 else away_american
            underdog_american = away_american if home_spread < 0 else home_american

            cover_chance, push_chance, fail_chance = self.get_spread_chance(favorite, underdog, favorite_spread)

            favorite_chance = odds_helper.convert_american_to_probability(favorite_american)
            underdog_chance = odds_helper.convert_american_to_probability(underdog_american)

            favorite_payout = 1 / favorite_chance
            underdog_payout = 1 / underdog_chance

            expected_favorite_payout = favorite_payout * cover_chance + push_chance
            expected_underdog_payout = underdog_payout * fail_chance + push_chance

            amount = 1.5
            alpha = self.config.get('betting_constants').get('risk_tolerance')
            favorite_bet_pct = minimize_scalar(utility,
                                               amount,
                                               bounds=(0.0, 1),
                                               args=(cover_chance, favorite_payout, push_chance, fail_chance, 1, alpha))
            underdog_bet_pct = minimize_scalar(utility,
                                               amount,
                                               bounds=(0.0, 1),
                                               args=(fail_chance, underdog_payout, push_chance, cover_chance, 1, alpha))

            favorite_bet_amount = favorite_bet_pct.x * pot
            underdog_bet_amount = underdog_bet_pct.x * pot

            overround = favorite_chance + underdog_chance
            margin = overround - 1
            vig = margin / overround

            favorite_row = {'Team': favorite,
                            'Spread': favorite_spread,
                            'Opponent': underdog,
                            'American Odds': favorite_american,
                            'Implied Prob': favorite_chance,
                            'Decimal': favorite_payout,
                            'Vig': vig,
                            'Model Chance': cover_chance,
                            'Push Chance': push_chance,
                            'Expected Value': expected_favorite_payout,
                            'Bet Percent': favorite_bet_pct.x,
                            'Bet Amount': favorite_bet_amount,
                            'Bet Type': 'ATS'}

            underdog_row = {'Team': underdog,
                            'Spread': underdog_spread,
                            'Opponent': favorite,
                            'American Odds': underdog_american,
                            'Implied Prob': underdog_chance,
                            'Decimal': underdog_payout,
                            'Vig': vig,
                            'Model Chance': fail_chance,
                            'Push Chance': push_chance,
                            'Expected Value': expected_underdog_payout,
                            'Bet Percent': underdog_bet_pct.x,
                            'Bet Amount': underdog_bet_amount,
                            'Bet Type': 'ATS'}

            bets.append(favorite_row)
            bets.append(underdog_row)

        for game in h2h_odds:
            home_team, away_team, _, _, home_american, away_american = game
            if not any(week_game.get('home') in home_team and
                       week_game.get('away') in away_team for week_game in week_games):
                continue

            home_team = home_team.split()[-1]
            away_team = away_team.split()[-1]

            home_bt = self.team_df.at[home_team, 'Bayes BT']
            away_bt = self.team_df.at[away_team, 'Bayes BT']

            home_chance = odds_helper.convert_american_to_probability(home_american)
            away_chance = odds_helper.convert_american_to_probability(away_american)

            home_payout = 1 / home_chance
            away_payout = 1 / away_chance

            home_bt_chance = math.exp(home_bt) / (math.exp(home_bt) + math.exp(away_bt))
            away_bt_chance = math.exp(away_bt) / (math.exp(home_bt) + math.exp(away_bt))

            expected_home_payout = home_payout * home_bt_chance
            expected_away_payout = away_payout * away_bt_chance

            amount = 1.5
            alpha = self.config.get('betting_constants').get('risk_tolerance')
            home_bet_pct = minimize_scalar(utility,
                                           amount,
                                           bounds=(0.0, 1),
                                           args=(home_bt_chance, home_payout, 0, 1 - home_bt_chance, 1, alpha))
            away_bet_pct = minimize_scalar(utility,
                                           amount,
                                           bounds=(0.0, 1),
                                           args=(away_bt_chance, away_payout, 0, 1 - away_bt_chance, 1, alpha))

            home_bet_amount = home_bet_pct.x * pot
            away_bet_amount = away_bet_pct.x * pot

            overround = home_chance + away_chance
            margin = overround - 1
            vig = margin / overround

            home_row = {'Team': home_team,
                        'Spread': 0,
                        'Opponent': away_team,
                        'American Odds': home_american,
                        'Implied Prob': home_chance,
                        'Decimal': home_payout,
                        'Vig': vig,
                        'Model Chance': home_bt_chance,
                        'Push Chance': 0,
                        'Expected Value': expected_home_payout,
                        'Bet Percent': home_bet_pct.x,
                        'Bet Amount': home_bet_amount,
                        'Bet Type': 'H2H'}

            away_row = {'Team': away_team,
                        'Spread': 0,
                        'Opponent': home_team,
                        'American Odds': away_american,
                        'Implied Prob': away_chance,
                        'Decimal': away_payout,
                        'Vig': vig,
                        'Model Chance': away_bt_chance,
                        'Push Chance': 0,
                        'Expected Value': expected_away_payout,
                        'Bet Percent': away_bet_pct.x,
                        'Bet Amount': away_bet_amount,
                        'Bet Type': 'H2H'}

            bets.append(home_row)
            bets.append(away_row)

        for game in o_u_odds:
            home_team, away_team, ou, over_american, under_american = game
            if not any(week_game.get('home') in home_team and
                       week_game.get('away') in away_team for week_game in week_games):
                continue

            home_team = home_team.split()[-1]
            away_team = away_team.split()[-1]

            over_chance = odds_helper.convert_american_to_probability(over_american)
            under_chance = odds_helper.convert_american_to_probability(under_american)

            over_payout = 1 / over_chance
            under_payout = 1 / under_chance

            over_dist_chance = self.get_over_chance(home_team, away_team, ou)
            under_dist_chance = 1 - over_dist_chance

            expected_over_payout = over_payout * over_dist_chance
            expected_under_payout = under_payout * under_dist_chance

            amount = 1.5
            alpha = self.config.get('betting_constants').get('risk_tolerance')
            over_bet_pct = minimize_scalar(utility,
                                           amount,
                                           bounds=(0.0, 1),
                                           args=(over_dist_chance, over_payout, 0, under_dist_chance, 1, alpha))
            under_bet_pct = minimize_scalar(utility,
                                            amount,
                                            bounds=(0.0, 1),
                                            args=(under_dist_chance, under_payout, 0, over_dist_chance, 1, alpha))

            over_bet_amount = over_bet_pct.x * pot
            under_bet_amount = under_bet_pct.x * pot

            overround = over_chance + under_chance
            margin = overround - 1
            vig = margin / overround

            over_row = {'Team': home_team,
                        'O/U': 'o' + str(ou),
                        'Opponent': away_team,
                        'American Odds': over_american,
                        'Implied Prob': over_chance,
                        'Decimal': over_payout,
                        'Vig': vig,
                        'Model Chance': over_dist_chance,
                        'Push Chance': 0,
                        'Expected Value': expected_over_payout,
                        'Bet Percent': over_bet_pct.x,
                        'Bet Amount': over_bet_amount,
                        'Bet Type': 'O/U'}

            under_row = {'Team': away_team,
                         'O/U': 'u' + str(ou),
                         'Opponent': home_team,
                         'American Odds': under_american,
                         'Implied Prob': under_chance,
                         'Decimal': under_payout,
                         'Vig': vig,
                         'Model Chance': under_dist_chance,
                         'Push Chance': 0,
                         'Expected Value': expected_under_payout,
                         'Bet Percent': under_bet_pct.x,
                         'Bet Amount': under_bet_amount,
                         'Bet Type': 'O/U'}

            bets.append(over_row)
            bets.append(under_row)

        bet_df = pd.DataFrame(bets)
        bet_df = bet_df.sort_values(by='Bet Percent', ascending=False)
        remaining_pot = pot
        for index, row in bet_df.iterrows():
            bet_amount = bet_df.at[index, 'Bet Percent'] * remaining_pot
            bet_df.at[index, 'Bet Amount'] = bet_amount
            remaining_pot = remaining_pot - bet_amount

        bet_df['Expected Return'] = bet_df.apply(lambda r: r['Bet Amount'] * r['Expected Value'], axis=1)
        bet_df['Expected Profit'] = bet_df.apply(lambda r: r['Expected Return'] - r['Bet Amount'], axis=1)
        bet_df['To Win'] = bet_df.apply(lambda r: r['Bet Amount'] * r['Decimal'] - r['Bet Amount'], axis=1)

        bet_df['Swing'] = bet_df.apply(lambda r: r['Bet Amount'] + r['To Win'], axis=1)
        bet_df = bet_df.sort_values(by='Swing', ascending=False)

        good_bet_df = bet_df.loc[bet_df['Expected Value'] > 1].reset_index(drop=True)
        bad_bet_df = bet_df.loc[bet_df['Expected Value'] <= 1].reset_index(drop=True)

        print_bet_table(good_bet_df, good_bets=True)
        print_bet_table(bad_bet_df, good_bets=False)


def utility(amount, pos_chance, pos_payout, push_chance, neg_chance, pot, alpha):
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    def crra_utility(x):
        # Risk Averse:
        #   alpha > 1
        # Risk Neutral:
        #   alpha = 1
        # Risk Seeking:
        #   alpha < 1
        if alpha == 1:
            return math.log(x)
        return (math.pow(x, 1 - alpha) - 1) / (1 - alpha)

    u = pos_chance * crra_utility(pot + (amount * pos_payout) - amount) + \
        push_chance * crra_utility(pot) + \
        neg_chance * crra_utility(pot - amount)
    return -u


def print_bet_table(df, good_bets=True):
    columns = ['Num', 'Team', 'Spread', 'O/U', 'Opponent', 'American Odds', 'Implied Prob', 'Decimal', 'Vig',
               'Model Chance', 'Push Chance', 'Bet Percent', 'Bet Amount', 'Expected Value', 'Expected Return',
               'Expected Profit', 'To Win']

    table = PrettyTable(columns)
    table.float_format = '0.3'

    row_num = 1
    for index, row in df.iterrows():
        table_row = list()
        table_row.append(str(row_num))
        row_num = row_num + 1
        for col in columns[1:]:
            if col == 'American Odds':
                val = '+' + str(row[col]) if row[col] > 0 else str(row[col])
            elif col == 'Spread':
                val = '--' if row['Bet Type'] != 'ATS' else str(row[col])
            elif col == 'O/U':
                val = '--' if row['Bet Type'] != 'O/U' else str(row[col])
            elif col == 'Decimal' or col == 'Expected Value':
                val = str(round(row[col], 2)) + 'x'
            elif col == 'Implied Prob' or \
                    col == 'Model Chance' or \
                    col == 'Push Chance' or \
                    col == 'Bet Percent' or \
                    col == 'Vig':
                val = f'{row[col] * 100:.3f}' + '%'
            elif col == 'Bet Amount' or col == 'Expected Return' or col == 'Expected Profit' or col == 'To Win':
                val = 0 if col == 'Expected Profit' and -.01 < row[col] < 0 else row[col]
                val = '${:,.2f}'.format(val)
            elif col == 'Expected Value':
                val = str(round(row[col], 2))
            else:
                val = str(row[col])
            table_row.append(val)

        table.add_row(table_row)

    # Print the table
    with open('resources/config.json', 'r') as f:
        config = json.load(f)

    output_path = config.get('output_locations').get('good_bets') if good_bets \
        else config.get('output_locations').get('bad_bets')
    with open(output_path, 'w') as f:
        f.write(str(table))
        f.close()
