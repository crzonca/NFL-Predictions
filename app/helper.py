import itertools
import json
import math
import os
import pickle
import shutil
import statistics
import warnings

import choix
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score


def get_bayes_avg(prior_avg, prior_var, sample_avg, sample_var, n):
    k_0 = sample_var / prior_var
    if k_0 == 0 or n == 0:
        return prior_avg
    posterior_avg = ((k_0 / (k_0 + n)) * prior_avg) + ((n / (k_0 + n)) * sample_avg)
    return posterior_avg


def get_color(value, mean=0, variance=np.nan, alpha=.05, enabled=True, invert=False):
    if not enabled:
        return value

    green = '\033[32m'
    red = '\033[31m'
    stop = '\033[0m'

    if pd.isna(variance):
        value = round(value, 3)
        return red + str(value) + stop if value < 0 else green + str(value) + stop if value > 0 else value
    elif variance == 0:
        return ''
    else:
        normal = norm(mean, math.sqrt(variance))
        if (not invert and normal.ppf(alpha) > value) or (invert and normal.ppf(1 - alpha) < value):
            return red
        if (not invert and normal.ppf(1 - alpha) < value) or (invert and normal.ppf(alpha) > value):
            return green
        return ''


class Helper:
    def __init__(self, team_df, individual_df, graph, gen_poisson_model):
        self.team_df = team_df
        self.individual_df = individual_df
        self.graph = graph
        self.gen_poisson_model = gen_poisson_model
        with open('resources/config.json', 'r') as f:
            self.config = json.load(f)

    def load_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(self.config.get('resource_locations').get('model'), 'rb') as f:
                clf = pickle.load(f)

                return clf

    def load_pt(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(self.config.get('resource_locations').get('power_transformer'), 'rb') as f:
                pt = pickle.load(f)

                return pt

    def predict_drives_team_averages(self, team1, team2):
        average_drives = self.config.get('preseason_values').get('preseason_avg_drives')
        if team1 in list(self.individual_df['Team']):
            if team2 in list(self.individual_df['Team']):
                relevant_df = self.individual_df.loc[(self.individual_df['Team'] == team1) |
                                                     (self.individual_df['Team'] == team2)]
                if not relevant_df.empty:
                    average_drives = relevant_df['Drives'].mean()
        else:
            if not self.individual_df.empty:
                average_drives = self.individual_df['Drives'].mean()
        return average_drives

    def predict_drives(self, team1, team2):
        if not self.team_df.at[team1, 'Drives Model Good']:
            return self.predict_drives_team_averages(team1, team2)

        intercept = self.team_df['Drives Intercept'].mean()
        team1_off_coef = self.team_df.at[team1, 'Off Drives Coef']
        team1_def_coef = self.team_df.at[team1, 'Def Drives Coef']
        team2_off_coef = self.team_df.at[team2, 'Off Drives Coef']
        team2_def_coef = self.team_df.at[team2, 'Def Drives Coef']

        team1_drives = intercept + team1_off_coef + team2_def_coef
        team2_drives = intercept + team2_off_coef + team1_def_coef
        average_drives = (team1_drives + team2_drives) / 2
        return average_drives

    def get_dist_from_gen_poisson_model(self, team1, team2, predicted_drives=None):
        if not predicted_drives:
            predicted_drives = self.predict_drives(team1, team2)

        explanatory = pd.get_dummies(self.individual_df[['Team', 'Opponent']], dtype=int)
        explanatory = sm.add_constant(explanatory)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            prediction_series = pd.Series(index=explanatory.columns)
            prediction_series.at['const'] = 1.0
            prediction_series.at['Team_' + team1] = 1.0
            prediction_series.at['Opponent_' + team2] = 1.0
            prediction_series = prediction_series.fillna(0.0)
            dist = self.gen_poisson_model.get_distribution(prediction_series, exposure=predicted_drives)
        return dist

    def predict_score_from_gen_poisson_model(self, team1, team2, team1_drives=None, team2_drives=None):
        team1_dist = self.get_dist_from_gen_poisson_model(team1, team2, team1_drives)
        team2_dist = self.get_dist_from_gen_poisson_model(team2, team1, team2_drives)

        # TODO Possibly the following code
        # self.gen_poisson_model.predict(prediction_series, exposure=predicted_drives)

        return float(team1_dist.mean()), float(team2_dist.mean())

    def gen_poisson_to_sim_skellam_pmf(self, team1_dist, team2_dist, x):
        max_points = self.config.get('betting_constants').get('max_possible_points')
        team2_score_chances = team2_dist.pmf([score for score in range(max_points)])
        team1_score_chances = team1_dist.pmf([score + x for score in range(max_points)])
        return sum(team1_score_chances * team2_score_chances)

    def score_grid(self, team1, team2):
        min_drives = 7
        max_drives = 16

        grid_range = list(range(min_drives, max_drives + 1))
        grid = pd.DataFrame(index=grid_range, columns=grid_range)
        for drives1 in grid_range:
            for drives2 in grid_range:
                score1, score2 = self.predict_score_from_gen_poisson_model(team1, team2, drives1, drives2)
                grid.at[drives2, drives1] = str(int(score1)) + ' - ' + str(int(score2))
                grid.at[drives1, drives2] = str(int(score2)) + ' - ' + str(int(score1))

        return grid

    def get_bayes_avg_wins(self, game_df, team_name):
        matching_games = game_df.loc[game_df['Team'] == team_name]

        prior_avg = self.config.get('bayes_priors').get('bayes_wins_prior_mean')
        prior_var = self.config.get('bayes_priors').get('bayes_wins_prior_var')

        wins = list(matching_games['Win'])
        if len(wins) < 2:
            return prior_avg

        win_pct = statistics.mean(wins)
        win_var = statistics.variance(wins)

        return get_bayes_avg(prior_avg, prior_var, win_pct, win_var, len(wins))

    def get_bayes_avg_points(self, game_df, team_name, allowed=False):
        matching_games = game_df.loc[game_df['Team'] == team_name]

        prior_avg = self.config.get('bayes_priors').get('bayes_points_prior_mean')
        prior_var = self.config.get('bayes_priors').get('bayes_points_prior_var')

        points = list(matching_games['Points Allowed']) if allowed else list(matching_games['Points'])
        if len(points) < 2:
            return prior_avg

        avg_points = statistics.mean(points)
        points_var = statistics.variance(points)

        return get_bayes_avg(prior_avg, prior_var, avg_points, points_var, len(points))

    def get_bradley_terry_from_graph(self):
        nodes = self.graph.nodes
        df = pd.DataFrame(nx.to_numpy_array(self.graph), columns=nodes)
        df.index = nodes

        teams = list(df.index)
        df = df.fillna(0)

        teams_to_index = {team: i for i, team in enumerate(teams)}
        index_to_teams = {i: team for team, i in teams_to_index.items()}

        graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
        edges = [list(itertools.repeat((teams_to_index.get(team2),
                                        teams_to_index.get(team1)),
                                       int(weight_dict.get('weight'))))
                 for team1, team2, weight_dict in graph.edges.data()]
        edges = list(itertools.chain.from_iterable(edges))

        if not edges:
            coef_df = pd.DataFrame(columns=['BT', 'Var'], index=self.team_df.index)
            coef_df['BT'] = coef_df['BT'].fillna(0)
            coef_df['Var'] = coef_df['Var'].fillna(1)
            return coef_df

        coeffs, cov = choix.ep_pairwise(n_items=len(teams), data=edges, alpha=1)
        coeffs = pd.Series(coeffs)
        cov = pd.Series(cov.diagonal())
        coef_df = pd.DataFrame([coeffs, cov]).T
        coef_df.columns = ['BT', 'Var']
        coef_df.index = [index_to_teams.get(index) for index in coef_df.index]
        coef_df = coef_df.sort_values(by='BT', ascending=False)
        return coef_df

    def get_tiers(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            bts = self.team_df['Bayes BT']
            x = bts.values.reshape(-1, 1)

            tier_count_dict = dict()
            for potential_num_tiers in range(self.config.get('plotting_values').get('min_team_tiers'),
                                             self.config.get('plotting_values').get('max_team_tiers')):
                k_means = KMeans(n_clusters=potential_num_tiers).fit(x)
                tier_count_dict[potential_num_tiers] = calinski_harabasz_score(x, k_means.labels_)

            sorted_tier_counts = list(sorted(tier_count_dict.items(), key=lambda t: t[1], reverse=True))
            num_tiers = sorted_tier_counts[0][0]
            k_means = KMeans(n_clusters=num_tiers).fit(x)

            self.team_df['Cluster'] = k_means.labels_

            cluster_averages = self.team_df.groupby('Cluster').mean(numeric_only=True)
            cluster_averages = cluster_averages.sort_values(by='Bayes BT', ascending=False)
            tiers = {cluster: tier for cluster, tier in zip(cluster_averages.index, range(num_tiers, 0, -1))}
            self.team_df['Tier'] = self.team_df['Cluster'].map(tiers)

            return {team: row['Tier'] for team, row in self.team_df.iterrows()}

    def delete_matchup_folder(self):
        folder = self.config.get('output_locations').get('matchups')
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def get_common_score(self, team1, team2):
        expected_poisson = self.get_dist_from_gen_poisson_model(team1, team2)

        empirical_score_df = pd.read_csv(self.config.get('resource_locations').get('empirical_scores'))

        empirical_score_df['poisson'] = empirical_score_df.apply(lambda r: expected_poisson.pmf(r['Score']).squeeze(), axis=1)
        empirical_score_df['posterior'] = empirical_score_df.apply(lambda r: r['poisson'] * r['Pct'], axis=1)

        common_score = empirical_score_df['posterior'].idxmax()
        return common_score
