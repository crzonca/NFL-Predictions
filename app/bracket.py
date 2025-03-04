import math
from scipy.stats import binom
from app.betting import Bettor


class Bracket:
    def __init__(self, label, team_df, individual_df, graph, gen_poisson_model, left=None, right=None, bt=None, series_length=1):
        self.reach_chances = []
        self.is_leaf = left is None and right is None
        self.label = label
        self.left = left
        self.right = right
        self.bt = bt
        self.series_length = series_length
        self.team_df = team_df
        self.individual_df = individual_df
        self.graph = graph
        self.gen_poisson_model = gen_poisson_model

    def size(self):
        return 1 if self.is_leaf else self.left.size() + self.right.size()

    def get_leaves(self):
        return [self] if self.is_leaf else self.left.get_leaves() + self.right.get_leaves()

    def get_possible_opponents(self, label):
        left_side = self.left.get_leaves()
        left_side = [] if label in [b.label for b in left_side] else left_side
        right_side = self.right.get_leaves()
        right_side = [] if label in [b.label for b in right_side] else right_side

        return left_side + right_side

    def get_victory_chance(self, label):
        # Get all possible teams that can reach this point in the bracket
        leaf_labels = {b.label for b in self.get_leaves()}

        # If the team in question is not one of them, they have a 0% chance
        if label not in leaf_labels:
            return 0

        # If the team in question is the only one, they have a 100% chance
        if self.is_leaf:
            return 1

        # The teams chance to reach this point of the bracket is their chance to win the previous point
        reach_chance = self.left.get_victory_chance(label) + self.right.get_victory_chance(label)

        team_leaf = [b for b in self.get_leaves() if b.label == label][0]
        possible_victory_chances = []
        for other_team in self.get_possible_opponents(label):
            other_label = other_team.label
            team_chance, opp_chance = self.get_best_of(team_leaf.label, other_team.label, self.series_length)
            # team_chance, opp_chance = get_best_of_bt(team_leaf.bt, other_team.bt, self.series_length)
            opp_reach_chance = self.left.get_victory_chance(other_label) + self.right.get_victory_chance(other_label)

            possible_victory_chances.append(team_chance * opp_reach_chance)

        victory_chance = reach_chance * sum(possible_victory_chances)

        return victory_chance

    def get_best_of(self, team1, team2, series_length):
        bets = Bettor(self.team_df, self.individual_df, self.graph, self.gen_poisson_model)
        win_chance, tie_chance, loss_chance = bets.get_spread_chance(team1, team2, 0.0)
        p = win_chance / (1 - tie_chance)
        required_wins = int(math.ceil(series_length / 2.0))

        team1_chance = binom.cdf(series_length - required_wins, series_length, 1 - p)
        team2_chance = binom.cdf(series_length - required_wins, series_length, p)

        return team1_chance, team2_chance


def get_best_of_bt(bt1, bt2, series_length):
    p = math.exp(bt1) / (math.exp(bt1) + math.exp(bt2))
    required_wins = int(math.ceil(series_length / 2.0))

    team1_chance = binom.cdf(series_length - required_wins, series_length, 1 - p)
    team2_chance = binom.cdf(series_length - required_wins, series_length, p)

    return team1_chance, team2_chance
