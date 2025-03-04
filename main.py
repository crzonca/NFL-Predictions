import json
from datetime import datetime

from app import season as nfl

if __name__ == '__main__':
    with open('resources/config.json', 'r') as f:
        config = json.load(f)

    season_start = config.get('season_start_date')
    season_start = datetime.strptime(season_start, '%Y-%m-%d')
    today = datetime.today()

    delta = today - season_start
    days = delta.days
    week = int(days / 7) + 1
    if week <= 0:
        week = 1
    if week < config.get('total_weeks') + 6:
        nfl.season(week,
                   manual_odds=config.get('manual_odds'),
                   include_parity=config.get('include_parity'))
