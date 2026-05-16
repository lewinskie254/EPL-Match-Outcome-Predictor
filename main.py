import json 
import xgboost as xgb 
import os 
import requests 
from dotenv import load_dotenv
import pandas as pd 
from datetime import datetime
import numpy as np 
from difflib import get_close_matches
import argparse


load_dotenv()
model =  xgb.XGBClassifier() 
model.load_model('xgb_model.json')
API_KEY = os.getenv("API_KEY")

with open('elo.json', 'r') as f:
    elo = json.load(f)


def getElo(team):
    return elo[team]

def build_team_history(df):
    home = df[[
        "MatchDate", "Season", "HomeTeam", "AwayTeam",
        "FullTimeHomeGoals", "FullTimeAwayGoals",
        "HomePoints"
    ]].copy()

    home.columns = ["MatchDate", "Season", "Team", "Opponent",
                    "GoalsFor", "GoalsAgainst", "Points"]

    away = df[[
        "MatchDate", "Season", "AwayTeam", "HomeTeam",
        "FullTimeAwayGoals", "FullTimeHomeGoals",
        "AwayPoints"
    ]].copy()

    away.columns = ["MatchDate", "Season", "Team", "Opponent",
                    "GoalsFor", "GoalsAgainst", "Points"]

    return pd.concat([home, away], ignore_index=True)

def get_last_5(team_history, team, match_date):
    history = team_history[team_history["Team"] == team]

    past = history[history["MatchDate"] < match_date]
    data = past.sort_values("MatchDate", ascending=False).head(5)
    return data['Points'].mean()


def match_team(name, elo_keys):
    match = get_close_matches(name, elo_keys, n=1, cutoff=0.5)
    return match[0] if match else None


if __name__ == "__main__": 

    parser = argparse.ArgumentParser()

    parser.add_argument("-home", type=str, required=True)
    parser.add_argument("-away", type=str, required=True)
    parser.add_argument("-date", type=str, required=True)

    args = parser.parse_args()

    home_team = args.home
    away_team = args.away
    date_str = datetime.strptime(args.date, "%Y-%m-%d")

    df = pd.read_csv("epl.csv")
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])

    team_history = build_team_history(df)

    teams = list(elo.keys())

    home = get_close_matches(home_team, teams)[0]
    away = get_close_matches(away_team, teams)[0]

    print("-" * 50)
    print()
    print(f"{home} Vs. {away}: ")
    print()
    print("-" * 50)


    date = "2026-05-26" 
    dt = datetime.strptime(date, "%Y-%m-%d") 

    last5_home = get_last_5(
        team_history,
        home,
        dt
    )

    last5_away = get_last_5(
        team_history,
        away, 
        dt
    )

    # features = ['EloDiff', 'HomeElo', 'AwayElo', 'HomeFormPoints_5', 'AwayFormPoints_5', 'SeasonEncoded', 'SeasonYear', 'Month', 'Date', 'Target']

    features = np.array([
        elo[home] - elo[away], 
        elo[home], 
        elo[away], 
        last5_home, 
        last5_away, 
        25, 
        2025,
        5, 
        17
    ]).reshape(1, -1)

    pred = model.predict_proba(features)

    win = pred[0][0] * 100
    draw = pred[0][1] * 100
    loss = pred[0][2] * 100
    results = {"Win": f"{win:.2f}%", "Draw": f"{draw:.2f}%", "Loss": f"{loss:.2f}%"}

    final = json.dumps(results, indent=4)
    print(final)