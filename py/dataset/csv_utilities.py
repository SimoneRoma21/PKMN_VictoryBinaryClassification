import pandas as pd
import json


def open_pkmn_database_csv() -> pd.DataFrame:
    """
    Opens the pokemon database calculated from the train json
    and returns the opened json as a pandas dataframe
    """

    pkmn_db = pd.read_csv("../data/pkmn_database.csv")
    pkmn_db = pkmn_db.drop("Unnamed: 0", axis=1)
    return pkmn_db


def open_train_json() -> list:
    """
    Opens the train dataset json and returns the list
    of rows/lines seen in the dataset
    """
    list = []
    with open("../data/train.jsonl", "r") as f:
        for line in f:
            list.append(json.loads(line))
    list.remove(list[4877])  # dropping row 4877 due to bad format (confirmed in class)
    return list


def open_type_chart_json() -> pd.DataFrame:
    """
    Opens the chart json of pokemon types to be able to compute
    weaknesses of teams and pokemons, return a dataframe with
    multipliers
    """
    with open("../data/type_chart.json", "r") as f:
        data = json.load(f)
    return pd.DataFrame(data).transpose()
