import pandas as pd
import numpy as np

from . import csv_utilities as csv_u


def extract_all_pokemon_p1_teams(dataset) -> pd.DataFrame:
    """
    Extracts all pokemon teams of p1 from all games
    and returns a dataframe of them
    """
    # extracting all p1 teams from dataset
    db_pkmn_p1 = pd.DataFrame(
        [team for game in dataset for team in game["p1_team_details"]]
    )
    db_pkmn_p1.drop_duplicates(subset=["name"], inplace=True)  # dropping duplicates

    # extracting types for p1 teams
    db_types_p1 = pd.concat(
        [extract_types_from_team_p1(game) for game in dataset]
    ).drop_duplicates(subset="name", keep="first")
    db_pkmn_p1 = db_pkmn_p1.merge(
        db_types_p1, how="inner", on="name"
    )  # merging pokemon and types of p1
    return db_pkmn_p1


def extract_all_pokemon_p2_seen(dataset) -> pd.DataFrame:
    """
    Extracts all pokemon of p2 seen from all games
    and returns a dataframe of them
    """
    # extracting all p2 seens pokemons
    db_pkmn_p2_battles = pd.DataFrame(
        [
            elem["p2_pokemon_state"]["name"]
            for game in dataset
            for elem in game["battle_timeline"]
        ]
    )

    # dropping duplicates and renaming columns
    db_pkmn_p2_battles.drop_duplicates(inplace=True)
    db_pkmn_p2_battles.rename(columns={0: "name"}, inplace=True)
    return db_pkmn_p2_battles


def extract_all_pokemon_p2_lead(dataset, duplicates) -> pd.DataFrame:
    """
    Extracts all pokemon lead of p2 from all games
    and returns a dataframe of them
    """

    # getting all p2 leads
    db_pkmn_p2_lead = pd.DataFrame([game["p2_lead_details"] for game in dataset])
    if not (duplicates):  # admitting duplicates or not
        db_pkmn_p2_lead.drop_duplicates(subset=["name"], inplace=True)
    return db_pkmn_p2_lead


def extract_all_pokemon_p2(dataset) -> pd.DataFrame:
    """
    Extracts all pokemon teams of p2 from all games
    and returns a dataframe of them
    """

    # picking all pokemons seen in all battles of p2
    db_pkmn_p2_battles = extract_all_pokemon_p2_seen(dataset)
    # picking all pokemon leads of p2 (has to be subset of db_pkmn_p2_battles)
    db_pkmn_p2_lead = extract_all_pokemon_p2_lead(dataset, False)
    # merging the two dataset
    db_pkmn_p2 = db_pkmn_p2_lead.merge(db_pkmn_p2_battles, how="inner", on=["name"])

    db_types_p2 = pd.concat(
        [extract_types_from_team_p2(game) for game in dataset]
    ).drop_duplicates(subset="name", keep="first")
    db_pkmn_p2 = db_pkmn_p2.merge(db_types_p2, how="inner", on="name")

    return db_pkmn_p2


def extract_p1_team_from_game_start(game) -> pd.Series:
    """
    Extracts all pokemon teams of p1 at the start of all games
    and returns a dataframe of them
    """
    return pd.DataFrame(game["p1_team_details"])["name"]


def extract_p1_team_from_game_last(game) -> pd.Series:
    """
    Extracts all pokemon teams of p1 at the end of the 30 turns
    for all games and returns a dataframe of them
    """

    turns = pd.DataFrame([turn["p1_pokemon_state"] for turn in game["battle_timeline"]])
    # picking dead pokemons --> it can  be optimized
    pkmn_dead_p1 = turns[turns["status"] == "fnt"]["name"].drop_duplicates(keep="last")

    # picking all pokemon who aren't dead
    team_start_p1 = extract_p1_team_from_game_start(game)
    team_remain_p1 = team_start_p1[~team_start_p1.isin(pkmn_dead_p1)]

    return team_remain_p1


def extract_p1_team_from_game_start_with_stats(game) -> pd.DataFrame:
    """
    Extracts all pokemon teams of p1 with all stats at the starts
    for all games and returns a dataframe of them
    """
    turns = pd.DataFrame([turn["p1_pokemon_state"] for turn in game["battle_timeline"]])
    pkmn_p1_start = turns.drop_duplicates(subset="name", keep="last")

    return pkmn_p1_start


def extract_p1_team_from_game_last_with_stats(game) -> pd.Series:
    """
    Extracts all pokemon teams of p1 with all stats at the end of the 30 turns
    for all games and returns a dataframe of them
    """
    turns = pd.DataFrame([turn["p1_pokemon_state"] for turn in game["battle_timeline"]])
    pkmn_dead_p1 = turns[turns["status"] == "fnt"]["name"].drop_duplicates(keep="last")

    team_start_p1 = extract_p1_team_from_game_start_with_stats(game)
    team_remain_p1 = team_start_p1[~team_start_p1["name"].isin(pkmn_dead_p1)]

    return team_remain_p1


def extract_p2_team_from_game_start(game) -> pd.Series:
    """
    Extracts all pokemon teams of p2 at the start
    for all games and returns a dataframe of them
    """
    turns = pd.DataFrame([turn["p2_pokemon_state"] for turn in game["battle_timeline"]])
    pkmn_p2_start = turns.drop_duplicates(subset="name", keep="last")["name"]

    return pkmn_p2_start


def extract_p2_team_from_game_last(game) -> pd.Series:
    """
    Extracts all pokemon teams of p2 at the end of the 30 turns
    for all games and returns a dataframe of them
    """
    turns = pd.DataFrame([turn["p2_pokemon_state"] for turn in game["battle_timeline"]])
    pkmn_p2_fainted = turns[turns["status"] == "fnt"]["name"].drop_duplicates(
        keep="last"
    )

    pkmn_p2_start = extract_p2_team_from_game_start(game)
    pkmn_p2_last = pkmn_p2_start[~pkmn_p2_start.isin(pkmn_p2_fainted)]

    return pkmn_p2_last


def extract_p2_team_from_game_start_with_stats(game) -> pd.DataFrame:
    """
    Extracts all pokemon teams of p2 with all stats at the start
    for all games and returns a dataframe of them
    """
    turns = pd.DataFrame([turn["p2_pokemon_state"] for turn in game["battle_timeline"]])
    pkmn_p2_start = turns.drop_duplicates(subset="name", keep="last")

    return pkmn_p2_start


def extract_p2_team_from_game_last_with_stats(game) -> pd.Series:
    """
    Extracts all pokemon teams of p2 with all stats at the end of the 30 turns
    for all games and returns a dataframe of them
    """
    turns = pd.DataFrame([turn["p2_pokemon_state"] for turn in game["battle_timeline"]])
    pkmn_dead_p2 = turns[turns["status"] == "fnt"]["name"].drop_duplicates(keep="last")

    team_start_p2 = extract_p2_team_from_game_start_with_stats(game)
    team_remain_p2 = team_start_p2[~team_start_p2["name"].isin(pkmn_dead_p2)]

    return team_remain_p2


def extract_types_from_team_p1(game) -> pd.DataFrame:
    """
    Extracts all types from the teams of p1 at the start
    and returns it along pkmn infos as a dataframe
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    p1_team = extract_p1_team_from_game_start(game).to_frame()
    p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
    p1_team = p1_team[["name", "types"]]

    types = pd.DataFrame(
        [
            type.split(",")
            for pokemon in p1_team["types"]
            for type in [pokemon.strip("[]").replace("'", "").replace(" ", "")]
        ]
    )
    p1_team_types = p1_team.drop("types", axis=1)
    p1_team_types["type_1"] = types[0]
    p1_team_types["type_2"] = types[1]

    return p1_team_types


def extract_types_from_team_p1_last(game) -> pd.DataFrame:
    """
    Extracts all types from the teams of p1 at the end of the 30 turns
    and returns it along pkmn infos as a dataframe
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    p1_team = extract_p1_team_from_game_last(game).to_frame()
    if len(p1_team) != 0:
        p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
        p1_team = p1_team[["name", "types"]]

        types = pd.DataFrame(
            [
                type.split(",")
                for pokemon in p1_team["types"]
                for type in [pokemon.strip("[]").replace("'", "").replace(" ", "")]
            ]
        )
        p1_team_types = p1_team.drop("types", axis=1)
        p1_team_types["type_1"] = types[0]
        p1_team_types["type_2"] = types[1]

        return p1_team_types
    return pd.DataFrame()


def extract_types_from_team_p2(game) -> pd.DataFrame:
    """
    Extracts all types from the teams of p2 at the start
    and returns it along pkmn infos as a dataframe
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    p2_team = extract_p2_team_from_game_start(game).to_frame()
    p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
    p2_team = p2_team[["name", "types"]]

    types = pd.DataFrame(
        [
            type.split(",")
            for pokemon in p2_team["types"]
            for type in [pokemon.strip("[]").replace("'", "").replace(" ", "")]
        ]
    )
    p2_team_types = p2_team.drop("types", axis=1)
    p2_team_types["type_1"] = types[0]
    p2_team_types["type_2"] = types[1]

    return p2_team_types


def extract_types_from_team_p2_last(game) -> pd.DataFrame:
    """
    Extracts all types from the teams of p2 after the 30 turns
    and returns it along pkmn infos as a dataframe
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    p2_team = extract_p2_team_from_game_last(game).to_frame()
    if len(p2_team) != 0:
        p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
        p2_team = p2_team[["name", "types"]]

        types = pd.DataFrame(
            [
                type.split(",")
                for pokemon in p2_team["types"]
                for type in [pokemon.strip("[]").replace("'", "").replace(" ", "")]
            ]
        )
        p2_team_types = p2_team.drop("types", axis=1)
        p2_team_types["type_1"] = types[0]
        p2_team_types["type_2"] = types[1]

        return p2_team_types
    return pd.DataFrame()


def pkmn_database(dataset):
    """
    Creates the database of pokemons from all species seen
    in the train, then it saves it in csv format
    """
    # picking all pokemons seen of p1 in all games
    db_pkmn_p1 = extract_all_pokemon_p1_teams(dataset)

    # picking all pokemon seen of p2 in all games
    db_pkmn_p2 = extract_all_pokemon_p2(dataset)

    # union of dataframes and then dropping duplicates
    db_pkmn = pd.concat([db_pkmn_p1, db_pkmn_p2])
    db_pkmn.drop_duplicates(subset=["name"], inplace=True)

    # saving to csv
    pd.DataFrame.to_csv(db_pkmn, "../data/pkmn_database.csv")


def mean_hp_database(pkmn_database) -> float:
    return np.mean(pkmn_database["base_hp"])


def mean_atk_database(pkmn_database) -> float:
    return np.mean(pkmn_database["base_atk"])


def mean_def_database(pkmn_database) -> float:
    return np.mean(pkmn_database["base_def"])


def mean_spa_database(pkmn_database) -> float:
    return np.mean(pkmn_database["base_spa"])


def mean_spd_database(pkmn_database) -> float:
    return np.mean(pkmn_database["base_spd"])


def mean_spe_database(pkmn_database) -> float:
    return np.mean(pkmn_database["base_spe"])


def mean_total_database(pkmn_database) -> float:
    pkmn_database["total"] = np.sum(
        pkmn_database[
            ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
        ],
        axis=1,
    )
    return np.mean(pkmn_database["total"])


def mean_crit_database(pkmn_database) -> float:
    return np.mean(pkmn_database["base_spe"] / 512)


def all_pokemon_round(player: int, json):
    if player == 1:
        return set(
            [elem["p1_pokemon_state"]["name"] for elem in json["battle_timeline"]]
        )
    elif player == 2:
        return set(
            [elem["p2_pokemon_state"]["name"] for elem in json["battle_timeline"]]
        )
