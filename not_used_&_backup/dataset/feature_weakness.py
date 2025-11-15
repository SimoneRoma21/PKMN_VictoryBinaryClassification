import pandas as pd
import numpy as np

from ...py.dataset import csv_utilities as csv_u
from ...py.dataset import extract_utilities as ext_u


# to remove, not used, not good
def weakness_teams_not_opt(dataset) -> pd.DataFrame:  # ---> DONT USE <---
    weak_games_p1, weak_games_p2 = [], []
    for game in dataset:
        # if game['battle_id']==0:
        weakness_p1 = []
        p1_team_types = ext_u.extract_types_from_team_p1(game)
        for index, row in p1_team_types.iterrows():
            weaknesses = ext_u.calc_weakness(row.iloc[1], row.iloc[2])
            weakness_p1.append(weaknesses)
            # print(weaknesses,"\n")
        weakness_p1 = (
            pd.concat(weakness_p1)
            .reset_index()
            .rename(columns={"index": "type"})
            .drop_duplicates(subset="type")
            .reset_index(drop=True)
        )
        weakness_p1 = weakness_p1["type"]

        weakness_p2 = []
        p2_team_types = ext_u.extract_types_from_team_p2(game)
        for index, row in p2_team_types.iterrows():
            weaknesses = ext_u.calc_weakness(row.iloc[1], row.iloc[2])
            weakness_p2.append(weaknesses)
            # print(weaknesses,"\n")
        weakness_p2 = (
            pd.concat(weakness_p2)
            .reset_index()
            .rename(columns={"index": "type"})
            .drop_duplicates(subset="type")
            .reset_index(drop=True)
        )
        weakness_p2 = weakness_p2["type"]

        weak_games_p1.append(weakness_p1.count())
        weak_games_p2.append(weakness_p2.count())

    return pd.DataFrame(
        {"weakness_start_p1": weak_games_p1, "weakness_start_p2": weak_games_p2}
    )


def weakness_teams(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the weakness of the teams of p1 and p2 at the start per each game.
    Uses the matrix of types of pokemons. Returns results in a dataframe
    """

    # getting pokemons info and their weakness from databases
    pkmn_db_weak = csv_u.open_pkmn_database_weak_csv()
    pkmn_db_weak = pd.DataFrame(pkmn_db_weak[["name", "weaknesses"]])
    pkmn_db_weak["weaknesses"] = pkmn_db_weak["weaknesses"].apply(
        lambda x: x.strip("[] ").replace("'", "").replace(" ", "").split(",")
    )

    weak_games_p1, weak_games_p2 = [], []

    for game in dataset:
        # extract pokemon types from p1 team and merging with the weakness in pkmn_db_weak
        p1_team_types = ext_u.extract_types_from_team_p1(game)
        p1_team_types = p1_team_types.merge(pkmn_db_weak, how="inner", on="name")
        sw_1 = set(sum(p1_team_types["weaknesses"], []))
        weak_games_p1.append(
            len(sw_1)
        )  # append the len of the set (no duplicate weaknesses)

        # extract pokemon types from p2 team and merging with the weakness in pkmn_db_weak
        p2_team_types = ext_u.extract_types_from_team_p2(game)
        p2_team_types = p2_team_types.merge(pkmn_db_weak, how="inner", on="name")
        sw_2 = set(sum(p2_team_types["weaknesses"], []))
        weak_games_p2.append(
            len(sw_2)
        )  # append the len of the set (no duplicate weaknesses)

    weakness_teams = pd.DataFrame(
        {"weakness_start_p1": weak_games_p1, "weakness_start_p2": weak_games_p2}
    )
    weakness_teams["weakness_start_difference"] = np.subtract.reduce(
        weakness_teams[["weakness_start_p1", "weakness_start_p2"]], axis=1
    )
    return weakness_teams


def p1_weakness_start(dataset) -> pd.DataFrame:  # Feature
    """
    Returns the weakness of the team of p1. Results in a Dataframe.
    """
    return weakness_teams(dataset)["weakness_start_p1"]


def p2_weakness_start(dataset) -> pd.DataFrame:  # Feature
    """
    Returns the weakness of the team of p2. Results in a Dataframe.
    """
    return weakness_teams(dataset)["weakness_start_p2"]


def weakness_start_difference(dataset) -> pd.DataFrame:  # feature
    """
    Returns the difference between the number of weakness
    for the teams of p1 and p2. Results in a Dataframe.
    """
    return weakness_teams(dataset)["weakness_start_difference"]


def weakness_teams_last(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the weakness of the teams of p1 and p2 for the remaining pokemons
    after the 30 turns per each game. Uses the matrix of types of pokemons.
    Returns results in a dataframe.
    """

    # getting pokemons info and their weakness from databases
    pkmn_db_weak = csv_u.open_pkmn_database_weak_csv()
    pkmn_db_weak = pd.DataFrame(pkmn_db_weak[["name", "weaknesses"]])
    pkmn_db_weak["weaknesses"] = pkmn_db_weak["weaknesses"].apply(
        lambda x: x.strip("[] ").replace("'", "").replace(" ", "").split(",")
    )

    weak_games_p1, weak_games_p2 = [], []

    for game in dataset:
        # extract pokemon types from p1 team and merging with the weakness in pkmn_db_weak
        p1_team_types = ext_u.extract_types_from_team_p1_last(game)
        if len(p1_team_types) != 0:  # checks if p1 has pokemons
            p1_team_types = p1_team_types.merge(pkmn_db_weak, how="inner", on="name")
            sw_1 = set(sum(p1_team_types["weaknesses"], []))
            weak_games_p1.append(
                len(sw_1)
            )  # append the len of the set (no duplicate weaknesses)
        else:
            weak_games_p1.append(0)  # if no pokemon weaknesses of the team are 0

        # extract pokemon types from p2 team and merging with the weakness in pkmn_db_weak
        p2_team_types = ext_u.extract_types_from_team_p2_last(game)
        if len(p2_team_types) != 0:  # checks if p2 has pokemons
            p2_team_types = p2_team_types.merge(pkmn_db_weak, how="inner", on="name")
            sw_2 = set(sum(p2_team_types["weaknesses"], []))
            weak_games_p2.append(
                len(sw_2)
            )  # append the len of the set (no duplicate weaknesses)
        else:
            weak_games_p2.append(0)  # if no pokemon weaknesses of the team are 0

    weakness_teams = pd.DataFrame(
        {"weakness_last_p1": weak_games_p1, "weakness_last_p2": weak_games_p2}
    )
    weakness_teams["weakness_last_difference"] = np.subtract.reduce(
        weakness_teams[["weakness_last_p1", "weakness_last_p2"]], axis=1
    )
    return weakness_teams


def p1_weakness_last(dataset) -> pd.DataFrame:  # Feature
    """
    Returns the weakness of the team of p1 for pokemon who survived the 30 turns.
    Results in a Dataframe.
    """
    return weakness_teams_last(dataset)["weakness_last_p1"]


def p2_weakness_last(dataset) -> pd.DataFrame:  # Feature
    """
    Returns the weakness of the team of p2 for pokemon who survived the 30 turns.
    Results in a Dataframe.
    """
    return weakness_teams_last(dataset)["weakness_last_p2"]


def weakness_last_difference(dataset) -> pd.DataFrame:  # feature
    """
    Returns the difference between the number of weakness
    for the teams of p1 and p2. The pokemon types considered are the ones
    of alive pokemons after the 30 turns. Results in a Dataframe.
    """
    return weakness_teams_last(dataset)["weakness_last_difference"]


def advantage_weak_start(dataset) -> pd.DataFrame:  # feature
    """
    Calculates the advantage of types between the teams of p1 and p2 at the start doing an
    intersection of the types that a team has and the weakness of the other team. So
    the advantage of p1 on p2 will be the intersection between type coverage of p1 on types of
    team p2. Results will be returned in a Dataframe.
    """

    # getting pokemons info and their weakness from databases
    pkmn_db_weak = csv_u.open_pkmn_database_weak_csv()
    pkmn_db_weak = pd.DataFrame(pkmn_db_weak[["name", "weaknesses"]])
    pkmn_db_weak["weaknesses"] = pkmn_db_weak["weaknesses"].apply(
        lambda x: x.strip("[] ").replace("'", "").replace(" ", "").split(",")
    )

    adv_games_p1, adv_games_p2 = [], []

    for game in dataset:
        # extract pokemon types from p1 and p2 teams
        p1_team_types = ext_u.extract_types_from_team_p1(game)
        p2_team_types = ext_u.extract_types_from_team_p2(game)

        # merge the teams with their weakness
        p1_team_weakness = p1_team_types.merge(pkmn_db_weak, how="inner", on="name")
        p2_team_weakness = p2_team_types.merge(pkmn_db_weak, how="inner", on="name")

        # sum the weakness of all pokemons in a team within a list
        sw_1 = set(sum(p1_team_weakness["weaknesses"], []))
        sw_2 = set(sum(p2_team_weakness["weaknesses"], []))

        # calculate the type coverage of the team by getting the types of all pokemons in a team
        all_type_s1 = set(
            (p1_team_types["type_1"].to_list() + p1_team_types["type_2"].to_list())
        )
        all_type_s2 = set(
            (p2_team_types["type_1"].to_list() + p2_team_types["type_2"].to_list())
        )

        # if p1 has pokemon do intersection between types covered and weakness of p2
        if len(all_type_s1) != 0:
            all_type_s1.discard("notype")
            advantages_p1 = all_type_s1.intersection(sw_2)
            adv_games_p1.append(len(advantages_p1))  # append number of advantages
        else:
            adv_games_p1.append(0)  # 0 advantages if team is empty due to dead pokemons

        # if p2 has pokemon do intersection between types covered and weakness of p1
        if len(all_type_s2) != 0:
            all_type_s2.discard("notype")
            advantages_p2 = all_type_s2.intersection(sw_1)
            adv_games_p2.append(len(advantages_p2))  # append number of advantages
        else:
            adv_games_p2.append(0)  # 0 advantages if team is empty due to dead pokemons

    advantage_weak_teams = pd.DataFrame(
        {
            "advantage_weak_start_p1": adv_games_p1,
            "advantage_weak_start_p2": adv_games_p2,
        }
    )
    advantage_weak_teams["advantage_weak_difference"] = np.subtract.reduce(
        advantage_weak_teams[["advantage_weak_start_p1", "advantage_weak_start_p2"]],
        axis=1,
    )

    return advantage_weak_teams


def p1_advantage_weak_start(dataset) -> pd.DataFrame:  # Feature
    """
    Returns the advantages of p1 team respect to p2 team at the start of the battle.
    Return results in a dataframe.
    """
    return advantage_weak_last(dataset)["advantage_weak_start_p1"]


def p2_advantage_weak_start(dataset) -> pd.DataFrame:  # feature
    """
    Returns the advantages of p2 team respect to p1 team at the start of the battle.
    Return results in a dataframe.
    """
    return advantage_weak_last(dataset)["advantage_weak_start_p2"]


def advantage_weak_start_difference(dataset) -> pd.DataFrame:  # feature
    """
    Returns the difference between the number advantages of p1->p2 and p2->p1 at the start of the battle.
    Return results in a dataframe.
    """
    return advantage_weak_last(dataset)["advantage_weak_difference"]


def advantage_weak_last(dataset) -> pd.DataFrame:  # feature
    """
    Calculates the advantage of types between the teams of p1 and p2 at the end of the 30 turns
    doing an intersection of the types that a team has and the weakness of the other team. So
    the advantage of p1 on p2 will be the intersection between type coverage of p1 on types of
    team p2. Results will be returned in a Dataframe.
    """

    # getting pokemons info and their weakness from databases
    pkmn_db_weak = csv_u.open_pkmn_database_weak_csv()
    pkmn_db_weak = pd.DataFrame(pkmn_db_weak[["name", "weaknesses"]])
    pkmn_db_weak["weaknesses"] = pkmn_db_weak["weaknesses"].apply(
        lambda x: x.strip("[] ").replace("'", "").replace(" ", "").split(",")
    )

    adv_games_p1, adv_games_p2 = [], []

    for game in dataset:
        # extract pokemon types from p1 and p2 teams
        p1_team_types = ext_u.extract_types_from_team_p1_last(game)
        p2_team_types = ext_u.extract_types_from_team_p2_last(game)
        all_type_s1, all_type_s2 = set(), set()

        # if both teams aren't empty
        if len(p1_team_types) != 0 and len(p2_team_types) != 0:
            # merge the p1 team with its weakness, compute the total weakness of the team
            # and then the type coverage of the team
            p1_team_weakness = p1_team_types.merge(pkmn_db_weak, how="inner", on="name")
            sw_1 = set(sum(p1_team_weakness["weaknesses"], []))
            all_type_s1 = set(
                (p1_team_types["type_1"].to_list() + p1_team_types["type_2"].to_list())
            )

            # merge the p1 team with its weakness, compute the total weakness of the team
            # and then the type coverage of the team
            p2_team_weakness = p2_team_types.merge(pkmn_db_weak, how="inner", on="name")
            sw_2 = set(sum(p2_team_weakness["weaknesses"], []))
            all_type_s2 = set(
                (p2_team_types["type_1"].to_list() + p2_team_types["type_2"].to_list())
            )

        # if p1 has pokemon do intersection between types covered and weakness of p2
        if len(all_type_s1) != 0:
            all_type_s1.discard("notype")
            advantages_p1 = all_type_s1.intersection(sw_2)
            adv_games_p1.append(len(advantages_p1))  # append number of advantages
        else:
            adv_games_p1.append(0)  # 0 advantages if team is empty due to dead pokemons

        # if p2 has pokemon do intersection between types covered and weakness of p1
        if len(all_type_s2) != 0:
            all_type_s2.discard("notype")
            advantages_p2 = all_type_s2.intersection(sw_1)
            adv_games_p2.append(len(advantages_p2))  # append number of advantages
        else:
            adv_games_p2.append(0)  # 0 advantages if team is empty due to dead pokemons

    advantage_weak_teams = pd.DataFrame(
        {
            "advantage_weak_last_p1": adv_games_p1,
            "advantage_last_start_p2": adv_games_p2,
        }
    )
    advantage_weak_teams["advantage_weak_last_difference"] = np.subtract.reduce(
        advantage_weak_teams[["advantage_weak_last_p1", "advantage_last_start_p2"]],
        axis=1,
    )
    return advantage_weak_teams


def p1_advantage_weak_last(dataset) -> pd.DataFrame:  # Feature
    """
    Returns the advantages of p1 team respect to p2 team after the 30 turns.
    Return results in a dataframe.
    """
    return advantage_weak_last(dataset)["advantage_weak_last_p1"]


def p2_advantage_weak_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the advantages of p2 team respect to p1 team after the 30 turns.
    Return results in a dataframe.
    """
    return advantage_weak_last(dataset)["advantage_weak_last_p2"]


def advantage_weak_last_difference(dataset) -> pd.DataFrame:  # feature
    """
    Returns the difference between the number advantages of p1->p2 and p2->p1 at the
    end of the 30 turns of the battle. Return results in a dataframe.
    """
    return advantage_weak_last(dataset)["advantage_weak_last_difference"]


def p1_psy_pkmn(dataset) -> pd.DataFrame:  # feature
    """
    Counts the number of psychic pokemon in the team of p1 per each game. Psy pokemons doesn't have
    weaknesses in gen1 OU, so its useful to know, they're more difficult to bring exaust.
    Return results in a dataframe.
    """
    p1_count = []
    for game in dataset:
        # taking p1 team
        p1_team = ext_u.extract_types_from_team_p1_last(game)
        if len(p1_team) != 0:
            # getting all psychic pkmn and appending the count
            p1_team = p1_team.query("type_1=='psychic' or type_2=='psychic'")
            p1_count.append(len(p1_team))
        else:
            p1_count.append(0)  # 0 if no pokemon in team
    return pd.DataFrame({"p1_psychic_pkmn_last": p1_count})


def p2_psy_pkmn(dataset) -> pd.DataFrame:  # feature
    """
    Counts the number of psychic pokemon in the team of p2 per each game. Psy pokemons doesn't have
    weaknesses in gen1 OU, so its useful to know, they're more difficult to bring exaust.
    Return results in a dataframe.
    """
    p2_count = []
    for game in dataset:
        # taking p2 team
        p2_team = ext_u.extract_types_from_team_p2_last(game)
        if len(p2_team) != 0:
            # getting all psychic pkmn and appending the count
            p2_team = p2_team.query("type_1=='psychic' or type_2=='psychic'")
            p2_count.append(len(p2_team))
        else:
            p2_count.append(0)  # 0 if no pokemon in team
    return pd.DataFrame({"p2_psychic_pkmn_last": p2_count})
