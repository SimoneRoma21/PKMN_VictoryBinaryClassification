import numpy as np
import pandas as pd

from . import extract_utilities as ext_u
from . import csv_utilities as csv_u
from . import feature_during_battle as fdb


def off_def_ratio(dataset) -> pd.DataFrame:
    """
    Calculate the offense-to-defense ratio for both players.
    This metric helps identify aggressive vs defensive team compositions.
    A higher ratio indicates a more offensive-oriented team.

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with columns:
            - p1_off_def_ratio: Player 1's attack/defense ratio
            - p2_off_def_ratio: Player 2's attack/defense ratio
    """
    atks = mean_atk_last_2(dataset)
    defs = mean_def_last_2(dataset)
    return pd.DataFrame(
        {
            "p1_off_def_ratio": atks["p1_mean_atk_last"]
            / (defs["p1_mean_def_last"] + 1),
            "p2_off_def_ratio": atks["p2_mean_atk_last"]
            / (defs["p2_mean_def_last"] + 1),
        }
    )


def off_spad_ratio(dataset) -> pd.DataFrame:
    """
    Calculate the special attack-to-special defense ratio for both players.
    This metric helps identify special offensive vs special defensive team compositions.
    A higher ratio indicates a more special-attack-oriented team.

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with columns:
            - p1_off_spad_ratio: Player 1's special attack/special defense ratio
            - p2_off_spad_ratio: Player 2's special attack/special defense ratio
    """
    atks = mean_spa_last_2(dataset)
    defs = mean_spd_last_2(dataset)
    return pd.DataFrame(
        {
            "p1_off_spad_ratio": atks["p1_mean_spa_last"]
            / (defs["p1_mean_spd_last"] + 1),
            "p2_off_spad_ratio": atks["p2_mean_spa_last"]
            / (defs["p2_mean_spd_last"] + 1),
        }
    )


def spe_atk_ratio(dataset) -> pd.DataFrame:
    """
    Calculate the speed-to-attack ratio for both players.
    This metric helps identify fast vs powerful team compositions.
    A higher ratio indicates a more speed-focused team relative to attack power.

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with columns:
            - p1_spe_atk_ratio: Player 1's speed/attack ratio
            - p2_spe_atk_ratio: Player 2's speed/attack ratio
    """
    spes = mean_spe_last_2(dataset)
    atks = mean_atk_last_2(dataset)

    return pd.DataFrame(
        {
            "p1_spe_atk_ratio": spes["p1_mean_spe_last"]
            / (atks["p1_mean_atk_last"] + 1),
            "p2_spe_atk_ratio": spes["p2_mean_spe_last"]
            / (atks["p2_mean_atk_last"] + 1),
        }
    )


def hp_bulk_ratio(dataset) -> pd.DataFrame:
    """
    Calculate the HP-to-bulk ratio for both players.
    Bulk is defined as the average of defense and special defense.
    This metric measures overall defensive capability combining HP with defensive stats.
    A higher ratio indicates better defensive survivability.

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with columns:
            - p1_hp_bulk_ratio: Player 1's HP/average defense ratio
            - p2_hp_bulk_ratio: Player 2's HP/average defense ratio
    """
    hps = mean_hp_last(dataset)
    defs = mean_def_last_2(dataset)
    defd = mean_spd_last_2(dataset)

    return pd.DataFrame(
        {
            "p1_hp_bulk_ratio": hps["p1_mean_hp_last"]
            / ((defs["p1_mean_def_last"] + defd["p1_mean_spd_last"]) / 2 + 1),
            "p2_hp_bulk_ratio": hps["p2_mean_hp_last"]
            / ((defs["p2_mean_def_last"] + defd["p2_mean_spd_last"]) / 2 + 1),
        }
    )


# Offense-speed product
def atk_spe_prod(dataset) -> pd.DataFrame:
    """
    Calculate the offense-speed product for both players.
    Combines offensive power (ATK) with speed (SPE).
    A high value indicates a more aggressive and fast team composition.

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with columns:
            - p1_offense_speed_product: Player 1's attack × speed
            - p2_offense_speed_product: Player 2's attack × speed
            - offense_speed_product_diff: Difference between players
    """
    # Calcola le feature di base
    mean_atk_df = mean_atk_last(dataset)
    mean_spe_df = mean_spe_last(dataset)

    # Calcolo del prodotto per p1 e p2
    p1_offense_speed = mean_atk_df["p1_mean_atk_last"] * mean_spe_df["p1_mean_spe_last"]
    p2_offense_speed = mean_atk_df["p2_mean_atk_last"] * mean_spe_df["p2_mean_spe_last"]

    # Differenza
    offense_speed_diff = p1_offense_speed - p2_offense_speed

    # Costruzione DataFrame finale
    df = pd.DataFrame(
        {
            "p1_offense_speed_product": p1_offense_speed,
            "p2_offense_speed_product": p2_offense_speed,
            "offense_speed_product_diff": offense_speed_diff,
        }
    ).fillna(0)

    return df


# Feature Trend
def hp_trend(dataset) -> pd.DataFrame:
    """
    Calculate the HP trend for both players between the start and the last state of the game.
    P1_HP_TREND = mean_hp_last - mean_hp_start
    P2_HP_TREND = mean_hp_last - mean_hp_start
    HP_TREND_DIFF = P1_HP_TREND - P2_HP_TREND
    """
    p1_start_df = p1_mean_hp_start(dataset)
    p2_start_df = p2_mean_hp_start(dataset)
    mean_last_df = mean_hp_last(dataset)

    p1_hp_trend = mean_last_df["p1_mean_hp_last"] - p1_start_df["p1_mean_hp_start"]
    p2_hp_trend = mean_last_df["p2_mean_hp_last"] - p2_start_df["p2_mean_hp_start"]
    hp_trend_diff = p1_hp_trend - p2_hp_trend

    df = pd.DataFrame(
        {
            "p1_hp_trend": p1_hp_trend,
            "p2_hp_trend": p2_hp_trend,
            "hp_trend_diff": hp_trend_diff,
        }
    ).fillna(0)
    return df


def atk_trend(dataset) -> pd.DataFrame:
    """
    Calculate the attack trend for both players between the start and the last state of the game.
    P1_ATK_TREND = mean_atk_last - mean_atk_start
    P2_ATK_TREND = mean_atk_last - mean_atk_start
    ATK_TREND_DIFF = P1_ATK_TREND - P2_ATK_TREND

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with columns:
            - p1_atk_trend: Player 1's attack change over the battle
            - p2_atk_trend: Player 2's attack change over the battle
            - atk_trend_diff: Difference between players' attack trends
    """
    p1_start_df = p1_mean_atk_start(dataset)
    p2_start_df = p2_mean_atk_start(dataset)
    mean_last_df = mean_atk_last(dataset)

    p1_atk_trend = mean_last_df["p1_mean_atk_last"] - p1_start_df["p1_mean_atk_start"]
    p2_atk_trend = mean_last_df["p2_mean_atk_last"] - p2_start_df["p2_mean_atk_start"]
    atk_trend_diff = p1_atk_trend - p2_atk_trend

    df = pd.DataFrame(
        {
            "p1_atk_trend": p1_atk_trend,
            "p2_atk_trend": p2_atk_trend,
            "atk_trend_diff": atk_trend_diff,
        }
    ).fillna(0)
    return df


def def_trend(dataset) -> pd.DataFrame:
    """
    Calculate the defense trend for both players between the start and the last state of the game.
    P1_DEF_TREND = mean_def_last - mean_def_start
    P2_DEF_TREND = mean_def_last - mean_def_start
    DEF_TREND_DIFF = P1_DEF_TREND - P2_DEF_TREND

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with columns:
            - p1_def_trend: Player 1's defense change over the battle
            - p2_def_trend: Player 2's defense change over the battle
            - def_trend_diff: Difference between players' defense trends
    """
    p1_start_df = p1_mean_def_start(dataset)
    p2_start_df = p2_mean_def_start(dataset)
    mean_last_df = mean_def_last(dataset)

    p1_def_trend = mean_last_df["p1_mean_def_last"] - p1_start_df["p1_mean_def_start"]
    p2_def_trend = mean_last_df["p2_mean_def_last"] - p2_start_df["p2_mean_def_start"]
    def_trend_diff = p1_def_trend - p2_def_trend

    df = pd.DataFrame(
        {
            "p1_def_trend": p1_def_trend,
            "p2_def_trend": p2_def_trend,
            "def_trend_diff": def_trend_diff,
        }
    ).fillna(0)
    return df


def spa_trend(dataset) -> pd.DataFrame:
    """
    Calculate the special attack trend for both players between the start and the last state of the game.
    P1_SPA_TREND = mean_spa_last - mean_spa_start
    P2_SPA_TREND = mean_spa_last - mean_spa_start
    SPA_TREND_DIFF = P1_SPA_TREND - P2_SPA_TREND

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with columns:
            - p1_spa_trend: Player 1's special attack change over the battle
            - p2_spa_trend: Player 2's special attack change over the battle
            - spa_trend_diff: Difference between players' special attack trends
    """
    p1_start_df = p1_mean_spa_start(dataset)
    p2_start_df = p2_mean_spa_start(dataset)
    mean_last_df = mean_spa_last(dataset)

    p1_spa_trend = mean_last_df["p1_mean_spa_last"] - p1_start_df["p1_mean_spa_start"]
    p2_spa_trend = mean_last_df["p2_mean_spa_last"] - p2_start_df["p2_mean_spa_start"]
    spa_trend_diff = p1_spa_trend - p2_spa_trend

    df = pd.DataFrame(
        {
            "p1_spa_trend": p1_spa_trend,
            "p2_spa_trend": p2_spa_trend,
            "spa_trend_diff": spa_trend_diff,
        }
    ).fillna(0)
    return df


def spd_trend(dataset) -> pd.DataFrame:
    """
    Calculate the special defense trend for both players between the start and the last state of the game.
    P1_SPD_TREND = mean_spd_last - mean_spd_start
    P2_SPD_TREND = mean_spd_last - mean_spd_start
    SPD_TREND_DIFF = P1_SPD_TREND - P2_SPD_TREND

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with columns:
            - p1_spd_trend: Player 1's special defense change over the battle
            - p2_spd_trend: Player 2's special defense change over the battle
            - spd_trend_diff: Difference between players' special defense trends
    """
    p1_start_df = p1_mean_spd_start(dataset)
    p2_start_df = p2_mean_spd_start(dataset)
    mean_last_df = mean_spd_last(dataset)

    p1_spd_trend = mean_last_df["p1_mean_spd_last"] - p1_start_df["p1_mean_spd_start"]
    p2_spd_trend = mean_last_df["p2_mean_spd_last"] - p2_start_df["p2_mean_spd_start"]
    spd_trend_diff = p1_spd_trend - p2_spd_trend

    df = pd.DataFrame(
        {
            "p1_spd_trend": p1_spd_trend,
            "p2_spd_trend": p2_spd_trend,
            "spd_trend_diff": spd_trend_diff,
        }
    ).fillna(0)
    return df


def spe_trend(dataset) -> pd.DataFrame:
    """
    Calculate the speed trend for both players between the start and the last state of the game.
    P1_SPE_TREND = mean_spe_last - mean_spe_start
    P2_SPE_TREND = mean_spe_last - mean_spe_start
    SPE_TREND_DIFF = P1_SPE_TREND - P2_SPE_TREND

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with columns:
            - p1_spe_trend: Player 1's speed change over the battle
            - p2_spe_trend: Player 2's speed change over the battle
            - spe_trend_diff: Difference between players' speed trends
    """
    p1_start_df = p1_mean_spe_start(dataset)
    p2_start_df = p2_mean_spe_start(dataset)
    mean_last_df = mean_spe_last(dataset)

    p1_spe_trend = mean_last_df["p1_mean_spe_last"] - p1_start_df["p1_mean_spe_start"]
    p2_spe_trend = mean_last_df["p2_mean_spe_last"] - p2_start_df["p2_mean_spe_start"]
    spe_trend_diff = p1_spe_trend - p2_spe_trend

    df = pd.DataFrame(
        {
            "p1_spe_trend": p1_spe_trend,
            "p2_spe_trend": p2_spe_trend,
            "spe_trend_diff": spe_trend_diff,
        }
    ).fillna(0)
    return df


def p1_mean_atk_start(dataset) -> pd.DataFrame:
    """
    Calculate the mean base attack for the team of p1 at the start of the game for all games.
    Unknown pokemons are replaced by the global mean base attack.
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_atk = ext_u.mean_atk_database(pkmn_database)
    p1_mean_atk = []

    for game in dataset:
        p1_team = ext_u.extract_p1_team_from_game_start(game)
        p1_known = len(p1_team)
        p1_team = pkmn_database[pkmn_database["name"].isin(p1_team)][
            ["name", "base_atk"]
        ]
        p1_mean_atk.append(
            (np.sum(p1_team["base_atk"]) + mean_atk * (6 - p1_known)) / 6
        )

    return pd.DataFrame({"p1_mean_atk_start": p1_mean_atk})


def p2_mean_atk_start(dataset) -> pd.DataFrame:
    """
    Calculate the mean base attack for the team of p2 at the start of the game for all games.
    Unknown pokemons are replaced by the global mean base attack.

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with column:
            - p2_mean_atk_start: Player 2's mean attack at battle start
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_atk = ext_u.mean_atk_database(pkmn_database)
    p2_mean_atk = []

    for game in dataset:
        p2_team = ext_u.extract_p2_team_from_game_start(game)
        p2_team = pkmn_database[pkmn_database["name"].isin(p2_team)][
            ["name", "base_atk"]
        ]
        p2_mean_atk.append(
            (np.sum(p2_team["base_atk"]) + mean_atk * (6 - len(p2_team))) / 6
        )

    return pd.DataFrame({"p2_mean_atk_start": p2_mean_atk})


def p1_mean_def_start(dataset) -> pd.DataFrame:
    """
    Calculate the mean base defense for the team of p1 at the start of the game for all games.
    Unknown pokemons are replaced by the global mean base defense.

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with column:
            - p1_mean_def_start: Player 1's mean defense at battle start
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_def = ext_u.mean_def_database(pkmn_database)
    p1_mean_def = []

    for game in dataset:
        p1_team = ext_u.extract_p1_team_from_game_start(game)
        p1_known = len(p1_team)
        p1_team = pkmn_database[pkmn_database["name"].isin(p1_team)][
            ["name", "base_def"]
        ]
        p1_mean_def.append(
            (np.sum(p1_team["base_def"]) + mean_def * (6 - p1_known)) / 6
        )

    return pd.DataFrame({"p1_mean_def_start": p1_mean_def})


def p2_mean_def_start(dataset) -> pd.DataFrame:
    """
    Calculate the mean base defense for the team of p2 at the start of the game for all games.
    Unknown pokemons are replaced by the global mean base defense.

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with column:
            - p2_mean_def_start: Player 2's mean defense at battle start
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_def = ext_u.mean_def_database(pkmn_database)
    p2_mean_def = []

    for game in dataset:
        p2_team = ext_u.extract_p2_team_from_game_start(game)
        p2_team = pkmn_database[pkmn_database["name"].isin(p2_team)][
            ["name", "base_def"]
        ]
        p2_mean_def.append(
            (np.sum(p2_team["base_def"]) + mean_def * (6 - len(p2_team))) / 6
        )

    return pd.DataFrame({"p2_mean_def_start": p2_mean_def})


def p1_mean_spa_start(dataset) -> pd.DataFrame:
    """
    Calculate the mean base special attack for the team of p1 at the start of the game for all games.
    Unknown pokemons are replaced by the global mean base special attack.

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with column:
            - p1_mean_spa_start: Player 1's mean special attack at battle start
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spa = ext_u.mean_spa_database(pkmn_database)
    p1_mean_spa = []

    for game in dataset:
        p1_team = ext_u.extract_p1_team_from_game_start(game)
        p1_known = len(p1_team)
        p1_team = pkmn_database[pkmn_database["name"].isin(p1_team)][
            ["name", "base_spa"]
        ]
        p1_mean_spa.append(
            (np.sum(p1_team["base_spa"]) + mean_spa * (6 - p1_known)) / 6
        )

    return pd.DataFrame({"p1_mean_spa_start": p1_mean_spa})


def p2_mean_spa_start(dataset) -> pd.DataFrame:
    """
    Calculate the mean base special attack for the team of p2 at the start of the game for all games.
    Unknown pokemons are replaced by the global mean base special attack.

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with column:
            - p2_mean_spa_start: Player 2's mean special attack at battle start
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spa = ext_u.mean_spa_database(pkmn_database)
    p2_mean_spa = []

    for game in dataset:
        p2_team = ext_u.extract_p2_team_from_game_start(game)
        p2_team = pkmn_database[pkmn_database["name"].isin(p2_team)][
            ["name", "base_spa"]
        ]
        p2_mean_spa.append(
            (np.sum(p2_team["base_spa"]) + mean_spa * (6 - len(p2_team))) / 6
        )

    return pd.DataFrame({"p2_mean_spa_start": p2_mean_spa})


def p1_mean_spd_start(dataset) -> pd.DataFrame:
    """
    Calculate the mean base special defense for the team of p1 at the start of the game for all games.
    Unknown pokemons are replaced by the global mean base special defense.

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with column:
            - p1_mean_spd_start: Player 1's mean special defense at battle start
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spd = ext_u.mean_spd_database(pkmn_database)
    p1_mean_spd = []

    for game in dataset:
        p1_team = ext_u.extract_p1_team_from_game_start(game)
        p1_known = len(p1_team)
        p1_team = pkmn_database[pkmn_database["name"].isin(p1_team)][
            ["name", "base_spd"]
        ]
        p1_mean_spd.append(
            (np.sum(p1_team["base_spd"]) + mean_spd * (6 - p1_known)) / 6
        )

    return pd.DataFrame({"p1_mean_spd_start": p1_mean_spd})


def p2_mean_spd_start(dataset) -> pd.DataFrame:
    """
    Calculate the mean base special defense for the team of p2 at the start of the game for all games.
    Unknown pokemons are replaced by the global mean base special defense.

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with column:
            - p2_mean_spd_start: Player 2's mean special defense at battle start
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spd = ext_u.mean_spd_database(pkmn_database)
    p2_mean_spd = []

    for game in dataset:
        p2_team = ext_u.extract_p2_team_from_game_start(game)
        p2_team = pkmn_database[pkmn_database["name"].isin(p2_team)][
            ["name", "base_spd"]
        ]
        p2_mean_spd.append(
            (np.sum(p2_team["base_spd"]) + mean_spd * (6 - len(p2_team))) / 6
        )

    return pd.DataFrame({"p2_mean_spd_start": p2_mean_spd})


def p1_mean_spe_start(dataset) -> pd.DataFrame:
    """
    Calculate the mean base speed for the team of p1 at the start of the game for all games.
    Unknown pokemons are replaced by the global mean base speed.

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with column:
            - p1_mean_spe_start: Player 1's mean speed at battle start
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spe = ext_u.mean_spe_database(pkmn_database)
    p1_mean_spe = []

    for game in dataset:
        p1_team = ext_u.extract_p1_team_from_game_start(game)
        p1_known = len(p1_team)
        p1_team = pkmn_database[pkmn_database["name"].isin(p1_team)][
            ["name", "base_spe"]
        ]
        p1_mean_spe.append(
            (np.sum(p1_team["base_spe"]) + mean_spe * (6 - p1_known)) / 6
        )

    return pd.DataFrame({"p1_mean_spe_start": p1_mean_spe})


def p2_mean_spe_start(dataset) -> pd.DataFrame:
    """
    Calculate the mean base speed for the team of p2 at the start of the game for all games.
    Unknown pokemons are replaced by the global mean base speed.

    Args:
        dataset: List of battle dictionaries

    Returns:
        DataFrame with column:
            - p2_mean_spe_start: Player 2's mean speed at battle start
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spe = ext_u.mean_spe_database(pkmn_database)
    p2_mean_spe = []

    for game in dataset:
        p2_team = ext_u.extract_p2_team_from_game_start(game)
        p2_team = pkmn_database[pkmn_database["name"].isin(p2_team)][
            ["name", "base_spe"]
        ]
        p2_mean_spe.append(
            (np.sum(p2_team["base_spe"]) + mean_spe * (6 - len(p2_team))) / 6
        )

    return pd.DataFrame({"p2_mean_spe_start": p2_mean_spe})


# ----Feature Base Stats HP----#
def p1_mean_hp_start(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the mean base hp for the team of p1 at the start of the game for all games.
    Since we might not know all pokemons in a team, we used a standardized global mean.
    So for the pokemons we know, we'll use the sum of its base hp, for the ones we
    don't know, we add a global mean hp calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean hp. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """
    # opening databases
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_hp = ext_u.mean_hp_database(pkmn_database)
    p1_mean_hp = []

    for game in dataset:
        p1_team = ext_u.extract_p1_team_from_game_start(game)  # taking p1 team
        p1_known = len(p1_team)
        p1_team = pkmn_database[pkmn_database["name"].isin(p1_team)]
        p1_team = p1_team[["name", "base_hp"]]  # getting stats
        # calculating and appending mean hp
        p1_mean_hp.append((np.sum(p1_team["base_hp"]) + mean_hp * (6 - p1_known)) / 6)

    mean_hp_start = pd.DataFrame({"p1_mean_hp_start": p1_mean_hp})
    return mean_hp_start


def p2_mean_hp_start(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the mean base hp for the team of p2 at the start of the game for all games.
    Since we might not know all pokemons in a team, we used a standardized global mean.
    So for the pokemons we know, we'll use the sum of its base hp, for the ones we
    don't know, we add a global mean hp calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean hp. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """
    # opening databases
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_hp = ext_u.mean_hp_database(pkmn_database)
    p2_mean_hp = []
    for game in dataset:
        p2_team = ext_u.extract_p2_team_from_game_start(game)  # taking p2 team
        p2_team = pkmn_database[pkmn_database["name"].isin(p2_team)]
        p2_team = p2_team[["name", "base_hp"]]  # getting stats
        # calculating and appending mean hp
        p2_mean_hp.append(
            (np.sum(p2_team["base_hp"]) + mean_hp * (6 - len(p2_team))) / 6
        )
    mean_hp_start = pd.DataFrame({"p2_mean_hp_start": p2_mean_hp})
    return mean_hp_start



def mean_hp_last(dataset):  # feature
    """
    Calculate the mean base hp for the team of p1 and p2, and the difference
    between the two after the 30 turns for all games.
    Since we might not know all pokemons in a team, we used a standardized global mean.
    So for the pokemons we know, we'll use the sum of its base hp, for the ones we
    don't know, we add a global mean hp calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean hp. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """

    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_hp = ext_u.mean_hp_database(pkmn_database)
    p1_mean_hp = []
    p2_mean_hp = []
    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last(game)
        p2_team = ext_u.extract_p2_team_from_game_last(game)

        # calculating the mean hp for p1 team
        p1_team = pkmn_database[pkmn_database["name"].isin(p1_team)]
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_hp"]]
        p1_mean_hp.append((np.sum(p1_team["base_hp"]) + (mean_hp * (6 - p1_known))) / 6)

        # calculating the mean hp for p2 team
        p2_team = pkmn_database[pkmn_database["name"].isin(p2_team)]
        p2_team = p2_team[["name", "base_hp"]]
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_mean_hp.append((np.sum(p2_team["base_hp"]) + (mean_hp * (6 - p2_known))) / 6)

    mean_hp_last = pd.DataFrame(
        {"p1_mean_hp_last": p1_mean_hp, "p2_mean_hp_last": p2_mean_hp}
    )
    # calculating the difference between the two means
    # mean_hp_last['mean_hp_last_difference']=np.subtract.reduce(mean_hp_last[['p1_mean_hp_last','p2_mean_hp_last']],axis=1)
    mean_hp_last = mean_hp_last.fillna(value=0)
    return mean_hp_last


def p1_mean_hp_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the mean hp for the team of p1 after the 30 turns in a Dataframe-
    """
    return mean_hp_last(dataset)["p1_mean_hp_last"]


def p2_mean_hp_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the mean hp for the team of p2 after the 30 turns in a Dataframe-
    """
    return mean_hp_last(dataset)["p2_mean_hp_last"]


def mean_hp_last_difference(dataset) -> pd.DataFrame:  # feature
    """
    Returns the difference between the mean hps of the team p1 and p2
    after the 30 turns. Result are returned in a Dataframe.
    """
    return mean_hp_last(dataset)["mean_hp_last_difference"]

def p1_final_team_hp(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the mean total remaining HP of team P1 at the end of the battle (30 turns).
    Adds the HP_pct of all the Pokémon still alive multiplied by their base_hp.
    (hp_pct*base_hp) then does the mean. Results are returned in a Dataframe.
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    final_hp = []

    for game in dataset:
        # taking the last turn
        last_turn = game["battle_timeline"][-1]

        # Find all alive pokemons
        all_turns = pd.DataFrame(
            [turn["p1_pokemon_state"] for turn in game["battle_timeline"]]
        )
        alive_pokemon = []

        # For each pokemon find the last state
        for pkmn in game["p1_team_details"]:
            pkmn_name = pkmn["name"]
            pkmn_turns = all_turns[all_turns["name"] == pkmn_name]

            if len(pkmn_turns) > 0:
                last_state = pkmn_turns.iloc[-1]
                if last_state["status"] != "fnt":
                    # Calculate remaining HP: hp_pct * base_hp
                    hp_remaining = last_state["hp_pct"] * pkmn["base_hp"]
                    alive_pokemon.append(hp_remaining)

        # Calculating the final hp
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        hp_team_known = sum(alive_pokemon) if alive_pokemon else 0
        final_hp.append(
            hp_team_known + (ext_u.mean_hp_database(pkmn_database) * (6 - p1_known))
        )

    return pd.DataFrame({"p1_final_team_hp": final_hp})


def p2_final_team_hp(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the mean total remaining HP of team P2 at the end of the battle (30 turns).
    Adds the HP_pct of all the Pokémon still alive multiplied by their base_hp.
    (hp_pct*base_hp) then does the mean. Results are returned in a Dataframe.
    """
    pkmn_database = csv_u.open_pkmn_database_csv()
    final_hp = []

    for game in dataset:
        # find all p2 pokemon seen in battle
        all_turns = pd.DataFrame(
            [turn["p2_pokemon_state"] for turn in game["battle_timeline"]]
        )
        alive_pokemon = []

        # for each p2 pokemon, take its last state
        unique_pokemon = all_turns["name"].unique()

        for pkmn_name in unique_pokemon:
            pkmn_turns = all_turns[all_turns["name"] == pkmn_name]
            last_state = pkmn_turns.iloc[-1]

            if last_state["status"] != "fnt":
                # get base hp from database
                pkmn_info = pkmn_database[pkmn_database["name"] == pkmn_name]
                if len(pkmn_info) > 0:
                    base_hp = pkmn_info.iloc[0]["base_hp"]
                    # calculate hp
                    hp_remaining = last_state["hp_pct"] * base_hp
                    alive_pokemon.append(hp_remaining)

        # Calculating the final hp
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        hp_team_known = sum(alive_pokemon) if alive_pokemon else 0
        final_hp.append(
            hp_team_known + (ext_u.mean_hp_database(pkmn_database) * (6 - p2_known))
        )

    return pd.DataFrame({"p2_final_team_hp": final_hp})


def final_team_hp_difference(dataset) -> pd.DataFrame:  # feature
    """
    Difference between the final mean hp (not percentage or base) of the teams
    of p1 and p2 after the 30 turns. Results returned in a dataframe
    """
    p1_hp = p1_final_team_hp(dataset)  # taking p1 final mean hp
    p2_hp = p2_final_team_hp(dataset)  # taking p2 final mean hp

    # calculating the difference
    diff = p1_hp["p1_final_team_hp"] - p2_hp["p2_final_team_hp"]

    return pd.DataFrame({"final_team_hp_difference": diff})


def mean_spe_start(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the mean speed and the difference between them for the team of
    p1 and p2 at the start of the game for all games. Since we might not know all pokemons in a team,
    we used a standardized global mean. So for the pokemons we know, we'll use the sum of its base spe,
    for the ones we don't know, we add a global mean spe calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean spe. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """

    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spe = ext_u.mean_spe_database(pkmn_database)
    p1_mean_spe = []
    p2_mean_spe = []
    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_start(game)
        p2_team = ext_u.extract_p2_team_from_game_start(game)

        # calculating the mean spe for p1 team
        p1_team = pkmn_database[pkmn_database["name"].isin(p1_team)]
        p1_known = len(p1_team)
        p1_team = p1_team[["name", "base_spe"]]
        p1_mean_spe.append(
            (np.sum(p1_team["base_spe"]) + (mean_spe * (6 - p1_known))) / 6
        )

        # calculating the mean spe for p2 team
        p2_team = pkmn_database[pkmn_database["name"].isin(p2_team)]
        p2_known = len(p2_team)
        p2_team = p2_team[["name", "base_spe"]]
        p2_mean_spe.append(
            (np.sum(p2_team["base_spe"]) + (mean_spe * (6 - p2_known))) / 6
        )

    mean_spe_start = pd.DataFrame(
        {"p1_mean_spe_start": p1_mean_spe, "p2_mean_spe_start": p2_mean_spe}
    )
    # calculating the difference between the two means
    mean_spe_start["mean_spe_start_difference"] = np.subtract.reduce(
        mean_spe_start[["p1_mean_spe_start", "p2_mean_spe_start"]], axis=1
    )
    return mean_spe_start


def mean_spe_start_difference(dataset) -> pd.DataFrame:  # feature
    """
    Returns the difference between the mean spe of the team p1 and p2
    at the start. Result are returned in a Dataframe.
    """
    return mean_spe_start(dataset)["mean_spe_start_difference"]


def mean_spe_last(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the mean speed and the difference between them for the team of
    p1 and p2 after the 30 turns of the game for all games. Since we might not know all pokemons in a team,
    we used a standardized global mean. So for the pokemons we know, we'll use the sum of its base spe,
    for the ones we don't know, we add a global mean spe calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean spe. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """
    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spe = ext_u.mean_spe_database(pkmn_database)
    p1_mean_spe = []
    p2_mean_spe = []

    # multipliers for boosts
    multipliers = {
        -6: 2 / 8,
        -5: 2 / 7,
        -4: 2 / 6,
        -3: 2 / 5,
        -2: 2 / 4,
        -1: 2 / 3,
        0: 1,
        +1: 3 / 2,
        2: 4 / 2,
        3: 5 / 2,
        4: 6 / 2,
        5: 7 / 2,
        6: 8 / 2,
    }

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last_with_stats(game)
        p2_team = ext_u.extract_p2_team_from_game_last_with_stats(game)

        # getting information for p1 team
        p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_spe", "boosts", "status"]]
        # checking if p1 team is not fully exausted
        if len(p1_team) != 0:
            # calculate the spe applying eventually the boosts and debuffs
            p1_team["total"] = (
                p1_team["base_spe"]
                * multipliers[p1_team["boosts"][0]["spe"]]
                * [1 if elem != "par" else 0.25 for elem in p1_team["status"]]
            )
            val = np.sum(p1_team["total"])
            # calculating and appending the mean
            p1_mean_spe.append((val) + (mean_spe * (6 - p1_known)) / 6)
        else:
            p1_mean_spe.append(0)  # 0 if pokemon are all exausted

        # getting information for p2 team
        p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_team = p2_team[["name", "base_spe", "boosts", "status"]]
        # checking if p2 team is not fully exausted
        if len(p2_team) != 0:
            p2_team["total"] = (
                p2_team["base_spe"]
                * multipliers[p2_team["boosts"][0]["spe"]]
                * [1 if elem != "par" else 0.25 for elem in p2_team["status"]]
            )
            val = np.sum(p2_team["total"])
            # calculate the spe applying eventually the boosts and debuffs
            p2_mean_spe.append(val + (mean_spe * (6 - p2_known)) / 6)
        else:
            p2_mean_spe.append(0)  # 0 if pokemon are all exausted

    mean_spe_last = pd.DataFrame(
        {"p1_mean_spe_last": p1_mean_spe, "p2_mean_spe_last": p2_mean_spe}
    )
    # calculating the difference between the two means
    mean_spe_last["mean_spe_last_difference"] = np.subtract.reduce(
        mean_spe_last[["p1_mean_spe_last", "p2_mean_spe_last"]], axis=1
    )
    mean_spe_last = mean_spe_last.fillna(value=0)
    return mean_spe_last


def mean_spe_last_2(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the mean speed and the difference between them for the team of
    p1 and p2 after the 30 turns of the game for all games. Since we might not know all pokemons in a team,
    we used a standardized global mean. So for the pokemons we know, we'll use the sum of its base spe,
    for the ones we don't know, we add a global mean spe calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean spe. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """

    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spe = ext_u.mean_spe_database(pkmn_database)
    p1_mean_spe = []
    p2_mean_spe = []

    # multipliers for boosts
    multipliers = {
        -6: 2 / 8,
        -5: 2 / 7,
        -4: 2 / 6,
        -3: 2 / 5,
        -2: 2 / 4,
        -1: 2 / 3,
        0: 1,
        +1: 3 / 2,
        2: 4 / 2,
        3: 5 / 2,
        4: 6 / 2,
        5: 7 / 2,
        6: 8 / 2,
    }

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last_with_stats(game)
        p2_team = ext_u.extract_p2_team_from_game_last_with_stats(game)

        # getting information for p1 team
        p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_spe", "boosts", "status"]]

        # checking if p1 team is not fully exausted
        if len(p1_team) != 0:
            # calculate the spe applying eventually the boosts and debuffs
            p1_team["total"] = p1_team["base_spe"] * [
                1 if elem != "par" else 0.25 for elem in p1_team["status"]
            ]  # *multipliers[p1_team['boosts'][0]['spe']]
            val = np.sum(p1_team["total"])

            # calculating and appending the mean
            p1_mean_spe.append((val + (mean_spe * (6 - p1_known))) / 6)
        else:
            p1_mean_spe.append(0)  # 0 if pokemon are all exausted

        # getting information for p2 team
        p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_team = p2_team[["name", "base_spe", "boosts", "status"]]

        # checking if p2 team is not fully exausted
        if len(p2_team) != 0:
            # calculate the spe applying eventually the boosts and debuffs
            p2_team["total"] = p2_team["base_spe"] * [
                1 if elem != "par" else 0.25 for elem in p2_team["status"]
            ]  # *multipliers[p2_team['boosts'][0]['spe']]
            val = np.sum(p2_team["total"])

            # calculating and appending the mean
            p2_mean_spe.append((val + (mean_spe * (6 - p2_known))) / 6)

        else:
            p2_mean_spe.append(0)  # 0 if pokemon are all exausted

    mean_spe_last = pd.DataFrame(
        {"p1_mean_spe_last": p1_mean_spe, "p2_mean_spe_last": p2_mean_spe}
    )
    # calculating the difference between the two means
    # mean_spe_last['mean_spe_last_difference']=np.subtract.reduce(mean_spe_last[['p1_mean_spe_last','p2_mean_spe_last']],axis=1)
    mean_spe_last = mean_spe_last.fillna(value=0)
    return mean_spe_last


def p1_mean_spe_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the mean spe for the team of p1 after of the 30 turn.
    Results are returned in a Dataframe.
    """
    return mean_spe_last(dataset)["p1_mean_spe_last"]


def p2_mean_spe_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the mean spe for the team of p2 after of the 30 turn.
    Results are returned in a Dataframe.
    """
    return mean_spe_last(dataset)["p2_mean_spe_last"]


def mean_spe_last_difference(dataset) -> pd.DataFrame:  # feature
    """
    Returns the difference between the mean spe of the team p1 and p2
    after the 30 turns. Result are returned in a Dataframe.
    """
    return mean_spe_last(dataset)["mean_spe_last_difference"]


def sum_spe_last(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the sum speed for the team of p1 and p2 after the 30 turns of the game for all games.
    Since we might not know all pokemons in a team, we used a standardized global mean.
    So for the pokemons we know, we'll use the sum of its base spe, for the ones we don't know,
    we add a global mean spe calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean spe.
    Results are returned in a dataframe.
    """
    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spe = ext_u.mean_spe_database(pkmn_database)
    p1_sum_spe = []
    p2_sum_spe = []

    # multipliers for boosts
    multipliers = {
        -6: 2 / 8,
        -5: 2 / 7,
        -4: 2 / 6,
        -3: 2 / 5,
        -2: 2 / 4,
        -1: 2 / 3,
        0: 1,
        +1: 3 / 2,
        2: 4 / 2,
        3: 5 / 2,
        4: 6 / 2,
        5: 7 / 2,
        6: 8 / 2,
    }

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last_with_stats(game)
        p2_team = ext_u.extract_p2_team_from_game_last_with_stats(game)

        # getting information for p1 team
        p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_spe", "boosts", "status"]]
        # checking if p1 team is not fully exausted
        if len(p1_team) != 0:
            # calculate the spe applying eventually the boosts and debuffs
            p1_team["total"] = (
                p1_team["base_spe"]
                * multipliers[p1_team["boosts"][0]["spe"]]
                * [1 if elem != "par" else 0.25 for elem in p1_team["status"]]
            )
            val = np.sum(p1_team["total"])
            # calculating and appending the sum
            p1_sum_spe.append(val + (mean_spe * (6 - p1_known)))
        else:
            p1_sum_spe.append(0)  # 0 if pokemon are all exausted

        # getting information for p2 team
        p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_team = p2_team[["name", "base_spe", "boosts", "status"]]
        # checking if p2 team is not fully exausted
        if len(p2_team) != 0:
            p2_team["total"] = (
                p2_team["base_spe"]
                * multipliers[p2_team["boosts"][0]["spe"]]
                * [1 if elem != "par" else 0.25 for elem in p2_team["status"]]
            )
            val = np.sum(p2_team["total"])
            # calculate the spe applying eventually the boosts and debuffs
            p2_sum_spe.append(val + (mean_spe * (6 - p2_known)))
        else:
            p2_sum_spe.append(0)  # 0 if pokemon are all exausted

    sum_spe_last = pd.DataFrame(
        {"p1_sum_spe_last": p1_sum_spe, "p2_sum_spe_last": p2_sum_spe}
    )
    sum_spe_last = sum_spe_last.fillna(value=0)
    return sum_spe_last

def mean_atk_last(dataset):  # feature
    """
    Calculate the mean attack and the difference between them for the team of
    p1 and p2 after the 30 turns of the game for all games. Since we might not know all pokemons in a team,
    we used a standardized global mean. So for the pokemons we know, we'll use the sum of its base atk,
    for the ones we don't know, we add a global mean atk calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean atk. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """
    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_atk = ext_u.mean_atk_database(pkmn_database)
    p1_mean_atk = []
    p2_mean_atk = []

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last_with_stats(game)
        p2_team = ext_u.extract_p2_team_from_game_last_with_stats(game)

        # calculating the mean atk for p1 team
        p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_atk", "boosts", "status"]]
        p1_mean_atk.append(
            (
                np.sum(
                    p1_team["base_atk"]
                    * [1 if elem != "brn" else 0.5 for elem in p1_team["status"]]
                )
                + mean_atk * (6 - p1_known)
            )
            / 6
        )

        # calculating the mean atk for p2 team
        p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
        p2_team = p2_team[["name", "base_atk", "boosts", "status"]]
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_mean_atk.append(
            (
                np.sum(
                    p2_team["base_atk"]
                    * [1 if elem != "brn" else 0.5 for elem in p2_team["status"]]
                )
                + mean_atk * (6 - p2_known)
            )
            / 6
        )

    mean_atk_last = pd.DataFrame(
        {"p1_mean_atk_last": p1_mean_atk, "p2_mean_atk_last": p2_mean_atk}
    )
    # calculating the difference between the two means
    mean_atk_last["mean_atk_last_difference"] = np.subtract.reduce(
        mean_atk_last[["p1_mean_atk_last", "p2_mean_atk_last"]], axis=1
    )
    mean_atk_last = mean_atk_last.fillna(value=0)
    return mean_atk_last


def mean_atk_last_2(dataset):  # feature
    """
    Calculate the mean attack and the difference between them for the team of
    p1 and p2 after the 30 turns of the game for all games. Since we might not know all pokemons in a team,
    we used a standardized global mean. So for the pokemons we know, we'll use the sum of its base atk,
    for the ones we don't know, we add a global mean atk calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean atk. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """

    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_atk = ext_u.mean_atk_database(pkmn_database)
    p1_mean_atk = []
    p2_mean_atk = []

    # multipliers for boosts
    multipliers = {
        -6: 2 / 8,
        -5: 2 / 7,
        -4: 2 / 6,
        -3: 2 / 5,
        -2: 2 / 4,
        -1: 2 / 3,
        0: 1,
        +1: 3 / 2,
        2: 4 / 2,
        3: 5 / 2,
        4: 6 / 2,
        5: 7 / 2,
        6: 8 / 2,
    }

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last_with_stats(game)
        p2_team = ext_u.extract_p2_team_from_game_last_with_stats(game)

        # calculating the mean atk for p1 team
        p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_atk", "boosts", "status"]]

        # checking if p1 team is not fully exausted
        if len(p1_team) != 0:
            # calculate the spe applying eventually the boosts and debuffs
            p1_team["total"] = p1_team["base_atk"] * [
                1 if elem != "brn" else 0.5 for elem in p1_team["status"]
            ]  # *multipliers[p1_team['boosts'][0]['atk']]
            val = np.sum(p1_team["total"])

            # calculating and appending the mean
            p1_mean_atk.append((val + (mean_atk * (6 - p1_known))) / 6)
        else:
            p1_mean_atk.append(0)  # 0 if pokemon are all exausted

        # calculating the mean atk for p2 team
        p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
        p2_team = p2_team[["name", "base_atk", "boosts", "status"]]
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))

        # checking if p1 team is not fully exausted
        if len(p2_team) != 0:
            # calculate the spe applying eventually the boosts and debuffs
            p2_team["total"] = p2_team["base_atk"] * [
                1 if elem != "brn" else 0.5 for elem in p2_team["status"]
            ]  # *multipliers[p2_team['boosts'][0]['atk']]
            val = np.sum(p2_team["total"])
            # calculating and appending the mean
            p2_mean_atk.append((val + (mean_atk * (6 - p2_known))) / 6)

        else:
            p2_mean_atk.append(0)  # 0 if pokemon are all exausted

    mean_atk_last = pd.DataFrame(
        {"p1_mean_atk_last": p1_mean_atk, "p2_mean_atk_last": p2_mean_atk}
    )
    # calculating the difference between the two means
    # mean_atk_last['mean_atk_last_difference']=np.subtract.reduce(mean_atk_last[['p1_mean_atk_last','p2_mean_atk_last']],axis=1)
    mean_atk_last = mean_atk_last.fillna(value=0)
    return mean_atk_last


def p1_mean_atk_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the mean atk for the team of p1 after of the 30 turn.
    Results are returned in a Dataframe.
    """
    return mean_atk_last(dataset)["p1_mean_atk_last"]


def p2_mean_atk_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the mean atk for the team of p2 after of the 30 turn.
    Results are returned in a Dataframe.
    """
    return mean_atk_last(dataset)["p2_mean_atk_last"]


def mean_atk_last_difference(dataset) -> pd.DataFrame:  # feature
    """
    Returns the difference between the mean spe of the team p1 and p2
    after the 30 turns. Result are returned in a Dataframe.
    """
    return mean_atk_last(dataset)["mean_atk_last_difference"]


def sum_atk_last(dataset):  # feature
    """
    Calculate the sum attack for the team of p1 and p2 after the 30 turns of the game for all games.
    Since we might not know all pokemons in a team, we used a standardized global mean.
    So for the pokemons we know, we'll use the sum of its base atk, for the ones we don't know,
    we add a global mean atk calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean atk.
    Results are returned in a dataframe.
    """
    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_atk = ext_u.mean_atk_database(pkmn_database)
    p1_sum_atk = []
    p2_sum_atk = []

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last_with_stats(game)
        p2_team = ext_u.extract_p2_team_from_game_last_with_stats(game)

        # calculating the sum atk for p1 team
        p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_atk", "boosts", "status"]]
        p1_sum_atk.append(
            np.sum(
                p1_team["base_atk"]
                * [1 if elem != "brn" else 0.5 for elem in p1_team["status"]]
            )
            + mean_atk * (6 - p1_known)
        )

        # calculating the sum atk for p2 team
        p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
        p2_team = p2_team[["name", "base_atk", "boosts", "status"]]
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_sum_atk.append(
            np.sum(
                p2_team["base_atk"]
                * [1 if elem != "brn" else 0.5 for elem in p2_team["status"]]
            )
            + mean_atk * (6 - p2_known)
        )

    sum_atk_last = pd.DataFrame(
        {"p1_sum_atk_last": p1_sum_atk, "p2_sum_atk_last": p2_sum_atk}
    )
    sum_atk_last = sum_atk_last.fillna(value=0)
    return sum_atk_last

def mean_def_last(dataset):  # feature
    """
    Calculate the mean defense and the difference between them for the team of
    p1 and p2 after the 30 turns of the game for all games. Since we might not know all pokemons in a team,
    we used a standardized global mean. So for the pokemons we know, we'll use the sum of its base def,
    for the ones we don't know, we add a global mean def calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean def. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """

    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_def = ext_u.mean_def_database(pkmn_database)
    p1_mean_def = []
    p2_mean_def = []
    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last(game)
        p2_team = ext_u.extract_p2_team_from_game_last(game)

        # calculating the mean def for p1 team
        p1_team = pkmn_database[pkmn_database["name"].isin(p1_team)]
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_def"]]
        p1_mean_def.append(
            (np.sum(p1_team["base_def"]) + mean_def * (6 - p1_known)) / 6
        )

        # calculating the mean def for p2 team
        p2_team = pkmn_database[pkmn_database["name"].isin(p2_team)]
        p2_team = p2_team[["name", "base_def"]]
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_mean_def.append(
            (np.sum(p2_team["base_def"]) + mean_def * (6 - p2_known)) / 6
        )

    mean_def_last = pd.DataFrame(
        {"p1_mean_def_last": p1_mean_def, "p2_mean_def_last": p2_mean_def}
    )
    # calculating the difference between the two means
    mean_def_last["mean_def_last_difference"] = np.subtract.reduce(
        mean_def_last[["p1_mean_def_last", "p2_mean_def_last"]], axis=1
    )
    mean_def_last = mean_def_last.fillna(value=0)
    return mean_def_last


def mean_def_last_2(dataset):  # feature
    """
    Calculate the mean defense and the difference between them for the team of
    p1 and p2 after the 30 turns of the game for all games. Since we might not know all pokemons in a team,
    we used a standardized global mean. So for the pokemons we know, we'll use the sum of its base def,
    for the ones we don't know, we add a global mean def calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean def. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """

    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_def = ext_u.mean_def_database(pkmn_database)
    p1_mean_def = []
    p2_mean_def = []

    # multipliers for boosts
    multipliers = {
        -6: 2 / 8,
        -5: 2 / 7,
        -4: 2 / 6,
        -3: 2 / 5,
        -2: 2 / 4,
        -1: 2 / 3,
        0: 1,
        +1: 3 / 2,
        2: 4 / 2,
        3: 5 / 2,
        4: 6 / 2,
        5: 7 / 2,
        6: 8 / 2,
    }

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last_with_stats(game)
        p2_team = ext_u.extract_p2_team_from_game_last_with_stats(game)

        # calculating the mean def for p1 team
        p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_def", "boosts", "status"]]

        # checking if p1 team is not fully exausted
        if len(p1_team) != 0:
            # calculate the spe applying eventually the boosts and debuffs
            p1_team["total"] = p1_team[
                "base_def"
            ]  # *multipliers[p1_team['boosts'][0]['def']]
            val = np.sum(p1_team["total"])

            # calculating and appending the mean
            p1_mean_def.append((val + (mean_def * (6 - p1_known))) / 6)
        else:
            p1_mean_def.append(0)  # 0 if pokemon are all exausted

        # calculating the mean def for p2 team
        p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
        p2_team = p2_team[["name", "base_def", "boosts", "status"]]
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))

        # checking if p1 team is not fully exausted
        if len(p2_team) != 0:
            # calculate the spe applying eventually the boosts and debuffs
            p2_team["total"] = p2_team[
                "base_def"
            ]  # *multipliers[p2_team['boosts'][0]['def']]
            val = np.sum(p2_team["total"])
            # calculating and appending the mean
            p2_mean_def.append((val + (mean_def * (6 - p2_known))) / 6)

        else:
            p2_mean_def.append(0)  # 0 if pokemon are all exausted

    mean_def_last = pd.DataFrame(
        {"p1_mean_def_last": p1_mean_def, "p2_mean_def_last": p2_mean_def}
    )
    # calculating the difference between the two means
    # mean_def_last['mean_def_last_difference']=np.subtract.reduce(mean_def_last[['p1_mean_def_last','p2_mean_def_last']],axis=1)
    mean_def_last = mean_def_last.fillna(value=0)
    return mean_def_last


def p1_mean_def_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the mean def for the team of p1 after of the 30 turn.
    Results are returned in a Dataframe.
    """
    return mean_def_last(dataset)["p1_mean_def_last"]


def p2_mean_def_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the mean def for the team of p2 after of the 30 turn.
    Results are returned in a Dataframe.
    """
    return mean_def_last(dataset)["p2_mean_def_last"]


def mean_def_last_difference(dataset) -> pd.DataFrame:  # feature
    """
    Returns the difference between the mean def of the team p1 and p2
    after the 30 turns. Result are returned in a Dataframe.
    """
    return mean_def_last(dataset)["mean_def_last_difference"]


def sum_def_last(dataset):  # feature
    """
    Calculate the sum defense for the team of p1 and p2 after the 30 turns of the game for all games.
    Since we might not know all pokemons in a team, we used a standardized global mean.
    So for the pokemons we know, we'll use the sum of its base def, for the ones we don't know,
    we add a global mean def calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean def.
    Results are returned in a dataframe.
    """
    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_def = ext_u.mean_def_database(pkmn_database)
    p1_sum_def = []
    p2_sum_def = []
    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last(game)
        p2_team = ext_u.extract_p2_team_from_game_last(game)

        # calculating the sum def for p1 team
        p1_team = pkmn_database[pkmn_database["name"].isin(p1_team)]
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_def"]]
        p1_sum_def.append(np.sum(p1_team["base_def"]) + mean_def * (6 - p1_known))

        # calculating the sum def for p2 team
        p2_team = pkmn_database[pkmn_database["name"].isin(p2_team)]
        p2_team = p2_team[["name", "base_def"]]
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_sum_def.append(np.sum(p2_team["base_def"]) + mean_def * (6 - p2_known))

    sum_def_last = pd.DataFrame(
        {"p1_sum_def_last": p1_sum_def, "p2_sum_def_last": p2_sum_def}
    )
    sum_def_last = sum_def_last.fillna(value=0)
    return sum_def_last


def mean_spa_last(dataset):  # feature
    """
    Calculate the mean special attack and the difference between them for the team of
    p1 and p2 after the 30 turns of the game for all games. Since we might not know all pokemons in a team,
    we used a standardized global mean. So for the pokemons we know, we'll use the sum of its base spa,
    for the ones we don't know, we add a global mean spa calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean spa. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """

    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spa = ext_u.mean_spa_database(pkmn_database)
    p1_mean_spa = []
    p2_mean_spa = []

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last(game)
        p2_team = ext_u.extract_p2_team_from_game_last(game)

        # calculating the mean spa for p1 team
        p1_team = pkmn_database[pkmn_database["name"].isin(p1_team)]
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_spa"]]
        p1_mean_spa.append(
            (np.sum(p1_team["base_spa"]) + mean_spa * (6 - p1_known)) / 6
        )

        # calculating the mean spa for p2 team
        p2_team = pkmn_database[pkmn_database["name"].isin(p2_team)]
        p2_team = p2_team[["name", "base_spa"]]
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_mean_spa.append(
            (np.sum(p2_team["base_spa"]) + mean_spa * (6 - p2_known)) / 6
        )

    mean_spa_last = pd.DataFrame(
        {"p1_mean_spa_last": p1_mean_spa, "p2_mean_spa_last": p2_mean_spa}
    )
    # calculating the difference between the two means
    mean_spa_last["mean_spa_last_difference"] = np.subtract.reduce(
        mean_spa_last[["p1_mean_spa_last", "p2_mean_spa_last"]], axis=1
    )
    mean_spa_last = mean_spa_last.fillna(value=0)
    return mean_spa_last


def mean_spa_last_2(dataset):  # feature
    """
    Calculate the mean special attack and the difference between them for the team of
    p1 and p2 after the 30 turns of the game for all games. Since we might not know all pokemons in a team,
    we used a standardized global mean. So for the pokemons we know, we'll use the sum of its base spa,
    for the ones we don't know, we add a global mean spa calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean spa. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """

    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spa = ext_u.mean_spa_database(pkmn_database)
    p1_mean_spa = []
    p2_mean_spa = []

    # multipliers for boosts
    multipliers = {
        -6: 2 / 8,
        -5: 2 / 7,
        -4: 2 / 6,
        -3: 2 / 5,
        -2: 2 / 4,
        -1: 2 / 3,
        0: 1,
        +1: 3 / 2,
        2: 4 / 2,
        3: 5 / 2,
        4: 6 / 2,
        5: 7 / 2,
        6: 8 / 2,
    }

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last_with_stats(game)
        p2_team = ext_u.extract_p2_team_from_game_last_with_stats(game)

        # calculating the mean spa for p1 team
        p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_spa", "boosts", "status"]]

        # checking if p1 team is not fully exausted
        if len(p1_team) != 0:
            # calculate the spe applying eventually the boosts and debuffs
            p1_team["total"] = p1_team[
                "base_spa"
            ]  # *multipliers[p1_team['boosts'][0]['spa']]
            val = np.sum(p1_team["total"])

            # calculating and appending the mean
            p1_mean_spa.append((val + (mean_spa * (6 - p1_known))) / 6)
        else:
            p1_mean_spa.append(0)  # 0 if pokemon are all exausted

        # calculating the mean spa for p2 team
        p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
        p2_team = p2_team[["name", "base_spa", "boosts", "status"]]
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))

        # checking if p1 team is not fully exausted
        if len(p2_team) != 0:
            # calculate the spe applying eventually the boosts and debuffs
            p2_team["total"] = p2_team[
                "base_spa"
            ]  # *multipliers[p2_team['boosts'][0]['spa']]
            val = np.sum(p2_team["total"])
            # calculating and appending the mean
            p2_mean_spa.append((val + (mean_spa * (6 - p2_known))) / 6)

        else:
            p2_mean_spa.append(0)  # 0 if pokemon are all exausted

    mean_spa_last = pd.DataFrame(
        {"p1_mean_spa_last": p1_mean_spa, "p2_mean_spa_last": p2_mean_spa}
    )
    # calculating the difference between the two means
    # mean_spa_last['mean_spa_last_difference']=np.subtract.reduce(mean_spa_last[['p1_mean_spa_last','p2_mean_spa_last']],axis=1)
    mean_spa_last = mean_spa_last.fillna(value=0)
    return mean_spa_last


def p1_mean_spa_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the mean spa for the team of p1 after of the 30 turn.
    Results are returned in a Dataframe.
    """
    return mean_spa_last(dataset)["p1_mean_spa_last"]


def p2_mean_spa_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the mean spa for the team of p2 after of the 30 turn.
    Results are returned in a Dataframe.
    """
    return mean_spa_last(dataset)["p2_mean_spa_last"]


def mean_spa_last_difference(dataset) -> pd.DataFrame:  # feature
    """
    Returns the difference between the mean spa of the team p1 and p2
    after the 30 turns. Result are returned in a Dataframe.
    """
    return mean_spa_last(dataset)["mean_spa_last_difference"]


def sum_spa_last(dataset):  # feature
    """
    Calculate the sum special attack for the team of p1 and p2 after the 30 turns of the game for all games.
    Since we might not know all pokemons in a team, we used a standardized global mean.
    So for the pokemons we know, we'll use the sum of its base spa, for the ones we don't know,
    we add a global mean spa calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean spa.
    Results are returned in a dataframe.
    """
    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spa = ext_u.mean_spa_database(pkmn_database)
    p1_sum_spa = []
    p2_sum_spa = []

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last(game)
        p2_team = ext_u.extract_p2_team_from_game_last(game)

        # calculating the sum spa for p1 team
        p1_team = pkmn_database[pkmn_database["name"].isin(p1_team)]
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_spa"]]
        p1_sum_spa.append(np.sum(p1_team["base_spa"]) + mean_spa * (6 - p1_known))

        # calculating the sum spa for p2 team
        p2_team = pkmn_database[pkmn_database["name"].isin(p2_team)]
        p2_team = p2_team[["name", "base_spa"]]
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_sum_spa.append(np.sum(p2_team["base_spa"]) + mean_spa * (6 - p2_known))

    sum_spa_last = pd.DataFrame(
        {"p1_sum_spa_last": p1_sum_spa, "p2_sum_spa_last": p2_sum_spa}
    )
    sum_spa_last = sum_spa_last.fillna(value=0)
    return sum_spa_last

def mean_spd_last(dataset):  # feature
    """
    Calculate the mean special defense and the difference between them for the team of
    p1 and p2 after the 30 turns of the game for all games. Since we might not know all pokemons in a team,
    we used a standardized global mean. So for the pokemons we know, we'll use the sum of its base spd,
    for the ones we don't know, we add a global mean spd calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean spd. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """

    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spd = ext_u.mean_spd_database(pkmn_database)
    p1_mean_spd = []
    p2_mean_spd = []
    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last(game)
        p2_team = ext_u.extract_p2_team_from_game_last(game)

        # calculating the mean spa for p1 team
        p1_team = pkmn_database[pkmn_database["name"].isin(p1_team)]
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_spd"]]
        p1_mean_spd.append(
            (np.sum(p1_team["base_spd"]) + mean_spd * (6 - p1_known)) / 6
        )

        # calculating the mean spd for p2 team
        p2_team = pkmn_database[pkmn_database["name"].isin(p2_team)]
        p2_team = p2_team[["name", "base_spd"]]
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_mean_spd.append(
            (np.sum(p2_team["base_spd"]) + mean_spd * (6 - p2_known)) / 6
        )

    mean_spd_last = pd.DataFrame(
        {"p1_mean_spd_last": p1_mean_spd, "p2_mean_spd_last": p2_mean_spd}
    )
    # calculating the difference between the two means
    mean_spd_last["mean_spd_last_difference"] = np.subtract.reduce(
        mean_spd_last[["p1_mean_spd_last", "p2_mean_spd_last"]], axis=1
    )
    mean_spd_last = mean_spd_last.fillna(value=0)
    return mean_spd_last


def mean_spd_last_2(dataset):  # feature
    """
    Calculate the mean special defense and the difference between them for the team of
    p1 and p2 after the 30 turns of the game for all games. Since we might not know all pokemons in a team,
    we used a standardized global mean. So for the pokemons we know, we'll use the sum of its base spd,
    for the ones we don't know, we add a global mean spd calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean spd. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """

    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spd = ext_u.mean_spd_database(pkmn_database)
    p1_mean_spd = []
    p2_mean_spd = []

    # multipliers for boosts
    multipliers = {
        -6: 2 / 8,
        -5: 2 / 7,
        -4: 2 / 6,
        -3: 2 / 5,
        -2: 2 / 4,
        -1: 2 / 3,
        0: 1,
        +1: 3 / 2,
        2: 4 / 2,
        3: 5 / 2,
        4: 6 / 2,
        5: 7 / 2,
        6: 8 / 2,
    }

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last_with_stats(game)
        p2_team = ext_u.extract_p2_team_from_game_last_with_stats(game)

        # calculating the mean spa for p1 team
        p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_spd", "boosts", "status"]]

        # checking if p1 team is not fully exausted
        if len(p1_team) != 0:
            # calculate the spd applying eventually the boosts and debuffs
            p1_team["total"] = p1_team[
                "base_spd"
            ]  # *multipliers[p1_team['boosts'][0]['spd']]
            val = np.sum(p1_team["total"])

            # calculating and appending the mean
            p1_mean_spd.append((val + (mean_spd * (6 - p1_known))) / 6)
        else:
            p1_mean_spd.append(0)  # 0 if pokemon are all exausted

        # calculating the mean spd for p2 team
        p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
        p2_team = p2_team[["name", "base_spd", "boosts", "status"]]
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))

        # checking if p1 team is not fully exausted
        if len(p2_team) != 0:
            # calculate the spe applying eventually the boosts and debuffs
            p2_team["total"] = p2_team[
                "base_spd"
            ]  # *multipliers[p2_team['boosts'][0]['spd']]
            val = np.sum(p2_team["total"])
            # calculating and appending the mean
            p2_mean_spd.append((val + (mean_spd * (6 - p2_known))) / 6)

        else:
            p2_mean_spd.append(0)  # 0 if pokemon are all exausted

    mean_spd_last = pd.DataFrame(
        {"p1_mean_spd_last": p1_mean_spd, "p2_mean_spd_last": p2_mean_spd}
    )
    # calculating the difference between the two means
    # mean_spd_last['mean_spd_last_difference']=np.subtract.reduce(mean_spd_last[['p1_mean_spd_last','p2_mean_spd_last']],axis=1)
    mean_spd_last = mean_spd_last.fillna(value=0)
    return mean_spd_last


def p1_mean_spd_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the mean spd for the team of p1 after of the 30 turn.
    Results are returned in a Dataframe.
    """
    return mean_spd_last(dataset)["p1_mean_spd_last"]


def p2_mean_spd_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the mean spd for the team of p2 after of the 30 turn.
    Results are returned in a Dataframe.
    """
    return mean_spd_last(dataset)["p2_mean_spd_last"]


def mean_spd_last_difference(dataset) -> pd.DataFrame:  # feature
    """
    Returns the difference between the mean spd of the team p1 and p2
    after the 30 turns. Result are returned in a Dataframe.
    """
    return mean_spd_last(dataset)["mean_spd_last_difference"]


def sum_spd_last(dataset):  # feature
    """
    Calculate the sum special defense for the team of p1 and p2 after the 30 turns of the game for all games.
    Since we might not know all pokemons in a team, we used a standardized global mean.
    So for the pokemons we know, we'll use the sum of its base spd, for the ones we don't know,
    we add a global mean spd calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean spd.
    Results are returned in a dataframe.
    """
    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_spd = ext_u.mean_spd_database(pkmn_database)
    p1_sum_spd = []
    p2_sum_spd = []
    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last(game)
        p2_team = ext_u.extract_p2_team_from_game_last(game)

        # calculating the sum spd for p1 team
        p1_team = pkmn_database[pkmn_database["name"].isin(p1_team)]
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[["name", "base_spd"]]
        p1_sum_spd.append(np.sum(p1_team["base_spd"]) + mean_spd * (6 - p1_known))

        # calculating the sum spd for p2 team
        p2_team = pkmn_database[pkmn_database["name"].isin(p2_team)]
        p2_team = p2_team[["name", "base_spd"]]
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_sum_spd.append(np.sum(p2_team["base_spd"]) + mean_spd * (6 - p2_known))

    sum_spd_last = pd.DataFrame(
        {"p1_sum_spd_last": p1_sum_spd, "p2_sum_spd_last": p2_sum_spd}
    )
    sum_spd_last = sum_spd_last.fillna(value=0)
    return sum_spd_last


def mean_stats_last(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the mean of all stats summed and the difference between them for the team of
    p1 and p2 after the 30 turns of the game for all games. Since we might not know all pokemons of the team,
    we used a standardized global mean. So for the pokemons we know, we'll use the sum of its total stats,
    for the ones we don't know, we add a global mean total calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean total. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """

    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_total = ext_u.mean_total_database(pkmn_database)
    p1_mean_stats = []
    p2_mean_stats = []

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last(game).to_frame()
        p2_team = ext_u.extract_p2_team_from_game_last(game).to_frame()

        # getting information for p1 team
        p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team["total"] = np.sum(
            p1_team[
                ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
            ],
            axis=1,
        )
        p1_team = p1_team[["name", "total"]]
        p1_mean_stats.append(
            (np.sum(p1_team["total"]) + (mean_total * (6 - p1_known))) / 6
        )

        # getting information for p2 team
        p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_team["total"] = np.sum(
            p2_team[
                ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
            ],
            axis=1,
        )
        p2_team = p2_team[["name", "total"]]
        p2_mean_stats.append(
            (np.sum(p2_team["total"]) + mean_total * (6 - p2_known)) / 6
        )

    mean_stats = pd.DataFrame(
        {"p1_mean_stats_last": p1_mean_stats, "p2_mean_stats_last": p2_mean_stats}
    )
    # calculating the difference between the two means
    mean_stats["mean_stats_last_difference"] = np.subtract.reduce(
        mean_stats[["p1_mean_stats_last", "p2_mean_stats_last"]], axis=1
    )
    return mean_stats


def mean_stats_last_2(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the mean of all stats summed and the difference between them for the team of
    p1 and p2 after the 30 turns of the game for all games. Since we might not know all pokemons of the team,
    we used a standardized global mean. So for the pokemons we know, we'll use the sum of its total stats,
    for the ones we don't know, we add a global mean total calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean total. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """

    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_total = ext_u.mean_total_database(pkmn_database)
    p1_mean_stats = []
    p2_mean_stats = []

    # multipliers for boosts
    multipliers = {
        -6: 2 / 8,
        -5: 2 / 7,
        -4: 2 / 6,
        -3: 2 / 5,
        -2: 2 / 4,
        -1: 2 / 3,
        0: 1,
        +1: 3 / 2,
        2: 4 / 2,
        3: 5 / 2,
        4: 6 / 2,
        5: 7 / 2,
        6: 8 / 2,
    }

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last_with_stats(game)
        p2_team = ext_u.extract_p2_team_from_game_last_with_stats(game)

        # getting information for p1 team
        p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team = p1_team[
            [
                "name",
                "base_hp",
                "base_atk",
                "base_def",
                "base_spa",
                "base_spd",
                "base_spe",
                "boosts",
                "status",
            ]
        ]

        # checking if p1 team is not fully exausted
        if len(p1_team) != 0:
            # calculate the total stats applying eventually the boosts and debuffs
            # p1_team['base_atk']=p1_team['base_atk']*multipliers[p1_team['boosts'][0]['atk']] *[1 if elem!='brn' else 0.5 for elem in p1_team['status']]
            p1_team["base_atk"] = p1_team["base_atk"] * [
                1 if elem != "brn" else 0.5 for elem in p1_team["status"]
            ]
            p1_team["base_def"] = p1_team[
                "base_def"
            ]  # *multipliers[p1_team['boosts'][0]['def']]
            p1_team["base_spa"] = p1_team[
                "base_spa"
            ]  # *multipliers[p1_team['boosts'][0]['spa']]
            p1_team["base_spd"] = p1_team[
                "base_spd"
            ]  # *multipliers[p1_team['boosts'][0]['spd']]
            # p1_team['base_spe']=p1_team['base_spe']*multipliers[p1_team['boosts'][0]['spe']]*[1 if elem!='par' else 0.25 for elem in p1_team['status']]
            p1_team["base_spe"] = p1_team["base_spe"] * [
                1 if elem != "par" else 0.25 for elem in p1_team["status"]
            ]

            p1_team["total"] = np.sum(
                p1_team[
                    [
                        "base_hp",
                        "base_atk",
                        "base_def",
                        "base_spa",
                        "base_spd",
                        "base_spe",
                    ]
                ],
                axis=1,
            )
            p1_team = p1_team[["name", "total"]]
            # calculating and appending the mean
            p1_mean_stats.append(
                (np.sum(p1_team["total"]) + (mean_total * (6 - p1_known))) / 6
            )
        else:
            p1_mean_stats.append(0)  # 0 if pokemon are all exausted

        # getting information for p2 team
        p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_team = p2_team[
            [
                "name",
                "base_hp",
                "base_atk",
                "base_def",
                "base_spa",
                "base_spd",
                "base_spe",
                "boosts",
                "status",
            ]
        ]

        # checking if p2 team is not fully exausted
        if len(p2_team) != 0:
            # calculate the total stats applying eventually the boosts and debuffs
            # p2_team['base_atk']=p2_team['base_atk']*multipliers[p2_team['boosts'][0]['atk']]*[1 if elem!='brn' else 0.5 for elem in p2_team['status']]
            p2_team["base_atk"] = p2_team["base_atk"] * [
                1 if elem != "brn" else 0.5 for elem in p2_team["status"]
            ]
            p2_team["base_def"] = p2_team[
                "base_def"
            ]  # *multipliers[p2_team['boosts'][0]['def']]
            p2_team["base_spa"] = p2_team[
                "base_spa"
            ]  # *multipliers[p2_team['boosts'][0]['spa']]
            p2_team["base_spd"] = p2_team[
                "base_spd"
            ]  # *multipliers[p2_team['boosts'][0]['spd']]
            # p2_team['base_spe']=p2_team['base_spe']*multipliers[p2_team['boosts'][0]['spe']]*[1 if elem!='par' else 0.25 for elem in p2_team['status']]
            p2_team["base_spe"] = p2_team["base_spe"] * [
                1 if elem != "par" else 0.25 for elem in p2_team["status"]
            ]

            p2_team["total"] = np.sum(
                p2_team[
                    [
                        "base_hp",
                        "base_atk",
                        "base_def",
                        "base_spa",
                        "base_spd",
                        "base_spe",
                    ]
                ],
                axis=1,
            )
            p2_team = p2_team[["name", "total"]]
            # calculating and appending the mean
            p2_mean_stats.append(
                (np.sum(p2_team["total"]) + (mean_total * (6 - p2_known))) / 6
            )
        else:
            p2_mean_stats.append(0)  # 0 if pokemon are all exausted
    mean_stats = pd.DataFrame(
        {"p1_mean_stats_last": p1_mean_stats, "p2_mean_stats_last": p2_mean_stats}
    )
    # calculating the difference between the two means
    # mean_stats['mean_stats_last_difference']=np.subtract.reduce(mean_stats[['p1_mean_stats_last','p2_mean_stats_last']],axis=1)
    return mean_stats


def p1_mean_stats_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the mean total stats for the team of p1 after of the 30 turns.
    Results are returned in a Dataframe.
    """
    return mean_stats_last(dataset)["p1_mean_stats_last"]


def p2_mean_stats_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the mean total stats for the team of p2 after of the 30 turns.
    Results are returned in a Dataframe.
    """
    return mean_stats_last(dataset)["p2_mean_stats_last"]


def difference_mean_stats_last(dataset) -> pd.DataFrame:  # feature
    """
    Returns the difference between the mean total stats of the team p1 and p2
    after the 30 turns. Result are returned in a Dataframe.
    """
    return mean_stats_last(dataset)["mean_stats_last_difference"]


def sum_stats_last(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the sum of all stats for the team of p1 and p2 after the 30 turns of the game for all games.
    Since we might not know all pokemons of the team, we used a standardized global mean.
    So for the pokemons we know, we'll use the sum of its total stats, for the ones we don't know,
    we add a global mean total calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean total.
    Results are returned in a dataframe.
    """
    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_total = ext_u.mean_total_database(pkmn_database)
    p1_sum_stats = []
    p2_sum_stats = []

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last(game).to_frame()
        p2_team = ext_u.extract_p2_team_from_game_last(game).to_frame()

        # getting information for p1 team
        p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        p1_team["total"] = np.sum(
            p1_team[
                ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
            ],
            axis=1,
        )
        p1_team = p1_team[["name", "total"]]
        p1_sum_stats.append(np.sum(p1_team["total"]) + (mean_total * (6 - p1_known)))

        # getting information for p2 team
        p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_team["total"] = np.sum(
            p2_team[
                ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
            ],
            axis=1,
        )
        p2_team = p2_team[["name", "total"]]
        p2_sum_stats.append(np.sum(p2_team["total"]) + mean_total * (6 - p2_known))

    sum_stats = pd.DataFrame(
        {"p1_sum_stats_last": p1_sum_stats, "p2_sum_stats_last": p2_sum_stats}
    )
    return sum_stats

# ----Feature Crit----#
def mean_crit_2(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the mean crit chance for the team p1 and p2 at the end of the 30 turns for all games.
    The crit chance is calculated with base_spe/512, so we first divide all base_spe of pokemons in a team
    by 512 and the calculate the mean. Since we might not know all pokemons of the team, we used a standardized global mean.
    So for the pokemons we know, we'll use the sum of its base crit chance, for the ones we don't know,
    we add a global mean spe calculated from all pokemons species in the train.
    So if we don't know 2 pokemon, we add 2 times the global mean crit. Doing this
    we can calculated the mean by 6 without having imbalances.
    Results are returned in a dataframe.
    """

    # opening databases and taking the global mean
    pkmn_database = csv_u.open_pkmn_database_csv()
    mean_crit = ext_u.mean_crit_database(pkmn_database)
    p1_mean_crit = []
    p2_mean_crit = []

    for game in dataset:
        # taking teams for p1 and p2
        p1_team = ext_u.extract_p1_team_from_game_last_with_stats(game)
        p2_team = ext_u.extract_p2_team_from_game_last_with_stats(game)

        # getting information for p1 team
        p1_team = p1_team.merge(pkmn_database, how="inner", on="name")
        p1_known = len(ext_u.extract_p1_team_from_game_start(game))
        # checking if p1 team is not fully exausted
        p1_team = p1_team[["name", "base_spe", "boosts", "status"]]
        if len(p1_team) != 0:
            # calculating and appending the mean
            p1_team["total"] = p1_team["base_spe"] / 512
            val = np.sum(p1_team["total"])
            p1_mean_crit.append((val + (mean_crit * (6 - p1_known))) / 6)
        else:
            p1_mean_crit.append(0)  # 0 if all pokemons are exausted

        # getting information for p2 team
        p2_team = p2_team.merge(pkmn_database, how="inner", on="name")
        p2_known = len(ext_u.extract_p2_team_from_game_start(game))
        p2_team = p2_team[["name", "base_spe", "boosts", "status"]]
        # checking if p2 team is not fully exausted
        if len(p2_team) != 0:
            # calculating and appending the mean
            p2_team["total"] = p2_team["base_spe"] / 512
            val = np.sum(p2_team["total"])
            p2_mean_crit.append((val + (mean_crit * (6 - p2_known))) / 6)
        else:
            p2_mean_crit.append(0)  # 0 if all pokemons are exausted

    mean_crit = pd.DataFrame(
        {"p1_mean_crit": p1_mean_crit, "p2_mean_crit": p2_mean_crit}
    )
    mean_crit = mean_crit.fillna(value=0)
    return mean_crit
