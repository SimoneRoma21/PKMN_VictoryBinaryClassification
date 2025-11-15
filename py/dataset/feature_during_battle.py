import pandas as pd
import numpy as np

from .csv_utilities import *


def p1_alive_pkmn(dataset) -> pd.DataFrame:  # feature
    """
    Counts the number of pokemon alive after the 30 turns of the
    player p1 per each game. The num of pokemon is calculated with (6-pkmn_dead).
    The results is returned as a dataframe.
    """
    pkmn_alive_p1 = []
    for game in dataset:
        # extracting turns of p1
        turns = pd.DataFrame(
            [turn["p1_pokemon_state"] for turn in game["battle_timeline"]]
        )
        # getting the dead pokemons
        pkmn_dead_p1 = turns[turns["status"] == "fnt"]["name"].drop_duplicates(
            keep="last"
        )
        # appending num of pokemon alive
        pkmn_alive_p1.append(6 - len(pkmn_dead_p1))
    pkmn_alive_p1 = pd.DataFrame(pkmn_alive_p1).rename(columns={0: "p1_pkmn_alive"})
    return pkmn_alive_p1


def p2_alive_pkmn(dataset) -> pd.DataFrame:  # feature
    """
    Counts the number of pokemon alive after the 30 turns of the
    player p2 per each game. The num of pokemon is calculated with (6-pkmn_dead).
    The results is returned as a dataframe.
    """
    pkmn_alive_p2 = []
    for game in dataset:
        # extracting turns of p2
        turns = pd.DataFrame(
            [turn["p2_pokemon_state"] for turn in game["battle_timeline"]]
        )
        # getting the dead pokemons
        pkmn_dead_p2 = turns[turns["status"] == "fnt"]["name"].drop_duplicates(
            keep="last"
        )
        # appending num of pokemon alive
        pkmn_alive_p2.append(6 - len(pkmn_dead_p2))
    pkmn_alive_p2 = pd.DataFrame(pkmn_alive_p2).rename(columns={0: "p2_pkmn_alive"})
    return pkmn_alive_p2


def alive_pkmn_difference(dataset) -> pd.DataFrame:  # feature
    """
    Returns the difference of alive pokemons between p1 and p2 per
    each game. The data is returned in a dataframe.
    """
    p1_alive = p1_alive_pkmn(dataset)  # getting p1 alive pokemon
    p2_alive = p2_alive_pkmn(dataset)  # getting p2 alive pokemon

    # calculating difference
    pkmn_alive_difference = pd.concat([p1_alive, p2_alive], axis=1)
    pkmn_alive_difference["pkmn_alive_difference"] = np.subtract.reduce(
        pkmn_alive_difference[["p1_pkmn_alive", "p2_pkmn_alive"]], axis=1
    )
    return pkmn_alive_difference["pkmn_alive_difference"]


def p1_pokemon_stab(dataset) -> pd.DataFrame:  # feature
    """
    Counts the number of stab moves used by the pokemon of p1 per each game.
    A move is considered stab if the type of the move and the type of the
    pokemons are the same.
    The results are returned in a dataframe.
    """
    # opening pokemon database created from the train
    pkmn_database = open_pkmn_database_csv()
    p1_count = []
    for game in dataset:
        # getting pokemon and moves of p1
        turns_p1_state = pd.DataFrame(
            [turn["p1_pokemon_state"] for turn in game["battle_timeline"]]
        )
        turns_p1_state = turns_p1_state.merge(
            pkmn_database, how="inner", on="name"
        ).rename(columns={"name": "pkmn_name"})[
            ["pkmn_name", "status", "type_1", "type_2"]
        ]
        turns_p1_moves = pd.DataFrame(
            [
                (
                    turn["p1_move_details"]
                    if turn["p1_move_details"] != None
                    else {"name": "None", "type": "None", "category": "None"}
                )
                for turn in game["battle_timeline"]
            ]
        )[["name", "type", "category"]]
        turns_p1_moves = turns_p1_moves.rename(columns={"type": "move_type"})

        # checking if pokemons aren't all dead
        if len(turns_p1_state) != 0 and len(turns_p1_moves) != 0:
            turns = pd.concat([turns_p1_state, turns_p1_moves], axis=1).dropna()
            turns["move_type"] = turns["move_type"].apply(str.lower)
            turns["category"] = turns["category"].apply(str.lower)

            # picking only pokemon who did moves with corresponding type
            turns = turns.query(
                "category!='status' and ((type_1==move_type) or (type_2==move_type))"
            )
            p1_count.append(len(turns))  # appending count
        else:
            p1_count.append(0)  # appending 0 if pokemons are 0
    return pd.DataFrame({"p1_pkmn_stab_used": p1_count})


def p2_pokemon_stab(dataset) -> pd.DataFrame:  # feature
    """
    Counts the number of stab moves used by the pokemon of p2 per each game.
    A move is considered stab if the type of the move and the type of the
    pokemons are the same.
    The results are returned in a dataframe.
    """
    # opening pokemon database created from the train
    pkmn_database = open_pkmn_database_csv()
    p2_count = []

    for game in dataset:
        # getting pokemon and moves of p1
        turns_p2_state = pd.DataFrame(
            [turn["p2_pokemon_state"] for turn in game["battle_timeline"]]
        )
        turns_p2_state = turns_p2_state.merge(
            pkmn_database, how="inner", on="name"
        ).rename(columns={"name": "pkmn_name"})[
            ["pkmn_name", "status", "type_1", "type_2"]
        ]
        turns_p2_moves = pd.DataFrame(
            [
                (
                    turn["p2_move_details"]
                    if turn["p2_move_details"] != None
                    else {"name": "None", "type": "None", "category": "None"}
                )
                for turn in game["battle_timeline"]
            ]
        )[["name", "type", "category"]]
        turns_p2_moves = turns_p2_moves.rename(columns={"type": "move_type"})

        # checking if pokemons aren't all dead
        if len(turns_p2_state) != 0 and len(turns_p2_moves) != 0:
            turns = pd.concat([turns_p2_state, turns_p2_moves], axis=1).dropna()
            turns["move_type"] = turns["move_type"].apply(str.lower)
            turns["category"] = turns["category"].apply(str.lower)

            # picking only pokemon who did moves with corresponding type
            turns = turns.query(
                "category!='status' and ((type_1==move_type) or (type_2==move_type))"
            )
            p2_count.append(len(turns))  # appending count
        else:
            p2_count.append(0)  # appending 0 if pokemons are 0
    return pd.DataFrame({"p2_pkmn_stab_used": p2_count})


def p1_switches_count(dataset) -> pd.DataFrame:  # feature
    """
    Counts the number of times in which p1 switches a pokemon.
    Switches can represent offensive or defensive matchups.
    Results are returned in a Dataframe.
    """
    switches = []
    for game in dataset:
        switch_count = 0
        prev_pokemon = None

        for turn in game["battle_timeline"]:
            current_pokemon = turn["p1_pokemon_state"]["name"]
            if prev_pokemon is not None and current_pokemon != prev_pokemon:
                switch_count += 1
            prev_pokemon = current_pokemon

        switches.append(switch_count)

    return pd.DataFrame({"p1_switches_count": switches})


def p2_switches_count(dataset) -> pd.DataFrame:  # feature
    """
    Counts the number of times in which p1 switches a pokemon.
    Switches can represent offensive or defensive matchups.
    Results are returned in a Dataframe.
    """
    switches = []
    for game in dataset:
        switch_count = 0
        prev_pokemon = None

        for turn in game["battle_timeline"]:
            current_pokemon = turn["p2_pokemon_state"]["name"]
            if prev_pokemon is not None and current_pokemon != prev_pokemon:
                switch_count += 1
            prev_pokemon = current_pokemon

        switches.append(switch_count)

    return pd.DataFrame({"p2_switches_count": switches})


def p1_status_inflicted(dataset) -> pd.DataFrame:  # feature
    """
    Count how many status has p1 inflicted to p2.
    Include: paralysis (par), burn (brn), poison (psn), toxic (tox), sleep (slp), freeze (frz).
    Returns results in a dataframe.
    """
    status_count = []
    for game in dataset:
        count = 0
        prev_status = "nostatus"

        for turn in game["battle_timeline"]:
            current_status = turn["p2_pokemon_state"]["status"]

            # Counts when p2 switch from nostatus to a status (inflicted by p1)
            if prev_status == "nostatus" and current_status not in ["nostatus", "fnt"]:
                count += 1
            prev_status = current_status

        status_count.append(count)

    return pd.DataFrame({"p1_status_inflicted": status_count})


def p2_status_inflicted(dataset) -> pd.DataFrame:  # feature
    """
    Count how many status has p2 inflicted to p1.
    Include: paralysis (par), burn (brn), poison (psn), toxic (tox), sleep (slp), freeze (frz).
    Returns results in a dataframe.
    """
    status_count = []
    for game in dataset:
        count = 0
        pokemon_status_map = {}  # memorize status for pokemons

        for turn in game["battle_timeline"]:
            pokemon_name = turn["p1_pokemon_state"]["name"]
            current_status = turn["p1_pokemon_state"]["status"]

            # if pokemon is not in map, we add it
            if pokemon_name not in pokemon_status_map:
                pokemon_status_map[pokemon_name] = "nostatus"

            prev_status = pokemon_status_map[pokemon_name]

            # Counts when p1 switch from nostatus to a status (inflicted by p2)
            if prev_status == "nostatus" and current_status not in ["nostatus", "fnt"]:
                count += 1

            pokemon_status_map[pokemon_name] = current_status

        status_count.append(count)

    return pd.DataFrame({"p2_status_inflicted": status_count})


def switches_difference(dataset) -> pd.DataFrame:  # feature
    """
    Count switch differences between p1 and p2 per each game.
    Results are returned in a Dataframe.
    """
    p1_sw = p1_switches_count(dataset)
    p2_sw = p2_switches_count(dataset)

    diff = p1_sw["p1_switches_count"] - p2_sw["p2_switches_count"]

    return pd.DataFrame({"switches_difference": diff})


def status_inflicted_difference(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the difference of status inflicted between p1 and p2 per each game.
    Results are returned in a Dataframe.
    """
    p1_status = p1_status_inflicted(dataset)
    p2_status = p2_status_inflicted(dataset)

    diff = p1_status["p1_status_inflicted"] - p2_status["p2_status_inflicted"]

    return pd.DataFrame({"status_inflicted_difference": diff})


def p1_first_faint_turn(dataset) -> pd.DataFrame:  # feature
    """
    Calulate the turn in which P1 loses the first Pokémon.
    Higher values ​​indicate that P1 managed to hold out longer.
    Negative values ​​indicate the turn in which the first faint occurred
    (e.g., -3 means the first faint occurred on the third turn).
    Results per each game are returned in a dataframe.
    """
    first_faint = []

    for game in dataset:
        faint_turn = None
        pokemon_fainted = set()

        for turn in game["battle_timeline"]:
            pkmn_name = turn["p1_pokemon_state"]["name"]
            pkmn_status = turn["p1_pokemon_state"]["status"]

            # picking fainted pokemons
            if pkmn_status == "fnt" and pkmn_name not in pokemon_fainted:
                faint_turn = turn["turn"]
                pokemon_fainted.add(pkmn_name)
                break

        # if pokemon fainted, use the faint turn +1
        if faint_turn is None:
            faint_turn = -(len(game["battle_timeline"]) + 1)
        else:
            faint_turn = -faint_turn

        first_faint.append(faint_turn)

    return pd.DataFrame({"p1_first_faint_turn": first_faint})


def p1_avg_hp_when_switching(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the average/mean HP percentage of P1 Pokémon when switching.
    Low values ​​indicate defensive switches (Pokémon in difficulty).
    High values ​​indicate offensive/strategic switches.
    Results are returned in dataframe.
    """
    avg_hp_switches = []

    for game in dataset:
        switch_hp = []
        prev_pokemon = None

        for i, turn in enumerate(game["battle_timeline"]):
            current_pokemon = turn["p1_pokemon_state"]["name"]

            # if there has been a switch
            if prev_pokemon is not None and current_pokemon != prev_pokemon:
                # hp of pokemon switched out
                if i > 0:
                    prev_turn = game["battle_timeline"][i - 1]
                    switch_hp.append(prev_turn["p1_pokemon_state"]["hp_pct"])

            prev_pokemon = current_pokemon

        # average/mean hp when switching
        avg_hp = np.mean(switch_hp) if switch_hp else 1.0  # 1.0 se non ci sono switch
        avg_hp_switches.append(avg_hp)

    return pd.DataFrame({"p1_avg_hp_when_switching": avg_hp_switches})


def p2_avg_hp_when_switching(dataset) -> pd.DataFrame:  # feature
    """
    Calculate average/mean HP percentage of P2 Pokémon when they switch.
    Low values ​​indicate defensive switches (Pokémon in difficulty).
    High values ​​indicate offensive/strategic switches.
    Results are returned in dataframe.
    """
    avg_hp_switches = []

    for game in dataset:
        switch_hp = []
        prev_pokemon = None

        for i, turn in enumerate(game["battle_timeline"]):
            current_pokemon = turn["p2_pokemon_state"]["name"]

            # if there has been a switch
            if prev_pokemon is not None and current_pokemon != prev_pokemon:
                # hp of pokemon switched out
                if i > 0:
                    prev_turn = game["battle_timeline"][i - 1]
                    switch_hp.append(prev_turn["p2_pokemon_state"]["hp_pct"])

            prev_pokemon = current_pokemon

        # average/mean hp when switching
        avg_hp = np.mean(switch_hp) if switch_hp else 1.0  # 1.0 se non ci sono switch
        avg_hp_switches.append(avg_hp)

    return pd.DataFrame({"p2_avg_hp_when_switching": avg_hp_switches})


def p1_max_debuff_received(dataset) -> pd.DataFrame:  # feature
    """
    Count the max number of debuffs received by p1 pkmns between all stats per game.
    Results are returned in a dataframe.
    """
    max_debuff_list = []

    for game in dataset:
        max_debuff = 0

        for turn in game["battle_timeline"]:
            boosts = turn["p1_pokemon_state"]["boosts"]

            # search for the max bebuff between stats and update
            turn_min = min(boosts.values())
            if turn_min < max_debuff:
                max_debuff = turn_min

        max_debuff_list.append(max_debuff)  # appending max debuff

    return pd.DataFrame({"p1_max_debuff_received": max_debuff_list})


def p2_max_debuff_received(dataset) -> pd.DataFrame:  # feature
    """
    Count the max number of debuffs received by p2 pkmns between all stats per game.
    Results are returned in a dataframe.
    """
    max_debuff_list = []

    for game in dataset:
        max_debuff = 0

        for turn in game["battle_timeline"]:
            boosts = turn["p2_pokemon_state"]["boosts"]

            # search for the max bebuff between stats and update
            turn_min = min(boosts.values())
            if turn_min < max_debuff:
                max_debuff = turn_min

        max_debuff_list.append(max_debuff)  # appending max debuff

    return pd.DataFrame({"p2_max_debuff_received": max_debuff_list})


def p1_avg_move_power(dataset) -> pd.DataFrame:  # feature
    """
    Calculates the average/mean power of all attack moves used by p1 pokemons per game.
    Moves that don't do damage are not considered in the mean.
    Returns the data in a dataframe.
    """
    avg_powers = []

    for game in dataset:
        move_powers = []

        for turn in game["battle_timeline"]:
            move_details = turn["p1_move_details"]

            # control if the move does damage or not
            if move_details is not None and move_details["base_power"] > 0:
                move_powers.append(move_details["base_power"])

        # calculate the mean between the moves
        avg_power = np.mean(move_powers) if move_powers else 0
        avg_powers.append(avg_power)

    return pd.DataFrame({"p1_avg_move_power": avg_powers})


def p2_avg_move_power(dataset) -> pd.DataFrame:  # feature
    """
    Calculates the average/mean power of all attack moves used by p1 pokemons per game.
    Moves that don't do damage are not considered in the mean.
    Returns the data in a dataframe.
    """
    avg_powers = []

    for game in dataset:
        move_powers = []

        for turn in game["battle_timeline"]:
            move_details = turn["p2_move_details"]

            # control if the move does damage or not
            if move_details is not None and move_details["base_power"] > 0:
                move_powers.append(move_details["base_power"])

        # calculate the mean between the moves
        avg_power = np.mean(move_powers) if move_powers else 0
        avg_powers.append(avg_power)

    return pd.DataFrame({"p2_avg_move_power": avg_powers})


def avg_move_power_difference(dataset) -> pd.DataFrame:  # feature
    """
    Difference between the mean average power of p1 and p2 per each game.
    Return results in a Dataframe
    """
    p1_power = p1_avg_move_power(dataset)  # average power p1
    p2_power = p2_avg_move_power(dataset)  # average power p1

    # calculating difference
    diff = p1_power["p1_avg_move_power"] - p2_power["p2_avg_move_power"]

    return pd.DataFrame({"avg_move_power_difference": diff})


def p1_offensive_ratio(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the ratio betwen offensive (atk+spa) and defensive (def +spd) stats per p1 team.
    The ratio will be calculated for all games. Ratio is calculated (atk+spa)/(def+spd).
    Results are returned in a Dataframe.
    """
    ratios = []

    for game in dataset:
        # take all p1 pokemon
        p1_team = game["p1_team_details"]

        total_offensive = 0
        total_defensive = 0

        # summing stats
        for pokemon in p1_team:
            total_offensive += pokemon["base_atk"] + pokemon["base_spa"]
            total_defensive += pokemon["base_def"] + pokemon["base_spd"]

        # calculate the ratio minding eventually divisions by zero
        ratio = total_offensive / total_defensive if total_defensive > 0 else 0
        ratios.append(ratio)  # appending

    return pd.DataFrame({"p1_offensive_ratio": ratios})


def p2_offensive_ratio(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the ratio betwen offensive (atk+spa) and defensive (def +spd) stats per p2 team.
    The ratio will be calculated for all games. Ratio is calculated (atk+spa)/(def+spd).vb
    Since we don't know p2 pokemons at the start, we'll use the one seen in battle.
    Results are returned in a Dataframe.
    """
    pkmn_database = open_pkmn_database_csv()
    ratios = []

    for game in dataset:
        # find all p2 pokemon seen in the battle
        all_turns = pd.DataFrame(
            [turn["p2_pokemon_state"] for turn in game["battle_timeline"]]
        )
        unique_pokemon = all_turns["name"].unique()

        total_offensive = 0
        total_defensive = 0

        for pkmn_name in unique_pokemon:
            # obtain stats from the database of pokemon created with trains species
            pkmn_info = pkmn_database[pkmn_database["name"] == pkmn_name]

            # if pokemon is present in database
            if len(pkmn_info) > 0:
                # summing stats
                pkmn_stats = pkmn_info.iloc[0]
                total_offensive += pkmn_stats["base_atk"] + pkmn_stats["base_spa"]
                total_defensive += pkmn_stats["base_def"] + pkmn_stats["base_spd"]

        # calculate the ratio minding eventually divisions by zero
        ratio = total_offensive / total_defensive if total_defensive > 0 else 0
        ratios.append(ratio)  # appending

    return pd.DataFrame({"p2_offensive_ratio": ratios})


def offensive_ratio_difference(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the difference between the offensive ratio of p1 and p2 for each game.
    Results are returned in a Dataframe
    """
    p1_ratio = p1_offensive_ratio(dataset)  # calc p1 offensive ratio
    p2_ratio = p2_offensive_ratio(dataset)  # calc p2 offensive ratio

    # calculating difference
    diff = p1_ratio["p1_offensive_ratio"] - p2_ratio["p2_offensive_ratio"]

    return pd.DataFrame({"offensive_ratio_difference": diff})


def p1_moved_first_count(dataset) -> pd.DataFrame:  # feature
    """
    Counts the number of turns P2 attacked first per each game.
    The calcs are done considering the base speed, boosts and paralysis.
    Returns results in a dataframe.
    """
    pkmn_database = open_pkmn_database_csv()
    first_move_counts = []

    for game in dataset:
        p1_first = 0

        for turn in game["battle_timeline"]:
            p1_move = turn["p1_move_details"]
            p2_move = turn["p2_move_details"]

            # if they didn't switch or did nothing
            if p1_move is not None and p2_move is not None:
                p1_priority = p1_move["priority"]
                p2_priority = p2_move["priority"]

                # check who has major priority for move
                if p1_priority > p2_priority:
                    p1_first += 1
                elif p1_priority == p2_priority:
                    # if priority is equal, check speed
                    p1_name = turn["p1_pokemon_state"]["name"]
                    p2_name = turn["p2_pokemon_state"]["name"]

                    # Take base speed of the pokemons
                    p1_info = pkmn_database[pkmn_database["name"] == p1_name]
                    p2_info = pkmn_database[pkmn_database["name"] == p2_name]

                    if len(p1_info) > 0 and len(p2_info) > 0:
                        p1_base_spe = p1_info.iloc[0]["base_spe"]
                        p2_base_spe = p2_info.iloc[0]["base_spe"]

                        # Consider the speed boosts
                        p1_spe_boost = turn["p1_pokemon_state"]["boosts"]["spe"]
                        p2_spe_boost = turn["p2_pokemon_state"]["boosts"]["spe"]

                        # Applica i boost (ogni stage = 50% in più o in meno)
                        # Semplificazione: +1 = 1.5x, +2 = 2x, -1 = 0.67x, etc.
                        p1_effective_spe = (
                            p1_base_spe * (1 + 0.5 * p1_spe_boost)
                            if p1_spe_boost >= 0
                            else p1_base_spe / (1 + 0.5 * abs(p1_spe_boost))
                        )
                        p2_effective_spe = (
                            p2_base_spe * (1 + 0.5 * p2_spe_boost)
                            if p2_spe_boost >= 0
                            else p2_base_spe / (1 + 0.5 * abs(p2_spe_boost))
                        )

                        # consider paralysis for speed calculation
                        if turn["p1_pokemon_state"]["status"] == "par":
                            p1_effective_spe *= 0.25
                        if turn["p2_pokemon_state"]["status"] == "par":
                            p2_effective_spe *= 0.25

                        if p1_effective_spe > p2_effective_spe:
                            p1_first += 1

        first_move_counts.append(p1_first)  # appending

    return pd.DataFrame({"p1_moved_first_count": first_move_counts})


def p2_moved_first_count(dataset) -> pd.DataFrame:  # feature
    """
    Counts the number of turns P2 attacked first per each game.
    The calcs are done considering the base speed, boosts and paralysis.
    Returns results in a dataframe.
    """
    pkmn_database = open_pkmn_database_csv()
    first_move_counts = []

    for game in dataset:
        p2_first = 0

        for turn in game["battle_timeline"]:
            p1_move = turn["p1_move_details"]
            p2_move = turn["p2_move_details"]

            # if they didn't switch or did nothing
            if p1_move is not None and p2_move is not None:
                p1_priority = p1_move["priority"]
                p2_priority = p2_move["priority"]

                # check who has major priority for move
                if p2_priority > p1_priority:
                    p2_first += 1
                elif p1_priority == p2_priority:
                    # if priority is equal, check speed
                    p1_name = turn["p1_pokemon_state"]["name"]
                    p2_name = turn["p2_pokemon_state"]["name"]

                    # Take base speed of the pokemons
                    p1_info = pkmn_database[pkmn_database["name"] == p1_name]
                    p2_info = pkmn_database[pkmn_database["name"] == p2_name]

                    if len(p1_info) > 0 and len(p2_info) > 0:
                        p1_base_spe = p1_info.iloc[0]["base_spe"]
                        p2_base_spe = p2_info.iloc[0]["base_spe"]

                        # Consider the speed boosts
                        p1_spe_boost = turn["p1_pokemon_state"]["boosts"]["spe"]
                        p2_spe_boost = turn["p2_pokemon_state"]["boosts"]["spe"]

                        # apply the speed boost if necessary
                        p1_effective_spe = (
                            p1_base_spe * (1 + 0.5 * p1_spe_boost)
                            if p1_spe_boost >= 0
                            else p1_base_spe / (1 + 0.5 * abs(p1_spe_boost))
                        )
                        p2_effective_spe = (
                            p2_base_spe * (1 + 0.5 * p2_spe_boost)
                            if p2_spe_boost >= 0
                            else p2_base_spe / (1 + 0.5 * abs(p2_spe_boost))
                        )

                        # consider paralysis for speed calculation
                        if turn["p1_pokemon_state"]["status"] == "par":
                            p1_effective_spe *= 0.25
                        if turn["p2_pokemon_state"]["status"] == "par":
                            p2_effective_spe *= 0.25

                        if p2_effective_spe > p1_effective_spe:
                            p2_first += 1

        first_move_counts.append(p2_first)  # appending

    return pd.DataFrame({"p2_moved_first_count": first_move_counts})


def speed_advantage_ratio(dataset) -> pd.DataFrame:  # feature
    """
    Calculate the ratio between the number of turns in which P1 moves first vs. P2 per each game.
    The formula used is (p1_moved_first + 1) / (p2_moved_first + 1). We add +1 to avoid division by zero.
    Return the result in a dataframe.
    """
    p1_first = p1_moved_first_count(dataset)
    p2_first = p2_moved_first_count(dataset)

    # Calcola il rapporto con smoothing (+1) per evitare divisione per zero
    ratio = (p1_first["p1_moved_first_count"] + 1) / (
        p2_first["p2_moved_first_count"] + 1
    )

    return pd.DataFrame({"speed_advantage_ratio": ratio})
