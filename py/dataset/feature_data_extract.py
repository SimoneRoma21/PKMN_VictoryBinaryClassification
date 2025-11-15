from typing import Callable
from enum import Enum

from . import csv_utilities as csv_u
from . import extract_utilities as ext_u
from . import feature_base_stats as fbs
from . import feature_during_battle as fdb
from . import feature_moves as fm
from . import feature_status as fs


class Feature(Enum):
    """Enum with all available features"""

    # Features Trend
    # --- HP Trend ---
    HP_TREND_DIFF = "hp_trend_diff"

    # --- ATK Trend ---
    ATK_TREND_DIFF = "atk_trend_diff"

    # --- DEF Trend ---
    DEF_TREND_DIFF = "def_trend_diff"

    # --- SPA Trend ---
    SPA_TREND_DIFF = "spa_trend_diff"

    # --- SPD Trend ---
    SPD_TREND_DIFF = "spd_trend_diff"

    # --- SPE Trend ---
    SPE_TREND_DIFF = "spe_trend_diff"

    # ----Feature Base Stats Pokemon----#
    MEAN_SPE_LAST = "mean_spe_last"
    MEAN_HP_LAST = "mean_hp_last"
    P1_FINAL_TEAM_HP = "p1_final_team_hp"
    P2_FINAL_TEAM_HP = "p2_final_team_hp"
    MEAN_ATK_LAST = "mean_atk_last"
    MEAN_DEF_LAST = "mean_def_last"
    MEAN_SPA_LAST = "mean_spa_last"
    MEAN_SPD_LAST = "mean_spd_last"
    MEAN_STATS_LAST = "mean_stats_last"
    MEAN_CRIT = "mean_crit"

    # ----Feature Infos During Battle ----#
    P1_ALIVE_PKMN = "p1_alive_pkmn"
    P2_ALIVE_PKMN = "p2_alive_pkmn"
    P1_SWITCHES_COUNT = "p1_switches_count"
    P2_SWITCHES_COUNT = "p2_switches_count"
    P1_AVG_HP_WHEN_SWITCHING = "p1_avg_hp_when_switching"
    P2_AVG_HP_WHEN_SWITCHING = "p2_avg_hp_when_switching"
    P1_MAX_DEBUFF_RECEIVED = "p1_max_debuff_received"
    P2_MAX_DEBUFF_RECEIVED = "p2_max_debuff_received"
    P1_AVG_MOVE_POWER = "p1_avg_move_power"
    P2_AVG_MOVE_POWER = "p2_avg_move_power"
    AVG_MOVE_POWER_DIFFERENCE = "avg_move_power_difference"
    P1_OFFENSIVE_RATIO = "p1_offensive_ratio"
    P2_OFFENSIVE_RATIO = "p2_offensive_ratio"
    OFFENSIVE_RATIO_DIFFERENCE = "offensive_ratio_difference"
    P1_MOVED_FIRST_COUNT = "p1_moved_first_count"
    P2_MOVED_FIRST_COUNT = "p2_moved_first_count"
    SPEED_ADVANTAGE_RATIO = "speed_advantage_ratio"

    # ----Feature Status of Pokemons----#
    P1_FROZEN_PKMN = "p1_frozen_pkmn"
    P2_FROZEN_PKMN = "p2_frozen_pkmn"
    P1_PARALIZED_PKMN = "p1_paralized_pkmn"
    P2_PARALIZED_PKMN = "p2_paralized_pkmn"
    P1_SLEEP_PKMN = "p1_sleep_pkmn"
    P2_SLEEP_PKMN = "p2_sleep_pkmn"
    P1_POISON_PKMN = "p1_poison_pkmn"
    P2_POISON_PKMN = "p2_poison_pkmn"
    P1_BURNED_PKMN = "p1_burned_pkmn"
    P2_BURNED_PKMN = "p2_burned_pkmn"

    # ----Feature Pokemon Moves----#
    P1_PKMN_REFLECT = "p1_pkmn_reflect"
    P2_PKMN_REFLECT = "p2_pkmn_reflect"
    P1_PKMN_REST = "p1_pkmn_rest"
    P2_PKMN_REST = "p2_pkmn_rest"
    P1_PKMN_EXPLOSION = "p1_pkmn_explosion"
    P2_PKMN_EXPLOSION = "p2_pkmn_explosion"
    P1_PKMN_THUNDERWAVE = "p1_pkmn_thunderwave"
    P2_PKMN_THUNDERWAVE = "p2_pkmn_thunderwave"
    P1_PKMN_RECOVER = "p1_pkmn_recover"
    P2_PKMN_RECOVER = "p2_pkmn_recover"
    P1_PKMN_TOXIC = "p1_pkmn_toxic"
    P2_PKMN_TOXIC = "p2_pkmn_toxic"
    P1_PKMN_FIRESPIN = "p1_pkmn_firespin"
    P2_PKMN_FIRESPIN = "p2_pkmn_firespin"
    P1_REFLECT_RATIO = "p1_reflect_ratio"
    P2_REFLECT_RATIO = "p2_reflect_ratio"
    P1_LIGHTSCREEN_RATIO = "p1_lightscreen_ratio"
    P2_LIGHTSCREEN_RATIO = "p2_lightscreen_ratio"

    # -------------Ratios of Stats--------------------------#
    HP_BULK_RATIO = "hp_bulk_ratio"
    SPE_ATK_RATIO = "spe_atk_ratio"
    OFF_DEF_RATIO = "off_def_ratio"
    OFF_SPAD_RATIO = "off_spad_ratio"


class FeatureRegistry:
    """
    Registry that maps each feature to its extraction function.
    Automatically handles dependencies between features.
    """

    def __init__(self):
        self._extractors = {}
        self._register_all_extractors()

    def get_extractor(self, feature: Feature) -> Callable:
        """Returns the extractor for a feature"""
        return self._extractors.get(feature)

    def _register_all_extractors(self):
        """Registers all available extractors"""

        # Features Trend
        # --- Trend ---
        self._extractors[Feature.HP_TREND_DIFF] = fbs.hp_trend

        # --- ATK Trend ---
        self._extractors[Feature.ATK_TREND_DIFF] = fbs.atk_trend

        # --- DEF Trend ---
        self._extractors[Feature.DEF_TREND_DIFF] = fbs.def_trend # NOT USED

        # --- SPA Trend ---
        self._extractors[Feature.SPA_TREND_DIFF] = fbs.spa_trend

        # --- SPD Trend ---
        self._extractors[Feature.SPD_TREND_DIFF] = fbs.spd_trend # NOT USED

        # --- SPE Trend ---
        self._extractors[Feature.SPE_TREND_DIFF] = fbs.spe_trend

        # ----Feature Base Stats Pokemon----#
        self._extractors[Feature.MEAN_SPE_LAST] = fbs.mean_spe_last_2
        self._extractors[Feature.MEAN_HP_LAST] = fbs.mean_hp_last
        self._extractors[Feature.P1_FINAL_TEAM_HP] = fbs.p1_final_team_hp
        self._extractors[Feature.P2_FINAL_TEAM_HP] = fbs.p2_final_team_hp
        self._extractors[Feature.MEAN_ATK_LAST] = fbs.mean_atk_last_2
        self._extractors[Feature.MEAN_DEF_LAST] = fbs.mean_def_last_2
        self._extractors[Feature.MEAN_SPA_LAST] = fbs.mean_spa_last_2
        self._extractors[Feature.MEAN_SPD_LAST] = fbs.mean_spd_last_2
        self._extractors[Feature.MEAN_STATS_LAST] = fbs.mean_stats_last_2
        self._extractors[Feature.MEAN_CRIT] = fbs.mean_crit_2

        # ----Feature Infos During Battle ----#
        self._extractors[Feature.P1_ALIVE_PKMN] = fdb.p1_alive_pkmn
        self._extractors[Feature.P2_ALIVE_PKMN] = fdb.p2_alive_pkmn
        self._extractors[Feature.P1_SWITCHES_COUNT] = fdb.p1_switches_count
        self._extractors[Feature.P2_SWITCHES_COUNT] = fdb.p2_switches_count
        self._extractors[Feature.P1_AVG_HP_WHEN_SWITCHING] = (
            fdb.p1_avg_hp_when_switching
        )
        self._extractors[Feature.P2_AVG_HP_WHEN_SWITCHING] = (
            fdb.p2_avg_hp_when_switching
        )
        self._extractors[Feature.P1_MAX_DEBUFF_RECEIVED] = fdb.p1_max_debuff_received
        self._extractors[Feature.P2_MAX_DEBUFF_RECEIVED] = fdb.p2_max_debuff_received
        self._extractors[Feature.P1_AVG_MOVE_POWER] = fdb.p1_avg_move_power
        self._extractors[Feature.P2_AVG_MOVE_POWER] = fdb.p2_avg_move_power
        self._extractors[Feature.AVG_MOVE_POWER_DIFFERENCE] = (
            fdb.avg_move_power_difference
        )
        self._extractors[Feature.P1_OFFENSIVE_RATIO] = fdb.p1_offensive_ratio
        self._extractors[Feature.P2_OFFENSIVE_RATIO] = fdb.p2_offensive_ratio
        self._extractors[Feature.OFFENSIVE_RATIO_DIFFERENCE] = (
            fdb.offensive_ratio_difference
        )
        self._extractors[Feature.P1_MOVED_FIRST_COUNT] = fdb.p1_moved_first_count
        self._extractors[Feature.P2_MOVED_FIRST_COUNT] = fdb.p2_moved_first_count
        self._extractors[Feature.SPEED_ADVANTAGE_RATIO] = fdb.speed_advantage_ratio

        # ----Feature Status of Pokemons----#
        self._extractors[Feature.P1_FROZEN_PKMN] = fs.p1_frozen_pkmn
        self._extractors[Feature.P2_FROZEN_PKMN] = fs.p2_frozen_pkmn
        self._extractors[Feature.P1_PARALIZED_PKMN] = fs.p1_paralized_pkmn
        self._extractors[Feature.P2_PARALIZED_PKMN] = fs.p2_paralized_pkmn
        self._extractors[Feature.P1_SLEEP_PKMN] = fs.p1_sleep_pkmn
        self._extractors[Feature.P2_SLEEP_PKMN] = fs.p2_sleep_pkmn
        self._extractors[Feature.P1_POISON_PKMN] = fs.p1_poison_pkmn
        self._extractors[Feature.P2_POISON_PKMN] = fs.p2_poison_pkmn
        self._extractors[Feature.P1_BURNED_PKMN] = fs.p1_burned_pkmn
        self._extractors[Feature.P2_BURNED_PKMN] = fs.p2_burned_pkmn

        # ----Feature Pokemon Moves----#
        self._extractors[Feature.P1_PKMN_REFLECT] = fm.p1_pokemon_reflect
        self._extractors[Feature.P2_PKMN_REFLECT] = fm.p2_pokemon_reflect
        self._extractors[Feature.P1_PKMN_REST] = fm.p1_pokemon_rest
        self._extractors[Feature.P2_PKMN_REST] = fm.p2_pokemon_rest
        self._extractors[Feature.P1_PKMN_EXPLOSION] = fm.p1_pokemon_explosion
        self._extractors[Feature.P2_PKMN_EXPLOSION] = fm.p2_pokemon_explosion
        self._extractors[Feature.P1_PKMN_THUNDERWAVE] = fm.p1_pokemon_thunderwave
        self._extractors[Feature.P2_PKMN_THUNDERWAVE] = fm.p2_pokemon_thunderwave
        self._extractors[Feature.P1_PKMN_RECOVER] = fm.p1_pokemon_recover
        self._extractors[Feature.P2_PKMN_RECOVER] = fm.p2_pokemon_recover
        self._extractors[Feature.P1_PKMN_TOXIC] = fm.p1_pokemon_toxic
        self._extractors[Feature.P2_PKMN_TOXIC] = fm.p2_pokemon_toxic
        self._extractors[Feature.P1_PKMN_FIRESPIN] = fm.p1_pokemon_firespin
        self._extractors[Feature.P2_PKMN_FIRESPIN] = fm.p2_pokemon_firespin
        self._extractors[Feature.P1_REFLECT_RATIO] = fm.p1_reflect_ratio  # NOT USED
        self._extractors[Feature.P2_REFLECT_RATIO] = fm.p2_reflect_ratio # NOT USED
        self._extractors[Feature.P1_LIGHTSCREEN_RATIO] = fm.p1_lightscreen_ratio # NOT USED
        self._extractors[Feature.P2_LIGHTSCREEN_RATIO] = fm.p2_lightscreen_ratio # NOT USED

        # --------------------------------------
        self._extractors[Feature.HP_BULK_RATIO] = fbs.hp_bulk_ratio
        self._extractors[Feature.SPE_ATK_RATIO] = fbs.spe_atk_ratio
        self._extractors[Feature.OFF_DEF_RATIO] = fbs.off_def_ratio
        self._extractors[Feature.OFF_SPAD_RATIO] = fbs.off_spad_ratio


if __name__ == "__main__":
    dataset = csv_u.open_train_json()
