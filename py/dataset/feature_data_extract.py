import pandas as pd
import json
import numpy as np
from typing import List, Dict, Callable
from enum import Enum

import dataset.csv_utilities as csv_u
import dataset.extract_utilities as ext_u
import dataset.feature_base_stats as fbs
import dataset.feature_during_battle as fdb
import dataset.feature_moves as fm
import dataset.feature_status as fs
import dataset.feature_weakness as fw

class Feature(Enum):
    """Enum con tutte le feature disponibili"""


    OFFENSE_SPEED_PRODUCT = "atk_spe_prod"
    
    # Features Trend

    # --- HP Trend ---
    P1_HP_TREND = "p1_hp_trend"
    P2_HP_TREND = "p2_hp_trend"
    HP_TREND_DIFF = "hp_trend_diff"

    # --- ATK Trend ---
    P1_ATK_TREND = "p1_atk_trend"
    P2_ATK_TREND = "p2_atk_trend"
    ATK_TREND_DIFF = "atk_trend_diff"

    # --- DEF Trend ---
    P1_DEF_TREND = "p1_def_trend"
    P2_DEF_TREND = "p2_def_trend"
    DEF_TREND_DIFF = "def_trend_diff"

    # --- SPA Trend ---
    P1_SPA_TREND = "p1_spa_trend"
    P2_SPA_TREND = "p2_spa_trend"
    SPA_TREND_DIFF = "spa_trend_diff"

    # --- SPD Trend ---
    P1_SPD_TREND = "p1_spd_trend"
    P2_SPD_TREND = "p2_spd_trend"
    SPD_TREND_DIFF = "spd_trend_diff"

    # --- SPE Trend ---
    P1_SPE_TREND = "p1_spe_trend"
    P2_SPE_TREND = "p2_spe_trend"
    SPE_TREND_DIFF = "spe_trend_diff"



    #----Feature Base Stats Pokemon----#
    P1_MEAN_HP_START = "p1_mean_hp_start"
    P2_MEAN_HP_START = "p2_mean_hp_start"
    MEAN_HP_DIFFERENCE_START= "mean_hp_difference_start"
    LEAD_SPD = "lead_spd"
    MEAN_SPE_START = "mean_spe_start"
    MEAN_ATK_START = "mean_atk_start"
    MEAN_DEF_START = "mean_def_start"
    MEAN_SPA_START = "mean_spa_start"
    MEAN_SPD_START = "mean_spd_start"
    P1_MEAN_SPE_START="p1_mean_spe_start"
    P2_MEAN_SPE_START="p2_mean_spe_start"
    MEAN_SPE_DIFFERENCE_START="mean_spe_start_difference"
    MEAN_SPE_LAST = "mean_spe_last"
    P1_MEAN_SPE_LAST="p1_mean_spe_last"
    P2_MEAN_SPE_LAST="p2_mean_spe_last"
    MEAN_SPE_DIFFERENCE_LAST="mean_spe_last_difference"
    MEAN_HP_LAST = "mean_hp_last"
    P1_MEAN_HP_LAST= "p1_mean_hp_last" 
    P2_MEAN_HP_LAST= "p2_mean_hp_last"
    MEAN_HP_DIFFERENCE_LAST= "mean_hp_last_difference"
    P1_FINAL_TEAM_HP = "p1_final_team_hp"
    P2_FINAL_TEAM_HP = "p2_final_team_hp"
    FINAL_TEAM_HP_DIFFERENCE = "final_team_hp_difference"
    MEAN_ATK_LAST = "mean_atk_last"
    MEAN_DEF_LAST = "mean_def_last"
    MEAN_SPA_LAST = "mean_spa_last"
    MEAN_SPD_LAST = "mean_spd_last"
    MEAN_STATS_START= "mean_stats_start"
    MEAN_STATS_LAST= "mean_stats_last"
    MEAN_CRIT= "mean_crit"
    
    # Sum versions of mean_*_last features
    SUM_HP_LAST = "sum_hp_last"
    P1_SUM_HP_LAST = "p1_sum_hp_last"
    P2_SUM_HP_LAST = "p2_sum_hp_last"
    SUM_SPE_LAST = "sum_spe_last"
    P1_SUM_SPE_LAST = "p1_sum_spe_last"
    P2_SUM_SPE_LAST = "p2_sum_spe_last"
    SUM_ATK_LAST = "sum_atk_last"
    P1_SUM_ATK_LAST = "p1_sum_atk_last"
    P2_SUM_ATK_LAST = "p2_sum_atk_last"
    SUM_DEF_LAST = "sum_def_last"
    P1_SUM_DEF_LAST = "p1_sum_def_last"
    P2_SUM_DEF_LAST = "p2_sum_def_last"
    SUM_SPA_LAST = "sum_spa_last"
    P1_SUM_SPA_LAST = "p1_sum_spa_last"
    P2_SUM_SPA_LAST = "p2_sum_spa_last"
    SUM_SPD_LAST = "sum_spd_last"
    P1_SUM_SPD_LAST = "p1_sum_spd_last"
    P2_SUM_SPD_LAST = "p2_sum_spd_last"
    SUM_STATS_LAST = "sum_stats_last"
    P1_SUM_STATS_LAST = "p1_sum_stats_last"
    P2_SUM_STATS_LAST = "p2_sum_stats_last"

    #----Feature Infos During Battle ----#
    P1_ALIVE_PKMN = "p1_alive_pkmn"
    P2_ALIVE_PKMN = "p2_alive_pkmn"
    ALIVE_PKMN_DIFFERENCE="alive_pkmn_difference"
    P1_PKMN_STAB= "p1_pokemon_stab"
    P2_PKMN_STAB= "p2_pokemon_stab"
    P1_SWITCHES_COUNT = "p1_switches_count"
    P2_SWITCHES_COUNT = "p2_switches_count"
    SWITCHES_DIFFERENCE = "switches_difference"
    P1_STATUS_INFLICTED = "p1_status_inflicted"
    P2_STATUS_INFLICTED = "p2_status_inflicted"
    STATUS_INFLICTED_DIFFERENCE = "status_inflicted_difference"
    P1_FIRST_FAINT_TURN = "p1_first_faint_turn"  
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
   
    #----Feature Status of Pokemons----#
    P1_FROZEN_PKMN="p1_frozen_pkmn"
    P2_FROZEN_PKMN="p2_frozen_pkmn"
    P1_PARALIZED_PKMN="p1_paralized_pkmn"
    P2_PARALIZED_PKMN="p2_paralized_pkmn"
    P1_SLEEP_PKMN="p1_sleep_pkmn"
    P2_SLEEP_PKMN="p2_sleep_pkmn"
    P1_POISON_PKMN="p1_poison_pkmn"
    P2_POISON_PKMN="p2_poison_pkmn"
    P1_BURNED_PKMN="p1_burned_pkmn"
    P2_BURNED_PKMN="p2_burned_pkmn"
    
    #----Feature Pokemon Moves----#
    P1_PKMN_REFLECT="p1_pkmn_reflect"
    P2_PKMN_REFLECT="p2_pkmn_reflect"
    P1_PKMN_REST="p1_pkmn_rest"
    P2_PKMN_REST="p2_pkmn_rest"
    P1_PKMN_EXPLOSION="p1_pkmn_explosion"
    P2_PKMN_EXPLOSION="p2_pkmn_explosion"
    P1_PKMN_THUNDERWAVE="p1_pkmn_thunderwave"
    P2_PKMN_THUNDERWAVE="p2_pkmn_thunderwave"
    P1_PKMN_RECOVER="p1_pkmn_recover"
    P2_PKMN_RECOVER="p2_pkmn_recover"
    P1_PKMN_TOXIC="p1_pkmn_toxic"
    P2_PKMN_TOXIC="p2_pkmn_toxic"
    P1_PKMN_FIRESPIN="p1_pkmn_firespin"
    P2_PKMN_FIRESPIN="p2_pkmn_firespin"
    P1_REFLECT_RATIO="p1_reflect_ratio"
    P2_REFLECT_RATIO="p2_reflect_ratio"
    P1_LIGHTSCREEN_RATIO="p1_lightscreen_ratio"
    P2_LIGHTSCREEN_RATIO="p2_lightscreen_ratio"

    #----Feature Weaknesses of Teams / Team Composition----#
    WEAKNESS_TEAMS_START= "weakness_teams"
    WEAKNESS_TEAMS_LAST= "weakness_teams_last"
    ADVANTAGE_WEAK_START="advantage_weak_start"
    ADVANTAGE_WEAK_LAST="advantage_weak_last"
    P1_PSY_PKMN= "p1_psy_pkmn"    
    P2_PSY_PKMN= "p2_psy_pkmn"

class FeatureRegistry:
    """
    Registry che mappa ogni feature alla sua funzione di estrazione.
    Gestisce automaticamente le dipendenze tra feature.
    """
    
    def __init__(self):
        self._extractors = {}
        self._register_all_extractors()

    def get_extractor(self, feature: Feature) -> Callable:
        """Ritorna l'extractor per una feature"""
        return self._extractors.get(feature)
    
    def _register_all_extractors(self):
        """Registra tutti gli extractor disponibili"""

        self._extractors[Feature.OFFENSE_SPEED_PRODUCT] = fbs.atk_spe_prod
        
        # Features Trend
        # --- Trend ---
        self._extractors[Feature.P1_HP_TREND] = fbs.hp_trend
        self._extractors[Feature.P2_HP_TREND] = fbs.hp_trend
        self._extractors[Feature.HP_TREND_DIFF] = fbs.hp_trend

        # --- ATK Trend ---
        self._extractors[Feature.P1_ATK_TREND] = fbs.atk_trend
        self._extractors[Feature.P2_ATK_TREND] = fbs.atk_trend
        self._extractors[Feature.ATK_TREND_DIFF] = fbs.atk_trend

        # --- DEF Trend ---
        self._extractors[Feature.P1_DEF_TREND] = fbs.def_trend
        self._extractors[Feature.P2_DEF_TREND] = fbs.def_trend
        self._extractors[Feature.DEF_TREND_DIFF] = fbs.def_trend

        # --- SPA Trend ---
        self._extractors[Feature.P1_SPA_TREND] = fbs.spa_trend
        self._extractors[Feature.P2_SPA_TREND] = fbs.spa_trend
        self._extractors[Feature.SPA_TREND_DIFF] = fbs.spa_trend

        # --- SPD Trend ---
        self._extractors[Feature.P1_SPD_TREND] = fbs.spd_trend
        self._extractors[Feature.P2_SPD_TREND] = fbs.spd_trend
        self._extractors[Feature.SPD_TREND_DIFF] = fbs.spd_trend

        # --- SPE Trend ---
        self._extractors[Feature.P1_SPE_TREND] = fbs.spe_trend
        self._extractors[Feature.P2_SPE_TREND] = fbs.spe_trend
        self._extractors[Feature.SPE_TREND_DIFF] = fbs.spe_trend

        
        

        #----Feature Base Stats Pokemon----#
        self._extractors[Feature.P1_MEAN_HP_START] = fbs.p1_mean_hp_start
        self._extractors[Feature.P2_MEAN_HP_START] = fbs.p2_mean_hp_start
        self._extractors[Feature.MEAN_HP_DIFFERENCE_START]= fbs.mean_hp_difference_start
        self._extractors[Feature.LEAD_SPD] = fbs.lead_spd
        self._extractors[Feature.MEAN_SPE_START] = fbs.mean_spe_start
        self._extractors[Feature.MEAN_ATK_START] = fbs.mean_atk_start
        self._extractors[Feature.MEAN_DEF_START] = fbs.mean_def_start
        self._extractors[Feature.MEAN_SPA_START] = fbs.mean_spa_start
        self._extractors[Feature.MEAN_SPD_START] = fbs.mean_spd_start
        #self._extractors[Feature.P1_MEAN_SPE_START]= fbs.p1_mean_spe_start 
        #self._extractors[Feature.P2_MEAN_SPE_START]= fbs.p2_mean_spe_start 
        #self._extractors[Feature.MEAN_SPE_DIFFERENCE_START]= fbs.mean_spe_start_difference
        self._extractors[Feature.MEAN_SPE_LAST] = fbs.mean_spe_last_2
        #self._extractors[Feature.P1_MEAN_SPE_LAST]= fbs.p1_mean_spe_last
        #self._extractors[Feature.P2_MEAN_HP_LAST]= fbs.p2_mean_spe_last
        #self._extractors[Feature.MEAN_SPE_DIFFERENCE_LAST]= fbs.mean_spe_last_difference
        self._extractors[Feature.MEAN_HP_LAST] = fbs.mean_hp_last
        #self._extractors[Feature.P1_MEAN_HP_LAST]= fbs.p1_mean_hp_last 
        #self._extractors[Feature.P2_MEAN_HP_LAST]= fbs.p2_mean_hp_last
        #self._extractors[Feature.MEAN_HP_DIFFERENCE_LAST]= fbs.mean_hp_last_difference
        self._extractors[Feature.P1_FINAL_TEAM_HP] = fbs.p1_final_team_hp
        self._extractors[Feature.P2_FINAL_TEAM_HP] = fbs.p2_final_team_hp
        self._extractors[Feature.FINAL_TEAM_HP_DIFFERENCE] = fbs.final_team_hp_difference
        self._extractors[Feature.MEAN_ATK_LAST] = fbs.mean_atk_last_2
        self._extractors[Feature.MEAN_DEF_LAST] = fbs.mean_def_last_2
        self._extractors[Feature.MEAN_SPA_LAST] = fbs.mean_spa_last_2
        self._extractors[Feature.MEAN_SPD_LAST] = fbs.mean_spd_last_2
        self._extractors[Feature.MEAN_STATS_START] = fbs.mean_stats_start
        self._extractors[Feature.MEAN_STATS_LAST] = fbs.mean_stats_last_2
        self._extractors[Feature.MEAN_CRIT]= fbs.mean_crit_2
        
        # Sum versions of mean_*_last features
        self._extractors[Feature.SUM_HP_LAST] = fbs.sum_hp_last
        self._extractors[Feature.P1_SUM_HP_LAST] = fbs.p1_sum_hp_last
        self._extractors[Feature.P2_SUM_HP_LAST] = fbs.p2_sum_hp_last
        self._extractors[Feature.SUM_SPE_LAST] = fbs.sum_spe_last_2
        self._extractors[Feature.P1_SUM_SPE_LAST] = fbs.p1_sum_spe_last
        self._extractors[Feature.P2_SUM_SPE_LAST] = fbs.p2_sum_spe_last
        self._extractors[Feature.SUM_ATK_LAST] = fbs.sum_atk_last_2
        self._extractors[Feature.P1_SUM_ATK_LAST] = fbs.p1_sum_atk_last
        self._extractors[Feature.P2_SUM_ATK_LAST] = fbs.p2_sum_atk_last
        self._extractors[Feature.SUM_DEF_LAST] = fbs.sum_def_last_2
        self._extractors[Feature.P1_SUM_DEF_LAST] = fbs.p1_sum_def_last
        self._extractors[Feature.P2_SUM_DEF_LAST] = fbs.p2_sum_def_last
        self._extractors[Feature.SUM_SPA_LAST] = fbs.sum_spa_last_2
        self._extractors[Feature.P1_SUM_SPA_LAST] = fbs.p1_sum_spa_last
        self._extractors[Feature.P2_SUM_SPA_LAST] = fbs.p2_sum_spa_last
        self._extractors[Feature.SUM_SPD_LAST] = fbs.sum_spd_last_2
        self._extractors[Feature.P1_SUM_SPD_LAST] = fbs.p1_sum_spd_last
        self._extractors[Feature.P2_SUM_SPD_LAST] = fbs.p2_sum_spd_last
        self._extractors[Feature.SUM_STATS_LAST] = fbs.sum_stats_last_2
        self._extractors[Feature.P1_SUM_STATS_LAST] = fbs.p1_sum_stats_last
        self._extractors[Feature.P2_SUM_STATS_LAST] = fbs.p2_sum_stats_last

        #----Feature Infos During Battle ----#
        self._extractors[Feature.P1_ALIVE_PKMN] = fdb.p1_alive_pkmn
        self._extractors[Feature.P2_ALIVE_PKMN] = fdb.p2_alive_pkmn
        self._extractors[Feature.ALIVE_PKMN_DIFFERENCE]= fdb.alive_pkmn_difference
        self._extractors[Feature.P1_PKMN_STAB] =  fdb.p1_pokemon_stab
        self._extractors[Feature.P2_PKMN_STAB] =  fdb.p2_pokemon_stab
        self._extractors[Feature.P1_SWITCHES_COUNT] = fdb.p1_switches_count
        self._extractors[Feature.P2_SWITCHES_COUNT] = fdb.p2_switches_count
        self._extractors[Feature.SWITCHES_DIFFERENCE] = fdb.switches_difference
        self._extractors[Feature.P1_STATUS_INFLICTED] = fdb.p1_status_inflicted
        self._extractors[Feature.P2_STATUS_INFLICTED] = fdb.p2_status_inflicted
        self._extractors[Feature.STATUS_INFLICTED_DIFFERENCE] = fdb.status_inflicted_difference
        self._extractors[Feature.P1_FIRST_FAINT_TURN] = fdb.p1_first_faint_turn
        self._extractors[Feature.P1_AVG_HP_WHEN_SWITCHING] = fdb.p1_avg_hp_when_switching
        self._extractors[Feature.P2_AVG_HP_WHEN_SWITCHING] = fdb.p2_avg_hp_when_switching
        self._extractors[Feature.P1_MAX_DEBUFF_RECEIVED] = fdb.p1_max_debuff_received
        self._extractors[Feature.P2_MAX_DEBUFF_RECEIVED] = fdb.p2_max_debuff_received
        self._extractors[Feature.P1_AVG_MOVE_POWER] = fdb.p1_avg_move_power
        self._extractors[Feature.P2_AVG_MOVE_POWER] = fdb.p2_avg_move_power
        self._extractors[Feature.AVG_MOVE_POWER_DIFFERENCE] = fdb.avg_move_power_difference
        self._extractors[Feature.P1_OFFENSIVE_RATIO] = fdb.p1_offensive_ratio
        self._extractors[Feature.P2_OFFENSIVE_RATIO] = fdb.p2_offensive_ratio
        self._extractors[Feature.OFFENSIVE_RATIO_DIFFERENCE] = fdb.offensive_ratio_difference
        self._extractors[Feature.P1_MOVED_FIRST_COUNT] = fdb.p1_moved_first_count
        self._extractors[Feature.P2_MOVED_FIRST_COUNT] = fdb.p2_moved_first_count
        self._extractors[Feature.SPEED_ADVANTAGE_RATIO] = fdb.speed_advantage_ratio

         #----Feature Status of Pokemons----#
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
        
        #----Feature Pokemon Moves----#
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
        self._extractors[Feature.P1_REFLECT_RATIO] = fm.p1_reflect_ratio
        self._extractors[Feature.P2_REFLECT_RATIO] = fm.p2_reflect_ratio
        self._extractors[Feature.P1_LIGHTSCREEN_RATIO] = fm.p1_lightscreen_ratio
        self._extractors[Feature.P2_LIGHTSCREEN_RATIO] = fm.p2_lightscreen_ratio

        #----Feature Weaknesses of Teams / Team Composition----#
        self._extractors[Feature.WEAKNESS_TEAMS_START] = fw.weakness_teams
        self._extractors[Feature.WEAKNESS_TEAMS_LAST] = fw.weakness_teams_last
        self._extractors[Feature.ADVANTAGE_WEAK_START] = fw.advantage_weak_start
        self._extractors[Feature.ADVANTAGE_WEAK_LAST] = fw.advantage_weak_last
        self._extractors[Feature.P1_PSY_PKMN] = fw.p1_psy_pkmn
        self._extractors[Feature.P2_PSY_PKMN] =  fw.p2_psy_pkmn



if __name__=="__main__":
    dataset=csv_u.open_train_json()
    # pkmn_database=open_pkmn_database_csv()
    #print(dataset[0]['battle_timeline'])
    #extract_lead_velocity(dataset)
    #print(pkmn_database(dataset))
    #print(extract_all_pokemon_p2_seen(dataset))
    #pkmn_database(dataset)

    #open_pkmn_database_csv()

    #

    #print(pd.concat([extract_mean_spd_start(dataset,pkmn_database),extract_mean_spd_last(dataset,pkmn_database)],axis=1))
    
    #print(extract_mean_hp_start(dataset,pkmn_database))
    #print(extract_mean_hp_last(dataset,pkmn_database))

    #print(pd.concat([extract_mean_hp_start(dataset,pkmn_database),extract_mean_hp_last(dataset,pkmn_database)],axis=1))
    #print(pd.concat([p1_alive_pkmn(dataset),p2_alive_pkmn(dataset),p1_mean_hp_start(dataset),p2_mean_hp_start(dataset),mean_hp_last(dataset),mean_spd_start(dataset),mean_spd_last(dataset)],axis=1))
    #mhl=pd.concat([mean_hp_last(dataset),p1_alive_pkmn(dataset),p2_alive_pkmn(dataset),pd.DataFrame({"player_won":[game['player_won'] for game in dataset]})],axis=1)
    #print(mhl.iloc[4877])
    #print(p1_alive_pkmn(dataset).iloc[4877:4880],"\n",p2_alive_pkmn(dataset).iloc[4877:4880])
    #print(p1_alive_pkmn_try(dataset))
    pd.set_option('display.max_colwidth',None)
    #pd.set_option('display.max_rows',None)
    #msl=mean_spd_last(dataset)
    #print(msl.iloc[58])

    #print(p2_mean_hp_start(dataset))
    
    #print(pd.concat([p1_alive_pkmn(dataset),p2_alive_pkmn(dataset),mean_spe_start(dataset),mean_spe_last(dataset),mean_stats_start(dataset)],axis=1))
    #print(mean_stats_start(dataset))

    #print(pd.concat([p1_final_team_hp(dataset),p2_final_team_hp(dataset),final_team_hp_difference(dataset)],axis=1))
    

    #print(extract_types_from_team_p1(dataset[0]),"\n",extract_types_from_team_p2(dataset[0]))
    #print(open_type_chart_json())
    #print(weakness_teams(dataset))
    #pkmn_weak_database()

    #print(extract_all_pokemon_p1_teams(dataset)['types'][0])
    #print(extract_all_pokemon_p2(dataset)['types'])
    #print(open_pkmn_database_csv()['types'][0])

    #print(pd.concat([weakness_teams_opt(dataset),weakness_teams(dataset)],axis=1))
    #print(weakness_teams(dataset))
    
    #print(pd.concat([weakness_teams(dataset),weakness_teams_last(dataset),p1_alive_pkmn(dataset),p2_alive_pkmn(dataset)],axis=1))
    #print(weakness_teams_last(dataset))
    
    #print(pd.concat([advantage_weak_start(dataset),advantage_weak_last(dataset)],axis=1))
    #print(advantage_weak_last(dataset))
    #print(mean_hp_difference_start(dataset))
    #print(p1_alive_pkmn(dataset),p2_alive_pkmn(dataset),alive_pkmn_difference(dataset))
    #print(mean_crit(dataset))
    #print(pd.concat([p1_burned_pkmn(dataset),p2_burned_pkmn(dataset)],axis=1).iloc[9549])
    #print(p1_burned_pkmn(dataset).sum()," \n",p2_burned_pkmn(dataset).sum())
    #print(pd.concat([p1_psy_pkmn(dataset),p2_psy_pkmn(dataset)],axis=1))
    #print(pd.concat([p1_pokemon_stab(dataset),p2_pokemon_stab(dataset)],axis=1))
    #print(pd.concat([p1_reflect_ratio(dataset),p2_reflect_ratio(dataset),p1_lightscreen_ratio(dataset),p2_lightscreen_ratio(dataset)],axis=1))
    #print(pd.concat([p1_reflect_ratio(dataset).sum(),p2_reflect_ratio(dataset).sum(),p1_lightscreen_ratio(dataset).sum(),p2_lightscreen_ratio(dataset).sum()],axis=1))
    
   
    #reflect active
    #lightscreen active

    #data=pd.concat([mean_stats_last(dataset).rename(columns={'mean_stats_last_difference':'mean_def_last_difference1'}),mean_stats_last_2(dataset).rename(columns={'mean_stats_last_difference':'mean_def_last_difference2'})],axis=1)
    #print(data[data['mean_def_last_difference1']!=data['mean_def_last_difference2']])
    #print(mean_spa_last(dataset))