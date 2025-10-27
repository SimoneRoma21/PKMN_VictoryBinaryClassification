import pandas as pd
import json
import numpy as np
from typing import List, Dict, Callable
from enum import Enum

class Feature(Enum):
    """Enum con tutte le feature disponibili"""

    P1_MEAN_HP_START = "p1_mean_hp_start"
    P2_MEAN_HP_START = "p2_mean_hp_start"
    MEAN_HP_DIFFERENCE_START= "mean_hp_difference_start"
    LEAD_SPD = "lead_spd"
    MEAN_SPE_START = "mean_spe_start"
    MEAN_ATK_START = "mean_atk_start"
    MEAN_DEF_START = "mean_def_start"
    MEAN_SPA_START = "mean_spa_start"
    MEAN_SPD_START = "mean_spd_start"
    # P1_MEAN_SPE_LAST="p1_mean_spe_start"
    # P2_MEAN_SPE_LAST="p2_mean_spe_start"
    # MEAN_SPE_START_DIFFERENCE="mean_spe_start_difference"
    MEAN_SPE_LAST = "mean_spe_last"
    # P1_MEAN_SPE_LAST="p1_mean_spe_last"
    # P2_MEAN_SPE_LAST="p2_mean_spe_last"
    # MEAN_SPE_LAST_DIFFERENCE="mean_spe_last_difference"
    MEAN_HP_LAST = "mean_hp_last"
    # P1_MEAN_HP_LAST= "p1_mean_hp_last" 
    # P2_MEAN_HP_LAST= "p2_mean_hp_last"
    # MEAN_HP_DIFFERENCE_LAST= "mean_hp_last_difference"
    MEAN_ATK_LAST = "mean_atk_last"
    MEAN_DEF_LAST = "mean_def_last"
    MEAN_SPA_LAST = "mean_spa_last"
    MEAN_SPD_LAST = "mean_spd_last"
    MEAN_STATS_START= "mean_stats_start"
    MEAN_STATS_LAST= "mean_stats_last"
    MEAN_CRIT= "mean_crit"
    P1_ALIVE_PKMN = "p1_alive_pkmn"
    P2_ALIVE_PKMN = "p2_alive_pkmn"
    ALIVE_PKMN_DIFFERENCE="alive_pkmn_difference"
    WEAKNESS_TEAMS_START= "weakness_teams"
    WEAKNESS_TEAMS_LAST= "weakness_teams_last"
    ADVANTAGE_WEAK_START="advantage_weak_start"
    ADVANTAGE_WEAK_LAST="advantage_weak_last"

    P1_PSY_PKMN= "p1_psy_pkmn"    
    P2_PSY_PKMN= "p2_psy_pkmn"
    P1_PKMN_STAB= "p1_pokemon_stab"
    P2_PKMN_STAB= "p2_pokemon_stab"

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


    P1_SWITCHES_COUNT = "p1_switches_count"
    P2_SWITCHES_COUNT = "p2_switches_count"
    P1_STATUS_INFLICTED = "p1_status_inflicted"
    P2_STATUS_INFLICTED = "p2_status_inflicted"
    SWITCHES_DIFFERENCE = "switches_difference"
    STATUS_INFLICTED_DIFFERENCE = "status_inflicted_difference"
    P1_FINAL_TEAM_HP = "p1_final_team_hp"
    P2_FINAL_TEAM_HP = "p2_final_team_hp"
    FINAL_TEAM_HP_DIFFERENCE = "final_team_hp_difference"
    P1_FIRST_FAINT_TURN = "p1_first_faint_turn"  
    P1_AVG_HP_WHEN_SWITCHING = "p1_avg_hp_when_switching"
    P2_AVG_HP_WHEN_SWITCHING = "p2_avg_hp_when_switching"
    P1_MAX_DEBUFF_RECEIVED = "p1_max_debuff_received" #non sicuro se necessaria
    P2_MAX_DEBUFF_RECEIVED = "p2_max_debuff_received" #non sicuro se necessaria
    P1_AVG_MOVE_POWER = "p1_avg_move_power"
    P2_AVG_MOVE_POWER = "p2_avg_move_power"
    AVG_MOVE_POWER_DIFFERENCE = "avg_move_power_difference"
    P1_OFFENSIVE_RATIO = "p1_offensive_ratio"
    P2_OFFENSIVE_RATIO = "p2_offensive_ratio"
    OFFENSIVE_RATIO_DIFFERENCE = "offensive_ratio_difference"
    P1_MOVED_FIRST_COUNT = "p1_moved_first_count"
    P2_MOVED_FIRST_COUNT = "p2_moved_first_count"
    SPEED_ADVANTAGE_RATIO = "speed_advantage_ratio"


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
        
        self._extractors[Feature.P1_MEAN_HP_START] = p1_mean_hp_start
        self._extractors[Feature.P2_MEAN_HP_START] = p2_mean_hp_start
        self._extractors[Feature.MEAN_HP_DIFFERENCE_START]= mean_hp_difference_start
        self._extractors[Feature.LEAD_SPD] = lead_spd
        self._extractors[Feature.MEAN_SPE_START] = mean_spe_start
        self._extractors[Feature.MEAN_ATK_START] = mean_atk_start
        self._extractors[Feature.MEAN_DEF_START] = mean_def_start
        self._extractors[Feature.MEAN_SPA_START] = mean_spa_start
        self._extractors[Feature.MEAN_SPD_START] = mean_spd_start
        #self._extractors[Feature.P1_MEAN_SPE_START]= p1_mean_spe_start 
        #self._extractors[Feature.P2_MEAN_HP_START]= p2_mean_spe_start 
        #self._extractors[Feature.MEAN_SPE_DIFFERENCE_START]= mean_spe_start_difference
        self._extractors[Feature.MEAN_SPE_LAST] = mean_spe_last
        #self._extractors[Feature.P1_MEAN_SPE_LAST]= p1_mean_spe_last
        #self._extractors[Feature.P2_MEAN_HP_LAST]= p2_mean_spe_last
        #self._extractors[Feature.MEAN_SPE_DIFFERENCE_LAST]= mean_spe_last_difference
        self._extractors[Feature.MEAN_HP_LAST] = mean_hp_last
        #self._extractors[Feature.P1_MEAN_HP_LAST]= p1_mean_hp_last 
        #self._extractors[Feature.P2_MEAN_HP_LAST]= p2_mean_hp_last
        #self._extractors[Feature.MEAN_HP_DIFFERENCE_LAST]= mean_hp_last_difference
        self._extractors[Feature.MEAN_ATK_LAST] = mean_atk_last
        self._extractors[Feature.MEAN_DEF_LAST] = mean_def_last
        self._extractors[Feature.MEAN_SPA_LAST] = mean_spa_last
        self._extractors[Feature.MEAN_SPD_LAST] = mean_spd_last
        self._extractors[Feature.MEAN_STATS_START] = mean_stats_start
        self._extractors[Feature.MEAN_STATS_LAST] = mean_stats_last
        self._extractors[Feature.MEAN_CRIT]= mean_crit
        self._extractors[Feature.P1_ALIVE_PKMN] = p1_alive_pkmn
        self._extractors[Feature.P2_ALIVE_PKMN] = p2_alive_pkmn
        self._extractors[Feature.ALIVE_PKMN_DIFFERENCE]=alive_pkmn_difference
        self._extractors[Feature.WEAKNESS_TEAMS_START] = weakness_teams
        self._extractors[Feature.WEAKNESS_TEAMS_LAST] = weakness_teams_last
        self._extractors[Feature.ADVANTAGE_WEAK_START] = advantage_weak_start
        self._extractors[Feature.ADVANTAGE_WEAK_LAST] = advantage_weak_last

        self._extractors[Feature.P1_PSY_PKMN] = p1_psy_pkmn
        self._extractors[Feature.P2_PSY_PKMN] =  p2_psy_pkmn
        self._extractors[Feature.P1_PKMN_STAB] =  p1_pokemon_stab
        self._extractors[Feature.P2_PKMN_STAB] =  p2_pokemon_stab

        self._extractors[Feature.P1_FROZEN_PKMN] = p1_frozen_pkmn
        self._extractors[Feature.P2_FROZEN_PKMN] = p2_frozen_pkmn
        self._extractors[Feature.P1_PARALIZED_PKMN] = p1_paralized_pkmn
        self._extractors[Feature.P2_PARALIZED_PKMN] = p2_paralized_pkmn
        self._extractors[Feature.P1_SLEEP_PKMN] = p1_sleep_pkmn
        self._extractors[Feature.P2_SLEEP_PKMN] = p2_sleep_pkmn
        self._extractors[Feature.P1_POISON_PKMN] = p1_poison_pkmn
        self._extractors[Feature.P2_POISON_PKMN] = p2_poison_pkmn
        self._extractors[Feature.P1_BURNED_PKMN] = p1_burned_pkmn
        self._extractors[Feature.P2_BURNED_PKMN] = p2_burned_pkmn

        self._extractors[Feature.P1_PKMN_REFLECT] = p1_pokemon_reflect
        self._extractors[Feature.P2_PKMN_REFLECT] = p2_pokemon_reflect
        self._extractors[Feature.P1_PKMN_REST] = p1_pokemon_rest
        self._extractors[Feature.P2_PKMN_REST] = p2_pokemon_rest
        self._extractors[Feature.P1_PKMN_EXPLOSION] = p1_pokemon_explosion
        self._extractors[Feature.P2_PKMN_EXPLOSION] = p2_pokemon_explosion
        self._extractors[Feature.P1_PKMN_THUNDERWAVE] = p1_pokemon_thunderwave
        self._extractors[Feature.P2_PKMN_THUNDERWAVE] = p2_pokemon_thunderwave
        self._extractors[Feature.P1_PKMN_RECOVER] = p1_pokemon_recover
        self._extractors[Feature.P2_PKMN_RECOVER] = p2_pokemon_recover
        self._extractors[Feature.P1_PKMN_TOXIC] = p1_pokemon_toxic
        self._extractors[Feature.P2_PKMN_TOXIC] = p2_pokemon_toxic
        self._extractors[Feature.P1_PKMN_FIRESPIN] = p1_pokemon_firespin
        self._extractors[Feature.P2_PKMN_FIRESPIN] = p2_pokemon_firespin
        self._extractors[Feature.P1_REFLECT_RATIO] = p1_reflect_ratio
        self._extractors[Feature.P2_REFLECT_RATIO] = p2_reflect_ratio
        self._extractors[Feature.P1_LIGHTSCREEN_RATIO] = p1_lightscreen_ratio
        self._extractors[Feature.P2_LIGHTSCREEN_RATIO] = p2_lightscreen_ratio

        

        self._extractors[Feature.P1_SWITCHES_COUNT] = p1_switches_count
        self._extractors[Feature.P2_SWITCHES_COUNT] = p2_switches_count
        self._extractors[Feature.P1_STATUS_INFLICTED] = p1_status_inflicted
        self._extractors[Feature.P2_STATUS_INFLICTED] = p2_status_inflicted
        self._extractors[Feature.SWITCHES_DIFFERENCE] = switches_difference
        self._extractors[Feature.STATUS_INFLICTED_DIFFERENCE] = status_inflicted_difference
        self._extractors[Feature.P1_FINAL_TEAM_HP] = p1_final_team_hp
        self._extractors[Feature.P2_FINAL_TEAM_HP] = p2_final_team_hp
        self._extractors[Feature.FINAL_TEAM_HP_DIFFERENCE] = final_team_hp_difference
        self._extractors[Feature.P1_FIRST_FAINT_TURN] = p1_first_faint_turn
        self._extractors[Feature.P1_AVG_HP_WHEN_SWITCHING] = p1_avg_hp_when_switching
        self._extractors[Feature.P2_AVG_HP_WHEN_SWITCHING] = p2_avg_hp_when_switching
        self._extractors[Feature.P1_MAX_DEBUFF_RECEIVED] = p1_max_debuff_received
        self._extractors[Feature.P2_MAX_DEBUFF_RECEIVED] = p2_max_debuff_received
        self._extractors[Feature.P1_AVG_MOVE_POWER] = p1_avg_move_power
        self._extractors[Feature.P2_AVG_MOVE_POWER] = p2_avg_move_power
        self._extractors[Feature.AVG_MOVE_POWER_DIFFERENCE] = avg_move_power_difference
        self._extractors[Feature.P1_OFFENSIVE_RATIO] = p1_offensive_ratio
        self._extractors[Feature.P2_OFFENSIVE_RATIO] = p2_offensive_ratio
        self._extractors[Feature.OFFENSIVE_RATIO_DIFFERENCE] = offensive_ratio_difference
        self._extractors[Feature.P1_MOVED_FIRST_COUNT] = p1_moved_first_count
        self._extractors[Feature.P2_MOVED_FIRST_COUNT] = p2_moved_first_count
        self._extractors[Feature.SPEED_ADVANTAGE_RATIO] = speed_advantage_ratio


def open_pkmn_database_csv() -> pd.DataFrame:
    #opening pkmn database csv
    pkmn_db=pd.read_csv("../data/pkmn_database.csv")
    pkmn_db=pkmn_db.drop("Unnamed: 0",axis=1)
    return pkmn_db

def open_pkmn_database_weak_csv() -> pd.DataFrame:
    #opening pkmn database weakness csv
    pkmn_db_weak=pd.read_csv("../data/pkmn_database_weaknesses.csv")
    return pkmn_db_weak

def open_train_json() -> list:
    list = []
    with open("../data/train.jsonl", "r") as f:
        for line in f:
            list.append(json.loads(line))
    list.remove(list[4877])
    return list


def open_type_chart_json() -> pd.DataFrame:
    with open("../data/type_chart.json", "r") as f:
        data=json.load(f)
    return pd.DataFrame(data).transpose()

def extract_all_pokemon_p1_teams(dataset) -> pd.DataFrame:
    #extracting all p1 teams
    db_pkmn_p1= pd.DataFrame([team for game in dataset for team in game['p1_team_details']]) 
    db_pkmn_p1.drop_duplicates(subset=['name'],inplace=True)

    db_types_p1=pd.concat([extract_types_from_team_p1(game) for game in dataset]).drop_duplicates(subset="name",keep='first')
    db_pkmn_p1=db_pkmn_p1.merge(db_types_p1, how='inner',on='name')
    return db_pkmn_p1

def extract_all_pokemon_p2_seen(dataset) -> pd.DataFrame:
    #extracting all p2 seens pokemons
    db_pkmn_p2_battles=pd.DataFrame([elem['p2_pokemon_state']['name'] for game in dataset for elem in game['battle_timeline']])
    db_pkmn_p2_battles.drop_duplicates(inplace=True)
    db_pkmn_p2_battles.rename(columns={0:'name'},inplace=True)
    return db_pkmn_p2_battles

def extract_all_pokemon_p2_lead(dataset,duplicates) -> pd.DataFrame:
    # getting all p2 leads
    db_pkmn_p2_lead=pd.DataFrame([game['p2_lead_details'] for game in dataset ])
    if not(duplicates): # admitting duplicates or not
        db_pkmn_p2_lead.drop_duplicates(subset=['name'],inplace=True)
    return db_pkmn_p2_lead

def extract_all_pokemon_p2(dataset) -> pd.DataFrame:
 
    # picking all pokemons seen in all battles of p2 
    db_pkmn_p2_battles=extract_all_pokemon_p2_seen(dataset)
    # picking all pokemon leads of p2 (has to be subset of db_pkmn_p2_battles)
    db_pkmn_p2_lead=extract_all_pokemon_p2_lead(dataset,False)
    # merging the two dataset
    db_pkmn_p2=db_pkmn_p2_lead.merge(db_pkmn_p2_battles, how='inner', on=['name'])

    db_types_p2=pd.concat([extract_types_from_team_p2(game) for game in dataset]).drop_duplicates(subset="name",keep='first')
    db_pkmn_p2=db_pkmn_p2.merge(db_types_p2, how='inner',on='name')

    return db_pkmn_p2

def pkmn_database(dataset):

    # picking all pokemons seen of p1 in all games
    db_pkmn_p1=extract_all_pokemon_p1_teams(dataset)
        
    #picking all pokemon seen of p2 in all games
    db_pkmn_p2=extract_all_pokemon_p2(dataset)

    # union of dataframes and then dropping duplicates
    db_pkmn=pd.concat([db_pkmn_p1,db_pkmn_p2])
    db_pkmn.drop_duplicates(subset=['name'],inplace=True)
    
    #saving to csv
    pd.DataFrame.to_csv(db_pkmn,"../data/pkmn_database.csv")
    

def pkmn_weak_database():
    db_pkmn=open_pkmn_database_csv()
    weaknesses=[]
    for index,row in db_pkmn.iterrows():
        weak=calc_weakness(row["type_1"],row["type_2"]).reset_index().rename(columns={'index':'type'})['type'].to_list()
        weaknesses.append(weak)

    db_weak=pd.DataFrame({"weaknesses":weaknesses})
    db_pkmn_weak=pd.concat([db_pkmn,db_weak],axis=1)
    pd.DataFrame.to_csv(db_pkmn_weak,"../data/pkmn_database_weaknesses.csv")

def moves_database():
    pass


def lead_spd(dataset) -> pd.DataFrame: # feature

    #extract all pokemon leads for p1
    p1_lead=pd.DataFrame([game['p1_team_details'][0] for game in dataset])
    p1_lead=p1_lead[['name','base_spd']]

    #extracts all pokemon leads for p2
    p2_lead=extract_all_pokemon_p2_lead(dataset,True)
    p2_lead=p2_lead[['name','base_spd']]

    strt_player=pd.DataFrame([],columns=['spd_adv'])
    strt_player['spd_adv']=(p1_lead['base_spd']<p2_lead['base_spd']).astype(int)

    #check=pd.concat([p1_lead,p2_lead,str_player],axis=1)
   
    return strt_player

    
def extract_p1_team_from_game_start(game)-> pd.Series:
    return pd.DataFrame(game['p1_team_details'])['name']


def extract_p1_team_from_game_last(game) -> pd.Series:
    turns=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])
    pkmn_dead_p1=turns[turns['status']=='fnt']['name'].drop_duplicates(keep='last')
    
    team_start_p1=extract_p1_team_from_game_start(game)
    team_remain_p1=team_start_p1[~team_start_p1.isin(pkmn_dead_p1)]
    
    return team_remain_p1

def extract_p1_team_from_game_start_with_stats(game)-> pd.DataFrame:
    turns=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])
    pkmn_p1_start=turns.drop_duplicates(subset='name',keep='last')

    return pkmn_p1_start

def extract_p1_team_from_game_last_with_stats(game) -> pd.Series:
    turns=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])
    pkmn_dead_p1=turns[turns['status']=='fnt']['name'].drop_duplicates(keep='last')
    
    team_start_p1=extract_p1_team_from_game_start_with_stats(game)
    team_remain_p1=team_start_p1[~team_start_p1['name'].isin(pkmn_dead_p1)]
    
    return team_remain_p1

def extract_p2_team_from_game_start(game) -> pd.Series:
    turns=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
    pkmn_p2_start=turns.drop_duplicates(subset='name',keep='last')['name']
    
    return pkmn_p2_start

def extract_p2_team_from_game_last(game) -> pd.Series:

    turns=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
    pkmn_p2_fainted=turns[turns['status']=='fnt']['name'].drop_duplicates(keep='last')

    pkmn_p2_start=extract_p2_team_from_game_start(game)
    pkmn_p2_last=pkmn_p2_start[~pkmn_p2_start.isin(pkmn_p2_fainted)]
    
    return pkmn_p2_last 

def extract_p2_team_from_game_start_with_stats(game)-> pd.DataFrame:
    turns=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
    pkmn_p2_start=turns.drop_duplicates(subset='name',keep='last')

    return pkmn_p2_start

def extract_p2_team_from_game_last_with_stats(game) -> pd.Series:
    turns=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
    pkmn_dead_p2=turns[turns['status']=='fnt']['name'].drop_duplicates(keep='last')
    
    team_start_p2=extract_p2_team_from_game_start_with_stats(game)
    team_remain_p2=team_start_p2[~team_start_p2['name'].isin(pkmn_dead_p2)]
    
    return team_remain_p2

def mean_spe_start(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()
    mean_spe=mean_spe_database(pkmn_database)
    p1_mean_spe=[]
    p2_mean_spe=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_start(game)
        p2_team=extract_p2_team_from_game_start(game)

        p1_team=pkmn_database[pkmn_database['name'].isin(p1_team)]
        p1_known=len(p1_team)
        p1_team=p1_team[['name','base_spe']]
        p1_mean_spe.append((np.sum(p1_team['base_spe'])+ (mean_spe*(6-p1_known)))/6)
    
        p2_team=pkmn_database[pkmn_database['name'].isin(p2_team)]
        p2_known=len(p2_team)
        p2_team=p2_team[['name','base_spe']]
        p2_mean_spe.append((np.sum(p2_team['base_spe'])+ mean_spe*(6-p2_known))/6)

    mean_spe_start=pd.DataFrame({'p1_mean_spe_start':p1_mean_spe,'p2_mean_spe_start':p2_mean_spe})
    mean_spe_start['mean_spe_start_difference']=np.subtract.reduce(mean_spe_start[['p1_mean_spe_start','p2_mean_spe_start']],axis=1)
    #return mean_spe_start['p2_mean_spe_start']
    return mean_spe_start

def p1_mean_start_last(dataset)-> pd.DataFrame: #feature
    return mean_spe_start(dataset)['p1_mean_spe_start']

def p2_mean_start_last(dataset)-> pd.DataFrame: #feature
    return mean_spe_start(dataset)['p2_mean_spe_start']

def mean_spe_start_difference(dataset)-> pd.DataFrame: #feature
    return mean_spe_start(dataset)['mean_spe_start_difference']

# --- Feature ATK DEF SPA SPD START ---
# ATK
def mean_atk_start(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()
    mean_atk=mean_atk_database(pkmn_database)
    p1_mean_atk=[]
    p2_mean_atk=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_start(game)
        p2_team=extract_p2_team_from_game_start(game)

        p1_team=pkmn_database[pkmn_database['name'].isin(p1_team)]
        p1_known=len(p1_team)
        p1_team=p1_team[['name','base_atk']]
        p1_mean_atk.append((np.sum(p1_team['base_atk'])+ (mean_atk*(6-p1_known)))/6)
    
        p2_team=pkmn_database[pkmn_database['name'].isin(p2_team)]
        p2_known=len(p2_team)
        p2_team=p2_team[['name','base_atk']]
        p2_mean_atk.append((np.sum(p2_team['base_atk'])+ mean_atk*(6-p2_known))/6)

    mean_atk_start=pd.DataFrame({'p1_mean_atk_start':p1_mean_atk,'p2_mean_atk_start':p2_mean_atk})
    mean_atk_start['mean_atk_start_difference']=np.subtract.reduce(mean_atk_start[['p1_mean_atk_start','p2_mean_atk_start']],axis=1)
    return mean_atk_start

def mean_atk_start_difference(dataset)-> pd.DataFrame: #feature
    return mean_atk_start(dataset)['mean_atk_start_difference']

# DEF
def mean_def_start(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()
    mean_def=mean_def_database(pkmn_database)
    p1_mean_def=[]
    p2_mean_def=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_start(game)
        p2_team=extract_p2_team_from_game_start(game)

        p1_team=pkmn_database[pkmn_database['name'].isin(p1_team)]
        p1_known=len(p1_team)
        p1_team=p1_team[['name','base_def']]
        p1_mean_def.append((np.sum(p1_team['base_def'])+ (mean_def*(6-p1_known)))/6)
    
        p2_team=pkmn_database[pkmn_database['name'].isin(p2_team)]
        p2_known=len(p2_team)
        p2_team=p2_team[['name','base_def']]
        p2_mean_def.append((np.sum(p2_team['base_def'])+ mean_def*(6-p2_known))/6)

    mean_def_start=pd.DataFrame({'p1_mean_def_start':p1_mean_def,'p2_mean_def_start':p2_mean_def})
    mean_def_start['mean_def_start_difference']=np.subtract.reduce(mean_def_start[['p1_mean_def_start','p2_mean_def_start']],axis=1)
    return mean_def_start

def mean_def_start_difference(dataset)-> pd.DataFrame: #feature
    return mean_def_start(dataset)['mean_def_start_difference']

# SPA
def mean_spa_start(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()
    mean_spa=mean_spa_database(pkmn_database)
    p1_mean_spa=[]
    p2_mean_spa=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_start(game)
        p2_team=extract_p2_team_from_game_start(game)

        p1_team=pkmn_database[pkmn_database['name'].isin(p1_team)]
        p1_known=len(p1_team)
        p1_team=p1_team[['name','base_spa']]
        p1_mean_spa.append((np.sum(p1_team['base_spa'])+ (mean_spa*(6-p1_known)))/6)
    
        p2_team=pkmn_database[pkmn_database['name'].isin(p2_team)]
        p2_known=len(p2_team)
        p2_team=p2_team[['name','base_spa']]
        p2_mean_spa.append((np.sum(p2_team['base_spa'])+ mean_spa*(6-p2_known))/6)

    mean_spa_start=pd.DataFrame({'p1_mean_spa_start':p1_mean_spa,'p2_mean_spa_start':p2_mean_spa})
    mean_spa_start['mean_spa_start_difference']=np.subtract.reduce(mean_spa_start[['p1_mean_spa_start','p2_mean_spa_start']],axis=1)
    return mean_spa_start

def mean_spa_start_difference(dataset)-> pd.DataFrame: #feature
    return mean_spa_start(dataset)['mean_spa_start_difference']

# SPD
def mean_spd_start(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()
    mean_spd=mean_spd_database(pkmn_database)
    p1_mean_spd=[]
    p2_mean_spd=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_start(game)
        p2_team=extract_p2_team_from_game_start(game)

        p1_team=pkmn_database[pkmn_database['name'].isin(p1_team)]
        p1_known=len(p1_team)
        p1_team=p1_team[['name','base_spd']]
        p1_mean_spd.append((np.sum(p1_team['base_spd'])+ mean_spd*(6-p1_known))/6)
    
        p2_team=pkmn_database[pkmn_database['name'].isin(p2_team)]
        p2_known=len(p2_team)
        p2_team=p2_team[['name','base_spd']]
        p2_mean_spd.append((np.sum(p2_team['base_spd'])+ mean_spd*(6-p2_known))/6)

    mean_spd_start=pd.DataFrame({'p1_mean_spd_start':p1_mean_spd,'p2_mean_spd_start':p2_mean_spd})
    mean_spd_start['mean_spd_start_difference']=np.subtract.reduce(mean_spd_start[['p1_mean_spd_start','p2_mean_spd_start']],axis=1)
    return mean_spd_start
def mean_spd_start_difference(dataset)-> pd.DataFrame: #feature
    return mean_spd_start(dataset)['mean_spd_start_difference']

#--- --------------

def mean_spe_last(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()
    mean_spe=mean_spe_database(pkmn_database)
    p1_mean_spe=[]
    p2_mean_spe=[]

    multipliers={-6: 2/8, -5: 2/7, -4: 2/6, -3: 2/5, -2:2/4, -1: 2/3, 0:1, +1: 3/2, 2: 4/2, 3: 5/2, 4: 6/2, 5: 7/2, 6: 8/2 }
    
    for game in dataset:
        p1_team=extract_p1_team_from_game_last_with_stats(game)
        p2_team=extract_p2_team_from_game_last_with_stats(game)

        p1_team=p1_team.merge(pkmn_database, how='inner', on='name')
        p1_known=len(extract_p1_team_from_game_start(game))
        p1_team=p1_team[['name','base_spe','boosts','status']]
        if(len(p1_team)!=0):
            p1_team['total']=p1_team['base_spe']*multipliers[p1_team['boosts'][0]['spe']]*[1 if elem!='par' else 0.25 for elem in p1_team['status']]
            val=np.sum(p1_team['total'])
            p1_mean_spe.append((val)+ (mean_spe*(6-p1_known))/6)
             #p1_mean_spd.append(np.mean(p1_team['base_spe']))
        else:
            p1_mean_spe.append(0)
       
    
        p2_team=p2_team.merge(pkmn_database, how='inner', on='name')
        p2_known=len(extract_p2_team_from_game_start(game))
        p2_team=p2_team[['name','base_spe','boosts','status']]
        if(len(p2_team)!=0):
            p2_team['total']=p2_team['base_spe']*multipliers[p2_team['boosts'][0]['spe']]*[1 if elem!='par' else 0.25 for elem in p2_team['status']]
            val=np.sum(p2_team['total'])
            p2_mean_spe.append(val+(mean_spe*(6-p2_known))/6)
            #p2_mean_spd.append(np.mean(p2_team['base_spe']))
        else:
            p2_mean_spe.append(0)

    mean_spe_last=pd.DataFrame({'p1_mean_spe_last':p1_mean_spe,'p2_mean_spe_last':p2_mean_spe})
    mean_spe_last['mean_spe_last_difference']=np.subtract.reduce(mean_spe_last[['p1_mean_spe_last','p2_mean_spe_last']],axis=1)
    mean_spe_last=mean_spe_last.fillna(value=0)
    return mean_spe_last 

def p1_mean_spe_last(dataset)-> pd.DataFrame: #feature
    return mean_spe_last(dataset)['p1_mean_spe_last']

def p2_mean_spe_last(dataset)-> pd.DataFrame: #feature
    return mean_spe_last(dataset)['p2_mean_spe_last']

def mean_spe_last_difference(dataset)-> pd.DataFrame: #feature
    return mean_spe_last(dataset)['mean_spe_last_difference']

def mean_crit(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()
    mean_crit=mean_crit_database(pkmn_database)
    p1_mean_crit=[]
    p2_mean_crit=[]

    for game in dataset:
        p1_team=extract_p1_team_from_game_last_with_stats(game)
        p2_team=extract_p2_team_from_game_last_with_stats(game)

        p1_team=p1_team.merge(pkmn_database, how='inner', on='name')
        p1_known=len(extract_p1_team_from_game_start(game))
        p1_team=p1_team[['name','base_spe','boosts','status']]
        if(len(p1_team)!=0):
            p1_team['total']=p1_team['base_spe']/512
            val=np.sum(p1_team['total'])
            p1_mean_crit.append((val)+( mean_crit*(6-p1_known))/6)
            #p1_mean_spd.append(np.mean(p1_team['base_spe']))
        else:
            p1_mean_crit.append(0)
       
    
        p2_team=p2_team.merge(pkmn_database, how='inner', on='name')
        p2_known=len(extract_p2_team_from_game_start(game))
        p2_team=p2_team[['name','base_spe','boosts','status']]
        if(len(p2_team)!=0):
            p2_team['total']=p2_team['base_spe']/512
            val=np.sum(p2_team['total'])
            p2_mean_crit.append((val)+(mean_crit*(6-p2_known))/6)
            #p2_mean_spd.append(np.mean(p2_team['base_spe']))
        else:
            p2_mean_crit.append(0)

    mean_crit=pd.DataFrame({'p1_mean_crit':p1_mean_crit,'p2_mean_crit':p2_mean_crit})
    mean_crit=mean_crit.fillna(value=0)
    return mean_crit

def mean_hp_database(pkmn_database) -> float:
    return np.mean(pkmn_database['base_hp'])

def mean_atk_database(pkmn_database) -> float:
    return np.mean(pkmn_database['base_atk'])

def mean_def_database(pkmn_database) -> float:
    return np.mean(pkmn_database['base_def'])

def mean_spa_database(pkmn_database) -> float:
    return np.mean(pkmn_database['base_spa'])

def mean_spd_database(pkmn_database) -> float:
    return np.mean(pkmn_database['base_spd'])

def mean_spe_database(pkmn_database) -> float:
    return np.mean(pkmn_database['base_spe'])

def mean_crit_database(pkmn_database) -> float:
    return np.mean(pkmn_database['base_spe']/512)

def p1_mean_hp_start(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()
    mean_hp=mean_hp_database(pkmn_database)
    p1_mean_hp=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_start(game)
        p1_known=len(p1_team)    
        p1_team=pkmn_database[pkmn_database['name'].isin(p1_team)]
        p1_team=p1_team[['name','base_hp']]
        p1_mean_hp.append((np.sum(p1_team['base_hp'])+ mean_hp*(6-p1_known))/6)

    mean_hp_start=pd.DataFrame({'p1_mean_hp_start':p1_mean_hp})
    return mean_hp_start

def p2_mean_hp_start(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()
    mean_hp=mean_hp_database(pkmn_database)
    p2_mean_hp=[]
    for game in dataset:
        p2_team=extract_p2_team_from_game_start(game)
        p2_team=pkmn_database[pkmn_database['name'].isin(p2_team)]
        p2_team=p2_team[['name','base_hp']]
        p2_mean_hp.append((np.sum(p2_team['base_hp'])+ mean_hp*(6-len(p2_team)))/6)
    mean_hp_start=pd.DataFrame({'p2_mean_hp_start':p2_mean_hp})
    return mean_hp_start

def mean_hp_difference_start(dataset)-> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()

    mean_hp_p1=p1_mean_hp_start(dataset)
    mean_hp_p2=p2_mean_hp_start(dataset)

    mean_hp_difference=pd.concat([mean_hp_p1,mean_hp_p2],axis=1)
    mean_hp_difference['mean_hp_difference']=np.subtract.reduce(mean_hp_difference[['p1_mean_hp_start','p2_mean_hp_start']],axis=1)

    #return mean_hp_difference['mean_hp_difference']
    #return mean_hp_difference['p2_mean_hp_start']
    return mean_hp_difference['mean_hp_difference']

def mean_hp_last(dataset): #feature
    pkmn_database = open_pkmn_database_csv()
    mean_hp=mean_hp_database(pkmn_database)
    p1_mean_hp=[]
    p2_mean_hp=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_last(game)
        p2_team=extract_p2_team_from_game_last(game)


        p1_team=pkmn_database[pkmn_database['name'].isin(p1_team)]
        p1_known=len(extract_p1_team_from_game_start(game))
        p1_team=p1_team[['name','base_hp']]
        p1_mean_hp.append((np.sum(p1_team['base_hp'])+ mean_hp*(6-p1_known))/6)
    
        p2_team=pkmn_database[pkmn_database['name'].isin(p2_team)]
        p2_team=p2_team[['name','base_hp']]
        p2_known=len(extract_p2_team_from_game_start(game))
        p2_mean_hp.append((np.sum(p2_team['base_hp'])+ mean_hp*(6-p2_known))/6)

    mean_hp_last=pd.DataFrame({'p1_mean_hp_last':p1_mean_hp,'p2_mean_hp_last':p2_mean_hp})
    mean_hp_last['mean_hp_last_difference']=np.subtract.reduce(mean_hp_last[['p1_mean_hp_last','p2_mean_hp_last']],axis=1)
    mean_hp_last=mean_hp_last.fillna(value=0)
    #return mean_hp_last['mean_hp_last_difference']
    return mean_hp_last

def p1_mean_hp_last(dataset)->pd.DataFrame: #feature
    return mean_hp_last(dataset)['p1_mean_hp_last']

def p2_mean_hp_last(dataset)->pd.DataFrame: #feature
    return mean_hp_last(dataset)['p2_mean_hp_last']

def mean_hp_last_difference(dataset)->pd.DataFrame: #feature
    return mean_hp_last(dataset)['mean_hp_last_difference']

# ---- Features ATK ------#
def mean_atk_last(dataset): #feature
    pkmn_database = open_pkmn_database_csv()
    mean_atk=mean_atk_database(pkmn_database)
    p1_mean_atk=[]
    p2_mean_atk=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_last_with_stats(game)
        p2_team=extract_p2_team_from_game_last_with_stats(game)

        p1_team=p1_team.merge(pkmn_database, how='inner', on='name')
        p1_known=len(extract_p1_team_from_game_start(game))
        p1_team=p1_team[['name','base_atk','boosts','status']]
        p1_mean_atk.append((np.sum(p1_team['base_atk']*[1 if elem!='brn' else 0.5 for elem in p1_team['status']])+ mean_atk*(6-p1_known))/6)
        #p1_mean_atk.append((np.sum(p1_team['base_atk'])+ mean_atk_database(pkmn_database)*(6-p1_known))/6)
        
        p2_team=p2_team.merge(pkmn_database, how='inner', on='name')
        p2_team=p2_team[['name','base_atk','boosts','status']]
        p2_known=len(extract_p2_team_from_game_start(game))
        p2_mean_atk.append((np.sum(p2_team['base_atk']*[1 if elem!='brn' else 0.5 for elem in p2_team['status']])+ mean_atk*(6-p2_known))/6)
        #p2_mean_atk.append((np.sum(p2_team['base_atk'])+ mean_atk_database(pkmn_database)*(6-p2_known))/6)

    mean_atk_last=pd.DataFrame({'p1_mean_atk_last':p1_mean_atk,'p2_mean_atk_last':p2_mean_atk})
    mean_atk_last['mean_atk_last_difference']=np.subtract.reduce(mean_atk_last[['p1_mean_atk_last','p2_mean_atk_last']],axis=1)
    mean_atk_last=mean_atk_last.fillna(value=0)
    return mean_atk_last

def p1_mean_atk_last(dataset)->pd.DataFrame: #feature
    return mean_atk_last(dataset)['p1_mean_atk_last']

def p2_mean_atk_last(dataset)->pd.DataFrame: #feature
    return mean_atk_last(dataset)['p2_mean_atk_last']

def mean_atk_last_difference(dataset)->pd.DataFrame: #feature
    return mean_atk_last(dataset)['mean_atk_last_difference']

# ---- Features DEF ------#
def mean_def_last(dataset): #feature
    pkmn_database = open_pkmn_database_csv()
    mean_def=mean_def_database(pkmn_database)
    p1_mean_def=[]
    p2_mean_def=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_last(game)
        p2_team=extract_p2_team_from_game_last(game)

        p1_team=pkmn_database[pkmn_database['name'].isin(p1_team)]
        p1_known=len(extract_p1_team_from_game_start(game))
        p1_team=p1_team[['name','base_def']]
        p1_mean_def.append((np.sum(p1_team['base_def'])+ mean_def*(6-p1_known))/6)

        p2_team=pkmn_database[pkmn_database['name'].isin(p2_team)]
        p2_team=p2_team[['name','base_def']]
        p2_known=len(extract_p2_team_from_game_start(game))
        p2_mean_def.append((np.sum(p2_team['base_def'])+ mean_def*(6-p2_known))/6)

    mean_def_last=pd.DataFrame({'p1_mean_def_last':p1_mean_def,'p2_mean_def_last':p2_mean_def})
    mean_def_last['mean_def_last_difference']=np.subtract.reduce(mean_def_last[['p1_mean_def_last','p2_mean_def_last']],axis=1)
    mean_def_last=mean_def_last.fillna(value=0)
    return mean_def_last

def p1_mean_def_last(dataset)->pd.DataFrame: #feature
    return mean_def_last(dataset)['p1_mean_def_last']

def p2_mean_def_last(dataset)->pd.DataFrame: #feature
    return mean_def_last(dataset)['p2_mean_def_last']

def mean_def_last_difference(dataset)->pd.DataFrame: #feature
    return mean_def_last(dataset)['mean_def_last_difference']

# ---- Features SPA ------#
def mean_spa_last(dataset): #feature
    pkmn_database = open_pkmn_database_csv()
    mean_spa=mean_spa_database(pkmn_database)
    p1_mean_spa=[]
    p2_mean_spa=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_last(game)
        p2_team=extract_p2_team_from_game_last(game)

        p1_team=pkmn_database[pkmn_database['name'].isin(p1_team)]
        p1_known=len(extract_p1_team_from_game_start(game))
        p1_team=p1_team[['name','base_spa']]
        p1_mean_spa.append((np.sum(p1_team['base_spa'])+ mean_spa*(6-p1_known))/6)

        p2_team=pkmn_database[pkmn_database['name'].isin(p2_team)]
        p2_team=p2_team[['name','base_spa']]
        p2_known=len(extract_p2_team_from_game_start(game))
        p2_mean_spa.append((np.sum(p2_team['base_spa'])+ mean_spa*(6-p2_known))/6)

    mean_spa_last=pd.DataFrame({'p1_mean_spa_last':p1_mean_spa,'p2_mean_spa_last':p2_mean_spa})
    mean_spa_last['mean_spa_last_difference']=np.subtract.reduce(mean_spa_last[['p1_mean_spa_last','p2_mean_spa_last']],axis=1)
    mean_spa_last=mean_spa_last.fillna(value=0)
    return mean_spa_last

def p1_mean_spa_last(dataset)->pd.DataFrame: #feature
    return mean_spa_last(dataset)['p1_mean_spa_last']

def p2_mean_spa_last(dataset)->pd.DataFrame: #feature
    return mean_spa_last(dataset)['p2_mean_spa_last']

def mean_spa_last_difference(dataset)->pd.DataFrame: #feature
    return mean_spa_last(dataset)['mean_spa_last_difference']

# ---- Features SPD ------#
def mean_spd_last(dataset): #feature
    pkmn_database = open_pkmn_database_csv()
    mean_spd=mean_spd_database(pkmn_database)
    p1_mean_spd=[]
    p2_mean_spd=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_last(game)
        p2_team=extract_p2_team_from_game_last(game)

        p1_team=pkmn_database[pkmn_database['name'].isin(p1_team)]
        p1_known=len(extract_p1_team_from_game_start(game))
        p1_team=p1_team[['name','base_spd']]
        p1_mean_spd.append((np.sum(p1_team['base_spd'])+ mean_spd*(6-p1_known))/6)

        p2_team=pkmn_database[pkmn_database['name'].isin(p2_team)]
        p2_team=p2_team[['name','base_spd']]
        p2_known=len(extract_p2_team_from_game_start(game))
        p2_mean_spd.append((np.sum(p2_team['base_spd'])+ mean_spd*(6-p2_known))/6)

    mean_spd_last=pd.DataFrame({'p1_mean_spd_last':p1_mean_spd,'p2_mean_spd_last':p2_mean_spd})
    mean_spd_last['mean_spd_last_difference']=np.subtract.reduce(mean_spd_last[['p1_mean_spd_last','p2_mean_spd_last']],axis=1)
    mean_spd_last=mean_spd_last.fillna(value=0)
    return mean_spd_last

def p1_mean_spd_last(dataset)->pd.DataFrame: #feature
    return mean_spd_last(dataset)['p1_mean_spd_last']

def p2_mean_spd_last(dataset)->pd.DataFrame: #feature
    return mean_spd_last(dataset)['p2_mean_spd_last']

def mean_spd_last_difference(dataset)->pd.DataFrame: #feature
    return mean_spd_last(dataset)['mean_spd_last_difference']


def mean_total_database(pkmn_database) -> float:
    pkmn_database['total']=np.sum(pkmn_database[['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']],axis=1)
    return np.mean(pkmn_database['total'])

def mean_stats_start(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()
    mean_total=mean_total_database(pkmn_database)
    p1_mean_stats=[]
    p2_mean_stats=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_start(game).to_frame()
        p2_team=extract_p2_team_from_game_start(game).to_frame()

        p1_team=p1_team.merge(pkmn_database,how='inner',on='name')
        p1_known=len(p1_team)
        p1_team['total']=np.sum(p1_team[['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']],axis=1)
        p1_team=p1_team[['name','total']]
        p1_mean_stats.append((np.sum(p1_team['total'])+ (mean_total*(6-p1_known)))/6)
    
        p2_team=p2_team.merge(pkmn_database,how='inner',on='name')
        p2_known=len(p2_team)
        p2_team['total']=np.sum(p2_team[['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']],axis=1)
        p2_team=p2_team[['name','total']]
        p2_mean_stats.append((np.sum(p2_team['total'])+ mean_total*(6-p2_known))/6)

    mean_stats=pd.DataFrame({'p1_mean_stats_start':p1_mean_stats,'p2_mean_stats_start':p2_mean_stats})
    mean_stats['mean_stats_start_difference']=np.subtract.reduce(mean_stats[['p1_mean_stats_start','p2_mean_stats_start']],axis=1)
    #return mean_stats['mean_stats_difference']
    return mean_stats

def p1_mean_stats_start(dataset)-> pd.DataFrame: #feature
    return mean_stats_last(dataset)['p1_mean_stats_start']

def p2_mean_stats_start(dataset)-> pd.DataFrame: #feature
    return mean_stats_last(dataset)['p2_mean_stats_start']

def difference_mean_stats_start(dataset)-> pd.DataFrame: #feature
    return mean_stats_last(dataset)['mean_stats_start_difference']

def mean_stats_last(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()
    mean_total=mean_total_database(pkmn_database)
    p1_mean_stats=[]
    p2_mean_stats=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_last(game).to_frame()
        p2_team=extract_p2_team_from_game_last(game).to_frame()

        p1_team=p1_team.merge(pkmn_database,how='inner',on='name')
        p1_known=len(extract_p1_team_from_game_start(game))
        p1_team['total']=np.sum(p1_team[['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']],axis=1)
        p1_team=p1_team[['name','total']]
        p1_mean_stats.append((np.sum(p1_team['total'])+ (mean_total*(6-p1_known)))/6)
    
        p2_team=p2_team.merge(pkmn_database,how='inner',on='name')
        p2_known=len(extract_p2_team_from_game_start(game))
        p2_team['total']=np.sum(p2_team[['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']],axis=1)
        p2_team=p2_team[['name','total']]
        p2_mean_stats.append((np.sum(p2_team['total'])+ mean_total*(6-p2_known))/6)

    mean_stats=pd.DataFrame({'p1_mean_stats_last':p1_mean_stats,'p2_mean_stats_last':p2_mean_stats})
    mean_stats['mean_stats_last_difference']=np.subtract.reduce(mean_stats[['p1_mean_stats_last','p2_mean_stats_last']],axis=1)
    #return mean_stats['mean_stats_last_difference']
    return mean_stats

def p1_mean_stats_last(dataset)-> pd.DataFrame: #feature
    return mean_stats_last(dataset)['p1_mean_stats_last']

def p2_mean_stats_last(dataset)-> pd.DataFrame: #feature
    return mean_stats_last(dataset)['p2_mean_stats_last']

def difference_mean_stats_last(dataset)-> pd.DataFrame: #feature
    return mean_stats_last(dataset)['mean_stats_last_difference']

def p1_alive_pkmn(dataset)->pd.DataFrame: #feature
    pkmn_alive_p1=[]
    for game in dataset:
        turns=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])
        pkmn_dead_p1=turns[turns['status']=='fnt']['name'].drop_duplicates(keep='last')
        pkmn_alive_p1.append(6-len(pkmn_dead_p1))
    pkmn_alive_p1=pd.DataFrame(pkmn_alive_p1).rename(columns={0:'p1_pkmn_alive'})
    return pkmn_alive_p1


def p2_alive_pkmn(dataset)->pd.DataFrame: #feature
    pkmn_alive_p2=[]
    for game in dataset:
        turns=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
        pkmn_dead_p2=turns[turns['status']=='fnt']['name'].drop_duplicates(keep='last')
        pkmn_alive_p2.append(6-len(pkmn_dead_p2))
    pkmn_alive_p2=pd.DataFrame(pkmn_alive_p2).rename(columns={0:'p2_pkmn_alive'})
    return pkmn_alive_p2
   
def alive_pkmn_difference(dataset)->pd.DataFrame: #feature
    
    p1_alive=p1_alive_pkmn(dataset)
    p2_alive=p2_alive_pkmn(dataset)
    pkmn_alive_difference=pd.concat([p1_alive,p2_alive],axis=1)
    pkmn_alive_difference['pkmn_alive_difference']=np.subtract.reduce(pkmn_alive_difference[['p1_pkmn_alive','p2_pkmn_alive']],axis=1)
    #return pkmn_alive_difference['pkmn_alive_difference']
    return pkmn_alive_difference['pkmn_alive_difference']

def all_pokemon_round(player: int,json):
    if player==1:
        return set([elem['p1_pokemon_state']['name'] for elem in json['battle_timeline']])
    elif player==2:
        return set([elem['p2_pokemon_state']['name'] for elem in json['battle_timeline']])

def p1_reflect_ratio(dataset)->pd.DataFrame: #feature
    p1_count=[]
    for game in dataset:
        p1_timeline=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])
        p1_timeline['n_reflects']=p1_timeline['effects'].apply(lambda x: x[0].count('reflect'))
        #p1_count.append(np.sum(p1_timeline['n_reflects'])/30)
        p1_count.append(np.sum(p1_timeline['n_reflects']))
    return pd.DataFrame({'p1_reflect_ratio':p1_count})

def p2_reflect_ratio(dataset)->pd.DataFrame: #feature
    p2_count=[]
    for game in dataset:
        p2_timeline=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
        p2_timeline['n_reflects']=p2_timeline['effects'].apply(lambda x: x[0].count('reflect'))
        #p2_count.append(np.sum(p2_timeline['n_reflects'])/30)
        p2_count.append(np.sum(p2_timeline['n_reflects']))
    return pd.DataFrame({'p2_reflect_ratio':p2_count})

def p1_lightscreen_ratio(dataset)->pd.DataFrame: #feature
    p1_count=[]
    for game in dataset:
        p1_timeline=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])
        p1_timeline['n_lightscreens']=p1_timeline['effects'].apply(lambda x: x[0].count('lightscreen'))
        #p1_count.append(np.sum(p1_timeline['n_lightscreens'])/30)
        p1_count.append(np.sum(p1_timeline['n_lightscreens']))
    return pd.DataFrame({'p1_lightscreens_ratio':p1_count})

def p2_lightscreen_ratio(dataset)->pd.DataFrame: #feature
    p2_count=[]
    for game in dataset:
        p2_timeline=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
        p2_timeline['n_lightscreens']=p2_timeline['effects'].apply(lambda x: x[0].count('lightscreen'))
        #p2_count.append(np.sum(p2_timeline['n_lightscreens'])/30)
        p2_count.append(np.sum(p2_timeline['n_lightscreens']))
    return pd.DataFrame({'p2_lightscreens_ratio':p2_count})


def p1_frozen_pkmn(dataset)-> pd.DataFrame: #feature
    p1_count=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_last_with_stats(game)
        if len(p1_team)!=0:
            p1_count.append(len(p1_team[p1_team['status']=='frz']))
        else:
            p1_count.append(0)
        
    return pd.DataFrame({'p1_frozen_pkmn':p1_count})

def p2_frozen_pkmn(dataset)-> pd.DataFrame: #feature
    p2_count=[]
    for game in dataset:
        p2_team=extract_p2_team_from_game_last_with_stats(game)
        if len(p2_team)!=0:
            p2_count.append(len(p2_team[p2_team['status']=='frz']))
        else:
            p2_count.append(0)
        
    return pd.DataFrame({'p2_frozen_pkmn':p2_count})
    
def p1_paralized_pkmn(dataset)-> pd.DataFrame: #feature
    p1_count=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_last_with_stats(game)
        if len(p1_team)!=0:
            p1_count.append(len(p1_team[p1_team['status']=='par']))
        else:
            p1_count.append(0)
        
    return pd.DataFrame({'p1_paralized_pkmn':p1_count})

def p2_paralized_pkmn(dataset)-> pd.DataFrame: #feature
    p2_count=[]
    for game in dataset:
        p2_team=extract_p2_team_from_game_last_with_stats(game)
        if len(p2_team)!=0:
            p2_count.append(len(p2_team[p2_team['status']=='par']))
        else:
            p2_count.append(0)
        
    return pd.DataFrame({'p2_paralized_pkmn':p2_count})
    
def p1_sleep_pkmn(dataset)-> pd.DataFrame: #feature
    p1_count=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_last_with_stats(game)
        if len(p1_team)!=0:
            p1_count.append(len(p1_team[p1_team['status']=='slp']))
        else:
            p1_count.append(0)
        
    return pd.DataFrame({'p1_sleep_pkmn':p1_count})

def p2_sleep_pkmn(dataset)-> pd.DataFrame: #feature
    p2_count=[]
    for game in dataset:
        p2_team=extract_p2_team_from_game_last_with_stats(game)
        if len(p2_team)!=0:
            p2_count.append(len(p2_team[p2_team['status']=='slp']))
        else:
            p2_count.append(0)
        
    return pd.DataFrame({'p2_sleep_pkmn':p2_count})

def p1_poison_pkmn(dataset)-> pd.DataFrame: #feature
    p1_count=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_last_with_stats(game)
        if len(p1_team)!=0:
            p1_count.append(len(p1_team[p1_team['status']=='psn']))
        else:
            p1_count.append(0)
        
    return pd.DataFrame({'p1_poison_pkmn':p1_count})
 
def p2_poison_pkmn(dataset)-> pd.DataFrame: #feature
    p2_count=[]
    for game in dataset:
        p2_team=extract_p2_team_from_game_last_with_stats(game)
        if len(p2_team)!=0:
            p2_count.append(len(p2_team[p2_team['status']=='psn']))
        else:
            p2_count.append(0)
        
    return pd.DataFrame({'p2_poison_pkmn':p2_count})
    
def p1_burned_pkmn(dataset)-> pd.DataFrame: #feature
    p1_count=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_last_with_stats(game)
        if len(p1_team)!=0:
            p1_count.append(len(p1_team[p1_team['status']=='brn']))
        else:
            p1_count.append(0)
        
    return pd.DataFrame({'p1_burned_pkmn':p1_count})

def p2_burned_pkmn(dataset)-> pd.DataFrame: #feature
    p2_count=[]
    for game in dataset:
        p2_team=extract_p2_team_from_game_last_with_stats(game)
        if len(p2_team)!=0:
            p2_count.append(len(p2_team[p2_team['status']=='brn']))
        else:
            p2_count.append(0)
        
    return pd.DataFrame({'p2_burned_pkmn':p2_count})
    
def p1_pokemon_rest(dataset)->pd.DataFrame: #feature
    p1_count=[]
    for game in dataset:

        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='rest']
            turns=turns[turns['status']!='fnt']
            p1_count.append(len(turns))
        else:
            p1_count.append(0)
    return pd.DataFrame({'p1_pkmn_rest':p1_count})

def p2_pokemon_rest(dataset)->pd.DataFrame: #feature
    p2_count=[]
    for game in dataset:

        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='rest']
            turns=turns[turns['status']!='fnt']
            p2_count.append(len(turns))
        else:
            p2_count.append(0)
    return pd.DataFrame({'p2_pkmn_rest':p2_count})

def p1_pokemon_reflect(dataset)->pd.DataFrame: #feature
    p1_count=[]
    for game in dataset:

        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='reflect']
            turns=turns[turns['status']!='fnt']
            p1_count.append(len(turns))
        else:
            p1_count.append(0)
    return pd.DataFrame({'p1_pkmn_reflect':p1_count})

def p2_pokemon_reflect(dataset)->pd.DataFrame: #feature
    p2_count=[]
    for game in dataset:

        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='reflect']
            turns=turns[turns['status']!='fnt']
            p2_count.append(len(turns))
        else:
            p2_count.append(0)
    return pd.DataFrame({'p2_pkmn_reflect':p2_count})

def p1_pokemon_explosion(dataset)->pd.DataFrame: #feature
    p1_count=[]
    for game in dataset:

        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=pd.concat([turns[turns['name']=='explosion'], turns[turns['name']=='selfdestruct']],axis=0)
            #turns=turns[turns['status']!='fnt']
            p1_count.append(len(turns))
        else:
            p1_count.append(0)
        
    return pd.DataFrame({'p1_pkmn_explosions':p1_count})

def p2_pokemon_explosion(dataset)->pd.DataFrame: #feature
    p2_count=[]
    for game in dataset:

        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=pd.concat([turns[turns['name']=='explosion'], turns[turns['name']=='selfdestruct']],axis=0)
            #turns=turns[turns['status']!='fnt']
            p2_count.append(len(turns))
        else:
            p2_count.append(0)
    return pd.DataFrame({'p2_pkmn_explosions':p2_count})

def p1_pokemon_thunderwave(dataset)->pd.DataFrame: #feature
    p1_count=[]
    for game in dataset:

        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='thunderwave']
            turns=turns[turns['status']!='fnt']
            p1_count.append(len(turns))
        else:
            p1_count.append(0)
    return pd.DataFrame({'p1_pkmn_thunderwave':p1_count})

def p2_pokemon_thunderwave(dataset)->pd.DataFrame: #feature
    p2_count=[]
    for game in dataset:

        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='thunderwave']
            turns=turns[turns['status']!='fnt']
            p2_count.append(len(turns))
        else:
            p2_count.append(0)
    return pd.DataFrame({'p2_pkmn_thunderwave':p2_count})

def p1_pokemon_recover(dataset)->pd.DataFrame: #feature
    p1_count=[]
    for game in dataset:

        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='recover']
            turns=turns[turns['status']!='fnt']
            p1_count.append(len(turns))
        else:
            p1_count.append(0)
    return pd.DataFrame({'p1_pkmn_recover':p1_count})

def p2_pokemon_recover(dataset)->pd.DataFrame: #feature
    p2_count=[]
    for game in dataset:

        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='recover']
            turns=turns[turns['status']!='fnt']
            p2_count.append(len(turns))
        else:
            p2_count.append(0)
    return pd.DataFrame({'p2_pkmn_recover':p2_count})

def p1_pokemon_toxic(dataset)->pd.DataFrame: #feature
    p1_count=[]
    for game in dataset:

        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='toxic']
            turns=turns[turns['status']!='fnt']
            p1_count.append(len(turns))
        else:
            p1_count.append(0)
    return pd.DataFrame({'p1_pkmn_toxic':p1_count})

def p2_pokemon_toxic(dataset)->pd.DataFrame: #feature
    p2_count=[]
    for game in dataset:

        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='toxic']
            turns=turns[turns['status']!='fnt']
            p2_count.append(len(turns))
        else:
            p2_count.append(0)
    return pd.DataFrame({'p2_pkmn_toxic':p2_count})

def p1_pokemon_firespin(dataset)->pd.DataFrame: #feature
    p1_count=[]
    for game in dataset:

        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='firespin']
            turns=turns[turns['status']!='fnt']
            p1_count.append(len(turns))
        else:
            p1_count.append(0)
    return pd.DataFrame({'p1_pkmn_firespin':p1_count})

def p2_pokemon_firespin(dataset)->pd.DataFrame: #feature
    p2_count=[]
    for game in dataset:

        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='firespin']
            turns=turns[turns['status']!='fnt']
            p2_count.append(len(turns))
        else:
            p2_count.append(0)
    return pd.DataFrame({'p2_pkmn_firespin':p2_count})

def p1_pokemon_stab(dataset)->pd.DataFrame: #feature
    pkmn_database=open_pkmn_database_csv()
    p1_count=[]
    for game in dataset:
        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])
        turns_p1_state=turns_p1_state.merge(pkmn_database,how='inner',on='name').rename(columns={'name':'pkmn_name'})[['pkmn_name','status','type_1','type_2']]
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None','type':'None','category':'None'} for turn in game['battle_timeline']])[['name','type','category']]
        turns_p1_moves=turns_p1_moves.rename(columns={'type':'move_type'})
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).dropna()
            turns['move_type']=turns['move_type'].apply(str.lower)
            turns['category']=turns['category'].apply(str.lower)
            turns=turns.query("category!='status' and ((type_1==move_type) or (type_2==move_type))")
            p1_count.append(len(turns))
        else:
            p1_count.append(0)
    return pd.DataFrame({'p1_pkmn_stab_used':p1_count})

def p2_pokemon_stab(dataset)->pd.DataFrame: #feature
    pkmn_database=open_pkmn_database_csv()
    p2_count=[]
    for game in dataset:
        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
        turns_p2_state=turns_p2_state.merge(pkmn_database,how='inner',on='name').rename(columns={'name':'pkmn_name'})[['pkmn_name','status','type_1','type_2']]
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None','type':'None','category':'None'} for turn in game['battle_timeline']])[['name','type','category']]
        turns_p2_moves=turns_p2_moves.rename(columns={'type':'move_type'})
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).dropna()
            turns['move_type']=turns['move_type'].apply(str.lower)
            turns['category']=turns['category'].apply(str.lower)
            turns=turns.query("category!='status' and ((type_1==move_type) or (type_2==move_type))")
            p2_count.append(len(turns))
        else:
            p2_count.append(0)
    return pd.DataFrame({'p2_pkmn_stab_used':p2_count})

def p1_switches_count(dataset) -> pd.DataFrame: #feature
    """
    Conta il numero di volte che P1 cambia Pokémon durante la battaglia.
    Molti switch possono indicare strategia difensiva o problemi di matchup.
    """
    switches = []
    for game in dataset:
        switch_count = 0
        prev_pokemon = None
        
        for turn in game['battle_timeline']:
            current_pokemon = turn['p1_pokemon_state']['name']
            if prev_pokemon is not None and current_pokemon != prev_pokemon:
                switch_count += 1
            prev_pokemon = current_pokemon
        
        switches.append(switch_count)
    
    return pd.DataFrame({'p1_switches_count': switches})


def p2_switches_count(dataset) -> pd.DataFrame: #feature
    """
    Conta il numero di volte che P2 cambia Pokémon durante la battaglia.
    """
    switches = []
    for game in dataset:
        switch_count = 0
        prev_pokemon = None
        
        for turn in game['battle_timeline']:
            current_pokemon = turn['p2_pokemon_state']['name']
            if prev_pokemon is not None and current_pokemon != prev_pokemon:
                switch_count += 1
            prev_pokemon = current_pokemon
        
        switches.append(switch_count)
    
    return pd.DataFrame({'p2_switches_count': switches})


def p1_status_inflicted(dataset) -> pd.DataFrame: #feature
    """
    Conta quanti status conditions P1 è riuscito ad infliggere a P2.
    Include: paralysis (par), burn (brn), poison (psn), toxic (tox), sleep (slp), freeze (frz).
    """
    status_count = []
    for game in dataset:
        count = 0
        prev_status = 'nostatus'
        
        for turn in game['battle_timeline']:
            current_status = turn['p2_pokemon_state']['status']
            # Conta quando P2 passa da nostatus a uno status (inflitto da P1)
            if prev_status == 'nostatus' and current_status not in ['nostatus', 'fnt']:
                count += 1
            prev_status = current_status
        
        status_count.append(count)
    
    return pd.DataFrame({'p1_status_inflicted': status_count})


def p2_status_inflicted(dataset) -> pd.DataFrame: #feature
    """
    Conta quanti status conditions P2 è riuscito ad infliggere a P1.
    """
    status_count = []
    for game in dataset:
        count = 0
        pokemon_status_map = {}  # tiene traccia dello status per ogni pokemon
        
        for turn in game['battle_timeline']:
            pokemon_name = turn['p1_pokemon_state']['name']
            current_status = turn['p1_pokemon_state']['status']
            
            # Se il pokemon non è ancora nella mappa, lo aggiungiamo
            if pokemon_name not in pokemon_status_map:
                pokemon_status_map[pokemon_name] = 'nostatus'
            
            prev_status = pokemon_status_map[pokemon_name]
            
            # Conta quando P1 passa da nostatus a uno status (inflitto da P2)
            if prev_status == 'nostatus' and current_status not in ['nostatus', 'fnt']:
                count += 1
            
            pokemon_status_map[pokemon_name] = current_status
        
        status_count.append(count)
    
    return pd.DataFrame({'p2_status_inflicted': status_count})


def switches_difference(dataset) -> pd.DataFrame: #feature
    """
    Differenza tra switch di P1 e P2.
    Valori positivi indicano che P1 ha switchato più spesso (potenzialmente più difensivo).
    """
    p1_sw = p1_switches_count(dataset)
    p2_sw = p2_switches_count(dataset)
    
    diff = p1_sw['p1_switches_count'] - p2_sw['p2_switches_count']
    
    return pd.DataFrame({'switches_difference': diff})


def status_inflicted_difference(dataset) -> pd.DataFrame: #feature
    """
    Differenza tra status inflitti da P1 e P2.
    Valori positivi indicano che P1 ha inflitto più status conditions.
    """
    p1_status = p1_status_inflicted(dataset)
    p2_status = p2_status_inflicted(dataset)
    
    diff = p1_status['p1_status_inflicted'] - p2_status['p2_status_inflicted']
    
    return pd.DataFrame({'status_inflicted_difference': diff})


def p1_final_team_hp(dataset) -> pd.DataFrame: #feature
    """
    Calcola l'HP totale rimanente del team P1 alla fine della battaglia.
    Somma gli hp_pct di tutti i Pokémon ancora vivi moltiplicati per i loro base_hp.
    """
    pkmn_database = open_pkmn_database_csv()
    final_hp = []
    
    for game in dataset:
        # Ottieni l'ultimo turno
        last_turn = game['battle_timeline'][-1]
        
        # Trova tutti i Pokémon P1 ancora vivi (non fnt)
        all_turns = pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])
        alive_pokemon = []
        
        # Per ogni Pokémon nel team, trova l'ultimo stato
        for pkmn in game['p1_team_details']:
            pkmn_name = pkmn['name']
            pkmn_turns = all_turns[all_turns['name'] == pkmn_name]
            
            if len(pkmn_turns) > 0:
                last_state = pkmn_turns.iloc[-1]
                if last_state['status'] != 'fnt':
                    # Calcola HP rimanente: hp_pct * base_hp
                    hp_remaining = last_state['hp_pct'] * pkmn['base_hp']
                    alive_pokemon.append(hp_remaining)
        
        #final_hp.append(sum(alive_pokemon) if alive_pokemon else 0)
        p1_known=len(extract_p1_team_from_game_start(game))
        hp_team_known=sum(alive_pokemon) if alive_pokemon else 0
        final_hp.append(hp_team_known+(mean_hp_database(pkmn_database)*(6-p1_known)))
    
    return pd.DataFrame({'p1_final_team_hp': final_hp})


def p2_final_team_hp(dataset) -> pd.DataFrame: #feature
    """
    Calcola l'HP totale rimanente del team P2 alla fine della battaglia.
    """
    pkmn_database = open_pkmn_database_csv()
    final_hp = []
    
    for game in dataset:
        # Trova tutti i Pokémon P2 visti durante la battaglia
        all_turns = pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
        alive_pokemon = []
        
        # Per ogni Pokémon unico di P2, trova l'ultimo stato
        unique_pokemon = all_turns['name'].unique()
        
        for pkmn_name in unique_pokemon:
            pkmn_turns = all_turns[all_turns['name'] == pkmn_name]
            last_state = pkmn_turns.iloc[-1]
            
            if last_state['status'] != 'fnt':
                # Ottieni base_hp dal database
                pkmn_info = pkmn_database[pkmn_database['name'] == pkmn_name]
                if len(pkmn_info) > 0:
                    base_hp = pkmn_info.iloc[0]['base_hp']
                    hp_remaining = last_state['hp_pct'] * base_hp
                    alive_pokemon.append(hp_remaining)
        
        #final_hp.append(sum(alive_pokemon) if alive_pokemon else 0) # forse qua andrebbe integrata con la media generale
        p2_known=len(extract_p2_team_from_game_start(game))
        hp_team_known=sum(alive_pokemon) if alive_pokemon else 0
        final_hp.append(hp_team_known+(mean_hp_database(pkmn_database)*(6-p2_known)))

    return pd.DataFrame({'p2_final_team_hp': final_hp})


def final_team_hp_difference(dataset) -> pd.DataFrame: #feature
    """
    Differenza tra HP finale del team P1 e P2.
    Valori positivi indicano che P1 ha finito con più HP totale.
    """
    p1_hp = p1_final_team_hp(dataset)
    p2_hp = p2_final_team_hp(dataset)
    
    diff = p1_hp['p1_final_team_hp'] - p2_hp['p2_final_team_hp']
    
    return pd.DataFrame({'final_team_hp_difference': diff})


def p1_first_faint_turn(dataset) -> pd.DataFrame: #feature
    """
    Turno in cui P1 perde il primo Pokémon.
        Valori alti indicano che P1 è riuscito a resistere più a lungo.
        Valori negativi indicano il turno in cui è avvenuto il primo faint (es. -3 significa che il primo faint è avvenuto al terzo turno).
        """
    first_faint = []
    
    for game in dataset:
        faint_turn = None
        pokemon_fainted = set()
        
        for turn in game['battle_timeline']:
            pkmn_name = turn['p1_pokemon_state']['name']
            pkmn_status = turn['p1_pokemon_state']['status']
            
            if pkmn_status == 'fnt' and pkmn_name not in pokemon_fainted:
                faint_turn = turn['turn']
                pokemon_fainted.add(pkmn_name)
                break
        
        # Se nessun Pokémon è svenuto, usa l'ultimo turno + 1
        if faint_turn is None:
            faint_turn = -(len(game['battle_timeline']) + 1)
        else:
            faint_turn = - faint_turn

        first_faint.append(faint_turn)
    
    return pd.DataFrame({'p1_first_faint_turn': first_faint})


def p1_avg_hp_when_switching(dataset) -> pd.DataFrame: #feature
    """
    HP percentuale medio dei Pokémon P1 quando effettuano uno switch.
    Valori bassi indicano switch difensivi (Pokémon in difficoltà).
    Valori alti indicano switch offensivi/strategici.
    """
    avg_hp_switches = []
    
    for game in dataset:
        switch_hp = []
        prev_pokemon = None
        
        for i, turn in enumerate(game['battle_timeline']):
            current_pokemon = turn['p1_pokemon_state']['name']
            
            # Se c'è stato uno switch
            if prev_pokemon is not None and current_pokemon != prev_pokemon:
                # HP del Pokémon che ha switchato out (turno precedente)
                if i > 0:
                    prev_turn = game['battle_timeline'][i-1]
                    switch_hp.append(prev_turn['p1_pokemon_state']['hp_pct'])
            
            prev_pokemon = current_pokemon
        
        # Media degli HP quando si switcha
        avg_hp = np.mean(switch_hp) if switch_hp else 1.0  # 1.0 se non ci sono switch
        avg_hp_switches.append(avg_hp)
    
    return pd.DataFrame({'p1_avg_hp_when_switching': avg_hp_switches})

def p2_avg_hp_when_switching(dataset) -> pd.DataFrame: #feature
    """
    HP percentuale medio dei Pokémon P2 quando effettuano uno switch.
    """
    avg_hp_switches = []
    
    for game in dataset:
        switch_hp = []
        prev_pokemon = None
        
        for i, turn in enumerate(game['battle_timeline']):
            current_pokemon = turn['p2_pokemon_state']['name']
            
            # Se c'è stato uno switch
            if prev_pokemon is not None and current_pokemon != prev_pokemon:
                # HP del Pokémon che ha switchato out (turno precedente)
                if i > 0:
                    prev_turn = game['battle_timeline'][i-1]
                    switch_hp.append(prev_turn['p2_pokemon_state']['hp_pct'])
            
            prev_pokemon = current_pokemon
        
        # Media degli HP quando si switcha
        avg_hp = np.mean(switch_hp) if switch_hp else 1.0  # 1.0 se non ci sono switch
        avg_hp_switches.append(avg_hp)
    
    return pd.DataFrame({'p2_avg_hp_when_switching': avg_hp_switches})

def p1_max_debuff_received(dataset) -> pd.DataFrame: #feature
    """
    Massimo debuff (boost negativo) ricevuto da P1 su una singola stat.
    Valori negativi più bassi indicano debuff pesanti subiti.
    """
    max_debuff_list = []
    
    for game in dataset:
        max_debuff = 0
        
        for turn in game['battle_timeline']:
            boosts = turn['p1_pokemon_state']['boosts']
            
            # Trova il debuff più forte (valore più negativo)
            turn_min = min(boosts.values())
            if turn_min < max_debuff:
                max_debuff = turn_min
        
        max_debuff_list.append(max_debuff)
    
    return pd.DataFrame({'p1_max_debuff_received': max_debuff_list})


def p2_max_debuff_received(dataset) -> pd.DataFrame: #feature
    """
    Massimo debuff (boost negativo) ricevuto da P2 su una singola stat.
    """
    max_debuff_list = []
    
    for game in dataset:
        max_debuff = 0
        
        for turn in game['battle_timeline']:
            boosts = turn['p2_pokemon_state']['boosts']
            
            # Trova il debuff più forte (valore più negativo)
            turn_min = min(boosts.values())
            if turn_min < max_debuff:
                max_debuff = turn_min
        
        max_debuff_list.append(max_debuff)
    
    return pd.DataFrame({'p2_max_debuff_received': max_debuff_list})


def extract_types_from_team_p1(game)-> pd.DataFrame:

    pkmn_database=open_pkmn_database_csv()
    p1_team=extract_p1_team_from_game_start(game).to_frame()
    p1_team=p1_team.merge(pkmn_database, how='inner', on='name')
    p1_team=p1_team[['name','types']]

    types=pd.DataFrame([type.split(",") for pokemon in p1_team['types'] for type in [pokemon.strip("[]").replace("'","").replace(" ","")]])
    p1_team_types=p1_team.drop('types',axis=1)
    p1_team_types['type_1']=types[0]
    p1_team_types['type_2']=types[1]
  
    return p1_team_types

def extract_types_from_team_p1_last (game)-> pd.DataFrame:

    pkmn_database=open_pkmn_database_csv()
    p1_team=extract_p1_team_from_game_last(game).to_frame()
    if len(p1_team)!=0:
        p1_team=p1_team.merge(pkmn_database, how='inner', on='name')
        p1_team=p1_team[['name','types']]

        types=pd.DataFrame([type.split(",") for pokemon in p1_team['types'] for type in [pokemon.strip("[]").replace("'","").replace(" ","")]])
        p1_team_types=p1_team.drop('types',axis=1)
        p1_team_types['type_1']=types[0]
        p1_team_types['type_2']=types[1]
    
        return p1_team_types
    return pd.DataFrame()

def extract_types_from_team_p2(game)-> pd.DataFrame:

    pkmn_database=open_pkmn_database_csv()
    p2_team=extract_p2_team_from_game_start(game).to_frame()
    p2_team=p2_team.merge(pkmn_database, how='inner', on='name')
    p2_team=p2_team[['name','types']]

    types=pd.DataFrame([type.split(",") for pokemon in p2_team['types'] for type in [pokemon.strip("[]").replace("'","").replace(" ","")]])
    p2_team_types=p2_team.drop('types',axis=1)
    p2_team_types['type_1']=types[0]
    p2_team_types['type_2']=types[1]
  
    return p2_team_types

def extract_types_from_team_p2_last(game)-> pd.DataFrame:

    pkmn_database=open_pkmn_database_csv()
    p2_team=extract_p2_team_from_game_last(game).to_frame()
    if len(p2_team)!=0:
        p2_team=p2_team.merge(pkmn_database, how='inner', on='name')
        p2_team=p2_team[['name','types']]

        types=pd.DataFrame([type.split(",") for pokemon in p2_team['types'] for type in [pokemon.strip("[]").replace("'","").replace(" ","")]])
        p2_team_types=p2_team.drop('types',axis=1)
        p2_team_types['type_1']=types[0]
        p2_team_types['type_2']=types[1]
    
        return p2_team_types
    return pd.DataFrame()

def p1_psy_pkmn(dataset)-> pd.DataFrame:
    p1_count=[]
    for game in dataset:
        p1_team=extract_types_from_team_p1_last(game)
        if len(p1_team)!=0:
            p1_team=p1_team.query("type_1=='psychic' or type_2=='psychic'")
            p1_count.append(len(p1_team))
        else:
            p1_count.append(0)
    return pd.DataFrame({'p1_psychic_pkmn_last':p1_count})

def p2_psy_pkmn(dataset)-> pd.DataFrame:
    p2_count=[]
    for game in dataset:
        p2_team=extract_types_from_team_p2_last(game)
        if len(p2_team)!=0:
            p2_team=p2_team.query("type_1=='psychic' or type_2=='psychic'")
            p2_count.append(len(p2_team))
        else:
            p2_count.append(0)
    return pd.DataFrame({'p2_psychic_pkmn_last':p2_count})


def calc_weakness(type_1,type_2)->pd.DataFrame:
    type_chart=open_type_chart_json()
    weaknesses=pd.DataFrame([])

    if type_1=='notype' and type_2!='notype':
            type_col=type_chart[type_2].copy()
            weaknesses=type_col[type_col>=2]

    elif type_1!='notype' and type_2=='notype':
            type_col=type_chart[type_1].copy()
            weaknesses=type_col[type_col>=2]

    elif type_1!='notype' and type_2!='notype':
            type_col=type_chart[[type_1,type_2]].copy()
            type_col['total']=type_col.prod(axis=1)
            #type=[elem for elem in type if elem[1]>=2]
            weaknesses=type_col[type_col['total']>=2]
            weaknesses=weaknesses['total']

    return weaknesses

def weakness_teams_not_opt(dataset) -> pd.DataFrame: #---> DONT USE <---
    weak_games_p1,weak_games_p2=[],[]
    for game in dataset:
        #if game['battle_id']==0:
            weakness_p1=[]
            p1_team_types=extract_types_from_team_p1(game)
            for index,row in p1_team_types.iterrows():
                weaknesses=calc_weakness(row.iloc[1],row.iloc[2])
                weakness_p1.append(weaknesses)
                #print(weaknesses,"\n")
            weakness_p1=pd.concat(weakness_p1).reset_index().rename(columns={'index':'type'}).drop_duplicates(subset='type').reset_index(drop=True)
            weakness_p1=weakness_p1['type']

            weakness_p2=[]
            p2_team_types=extract_types_from_team_p2(game)
            for index,row in p2_team_types.iterrows():
                weaknesses=calc_weakness(row.iloc[1],row.iloc[2])
                weakness_p2.append(weaknesses)
                #print(weaknesses,"\n")
            weakness_p2=pd.concat(weakness_p2).reset_index().rename(columns={'index':'type'}).drop_duplicates(subset='type').reset_index(drop=True)
            weakness_p2=weakness_p2['type']
    
            weak_games_p1.append(weakness_p1.count())
            weak_games_p2.append(weakness_p2.count())
    
    return pd.DataFrame({'weakness_start_p1':weak_games_p1,'weakness_start_p2':weak_games_p2})

def weakness_teams(dataset) ->pd.DataFrame:
    pkmn_db_weak=open_pkmn_database_weak_csv()
    pkmn_db_weak=pd.DataFrame(pkmn_db_weak[['name','weaknesses']])
    pkmn_db_weak['weaknesses']=pkmn_db_weak['weaknesses'].apply(lambda x: x.strip("[] ").replace("'","").replace(" ","").split(","))
    
    #print(pkmn_db_weak)
    weak_games_p1,weak_games_p2=[],[]

    for game in dataset:
            p1_team_types=extract_types_from_team_p1(game)
            p1_team_types=p1_team_types.merge(pkmn_db_weak,how='inner',on='name')
            sw_1=set(sum(p1_team_types['weaknesses'],[]))
            weak_games_p1.append(len(sw_1))

            p2_team_types=extract_types_from_team_p2(game)
            p2_team_types=p2_team_types.merge(pkmn_db_weak,how='inner',on='name')
            sw_2=set(sum(p2_team_types['weaknesses'],[]))
            weak_games_p2.append(len(sw_2))
            
    weakness_teams=pd.DataFrame({'weakness_start_p1':weak_games_p1,'weakness_start_p2':weak_games_p2})
    weakness_teams['weakness_start_difference']=np.subtract.reduce(weakness_teams[['weakness_start_p1','weakness_start_p2']],axis=1)
    #return weakness_teams['weakness_start_difference']
    return weakness_teams

def p1_weakness_start(dataset) -> pd.DataFrame : #Feature
    return weakness_teams(dataset)['weakness_start_p1']

def p2_weakness_start(dataset) -> pd.DataFrame : #Feature
    return weakness_teams(dataset)['weakness_start_p2']

def weakness_start_difference(dataset) -> pd.DataFrame: #feature
    return  weakness_teams(dataset)['weakness_start_difference']  

def weakness_teams_last(dataset) ->pd.DataFrame: #feature
    pkmn_db_weak=open_pkmn_database_weak_csv()
    pkmn_db_weak=pd.DataFrame(pkmn_db_weak[['name','weaknesses']])
    pkmn_db_weak['weaknesses']=pkmn_db_weak['weaknesses'].apply(lambda x: x.strip("[] ").replace("'","").replace(" ","").split(","))
    
    #print(pkmn_db_weak)
    weak_games_p1,weak_games_p2=[],[]

    
    for game in dataset:
            p1_team_types=extract_types_from_team_p1_last(game)
            if len(p1_team_types)!=0:
                p1_team_types=p1_team_types.merge(pkmn_db_weak,how='inner',on='name')
                sw_1=set(sum(p1_team_types['weaknesses'],[]))
                weak_games_p1.append(len(sw_1))
            else:
                weak_games_p1.append(0)

            p2_team_types=extract_types_from_team_p2_last(game)
            if len(p2_team_types)!=0:
                p2_team_types=p2_team_types.merge(pkmn_db_weak,how='inner',on='name')
                sw_2=set(sum(p2_team_types['weaknesses'],[]))
                weak_games_p2.append(len(sw_2))
            else:
                weak_games_p2.append(0)
            
    weakness_teams=pd.DataFrame({'weakness_last_p1':weak_games_p1,'weakness_last_p2':weak_games_p2})
    weakness_teams['weakness_last_difference']=np.subtract.reduce(weakness_teams[['weakness_last_p1','weakness_last_p2']],axis=1)
    #return weakness_teams['weakness_last_difference']
    return weakness_teams

def p1_weakness_last(dataset) -> pd.DataFrame : #Feature
    return weakness_teams_last(dataset)['weakness_last_p1']

def p2_weakness_last(dataset) -> pd.DataFrame : #Feature
    return weakness_teams_last(dataset)['weakness_last_p2']

def weakness_last_difference(dataset) -> pd.DataFrame: #feature
    return  weakness_teams_last(dataset)['weakness_last_difference']  

def advantage_weak_start(dataset) ->pd.DataFrame: #feature
    pkmn_db_weak=open_pkmn_database_weak_csv()
    pkmn_db_weak=pd.DataFrame(pkmn_db_weak[['name','weaknesses']])
    pkmn_db_weak['weaknesses']=pkmn_db_weak['weaknesses'].apply(lambda x: x.strip("[] ").replace("'","").replace(" ","").split(","))
    
    #print(pkmn_db_weak)
    adv_games_p1,adv_games_p2=[],[]

    
    for game in dataset:
        #if game['battle_id']==0 or game['battle_id']==9999:
            p1_team_types=extract_types_from_team_p1(game)
            p2_team_types=extract_types_from_team_p2(game)

            p1_team_weakness=p1_team_types.merge(pkmn_db_weak,how='inner',on='name')
            p2_team_weakness=p2_team_types.merge(pkmn_db_weak,how='inner',on='name')

            sw_1=set(sum(p1_team_weakness['weaknesses'],[]))
            sw_2=set(sum(p2_team_weakness['weaknesses'],[]))

            all_type_s1=set((p1_team_types['type_1'].to_list()+p1_team_types['type_2'].to_list()))
            all_type_s2=set((p2_team_types['type_1'].to_list()+p2_team_types['type_2'].to_list()))

            if len(all_type_s1)!=0:
                all_type_s1.discard("notype")
                advantages_p1=all_type_s1.intersection(sw_2)
                adv_games_p1.append(len(advantages_p1))
            else:
                adv_games_p1.append(0)
            if len(all_type_s2)!=0:
                all_type_s2.discard("notype")
                advantages_p2=all_type_s2.intersection(sw_1)
                adv_games_p2.append(len(advantages_p2))
            else:
                adv_games_p2.append(0)

            #print("Start:\n")
            #print(all_type_s1," ", sw_2," ",advantages_p1, "\n",all_type_s2, " ", sw_1," ",advantages_p2,"\n")
    
    advantage_weak_teams=pd.DataFrame({'advantage_weak_start_p1':adv_games_p1,'advantage_weak_start_p2':adv_games_p2})
    advantage_weak_teams['advantage_weak_difference']=np.subtract.reduce(advantage_weak_teams[['advantage_weak_start_p1','advantage_weak_start_p2']],axis=1)
    #return advantage_weak_teams['advantage_weak_difference']
    #return advantage_weak_teams['advantage_weak_start_p2']
    return advantage_weak_teams
        
def p1_advantage_weak_start(dataset) -> pd.DataFrame : #Feature
    return advantage_weak_last(dataset)['advantage_weak_start_p1']

def p2_advantage_weak_start(dataset) -> pd.DataFrame : #feature
    return advantage_weak_last(dataset)['advantage_weak_start_p2']  

def advantage_weak_start_difference(dataset) -> pd.DataFrame: #feature
    return advantage_weak_last(dataset)['advantage_weak_difference']  

def advantage_weak_last(dataset) ->pd.DataFrame: #feature
    pkmn_db_weak=open_pkmn_database_weak_csv()
    pkmn_db_weak=pd.DataFrame(pkmn_db_weak[['name','weaknesses']])
    pkmn_db_weak['weaknesses']=pkmn_db_weak['weaknesses'].apply(lambda x: x.strip("[] ").replace("'","").replace(" ","").split(","))
    
    #print(pkmn_db_weak)
    adv_games_p1,adv_games_p2=[],[]

    
    for game in dataset:
        #if game['battle_id']==0 or game['battle_id']==9999:
            p1_team_types=extract_types_from_team_p1_last(game)
            p2_team_types=extract_types_from_team_p2_last(game)
            all_type_s1,all_type_s2=set(),set()

            if len(p1_team_types)!=0 and len(p2_team_types)!=0:
                p1_team_weakness=p1_team_types.merge(pkmn_db_weak,how='inner',on='name')
                sw_1=set(sum(p1_team_weakness['weaknesses'],[]))
                all_type_s1=set((p1_team_types['type_1'].to_list()+p1_team_types['type_2'].to_list()))
            
                p2_team_weakness=p2_team_types.merge(pkmn_db_weak,how='inner',on='name')
                sw_2=set(sum(p2_team_weakness['weaknesses'],[]))
                all_type_s2=set((p2_team_types['type_1'].to_list()+p2_team_types['type_2'].to_list()))

            if len(all_type_s1)!=0:
                all_type_s1.discard("notype")
                advantages_p1=all_type_s1.intersection(sw_2)
                adv_games_p1.append(len(advantages_p1))
            else:
                adv_games_p1.append(0)

            if len(all_type_s2)!=0:
                all_type_s2.discard("notype")
                advantages_p2=all_type_s2.intersection(sw_1)
                adv_games_p2.append(len(advantages_p2))
            else:
                adv_games_p2.append(0)

            #print("Last:\n")
            #print(all_type_s1," ", sw_2," ",advantages_p1, "\n",all_type_s2, " ", sw_1," ",advantages_p2,"\n")
    advantage_weak_teams=pd.DataFrame({'advantage_weak_last_p1':adv_games_p1,'advantage_last_start_p2':adv_games_p2})
    advantage_weak_teams['advantage_weak_last_difference']=np.subtract.reduce(advantage_weak_teams[['advantage_weak_last_p1','advantage_last_start_p2']],axis=1)
    #return advantage_weak_teams['advantage_weak_last_difference'] 
    #return advantage_weak_teams['advantage_weak_last_p1']
    return advantage_weak_teams

def p1_advantage_weak_last(dataset) -> pd.DataFrame : #Feature
    return advantage_weak_last(dataset)['advantage_weak_last_p1']

def p2_advantage_weak_last(dataset) -> pd.DataFrame : #feature
    return advantage_weak_last(dataset)['advantage_weak_last_p2']  

def advantage_weak_last_difference(dataset) -> pd.DataFrame: #feature
    return advantage_weak_last(dataset)['advantage_weak_last_difference']  

def p1_avg_move_power(dataset) -> pd.DataFrame: #feature
    """
    Calcola la potenza media delle mosse usate da P1 durante tutta la battaglia.
    Esclude mosse con base_power = 0 (mosse di status).
    Valori più alti indicano un approccio più offensivo.
    """
    avg_powers = []
    
    for game in dataset:
        move_powers = []
        
        for turn in game['battle_timeline']:
            move_details = turn['p1_move_details']
            
            # Se c'è una mossa e ha base_power > 0 (non è una mossa di status)
            if move_details is not None and move_details['base_power'] > 0:
                move_powers.append(move_details['base_power'])
        
        # Calcola la media, se non ci sono mosse offensive usa 0
        avg_power = np.mean(move_powers) if move_powers else 0
        avg_powers.append(avg_power)
    
    return pd.DataFrame({'p1_avg_move_power': avg_powers})


def p2_avg_move_power(dataset) -> pd.DataFrame: #feature
    """
    Calcola la potenza media delle mosse usate da P2 durante tutta la battaglia.
    Esclude mosse con base_power = 0 (mosse di status).
    """
    avg_powers = []
    
    for game in dataset:
        move_powers = []
        
        for turn in game['battle_timeline']:
            move_details = turn['p2_move_details']
            
            # Se c'è una mossa e ha base_power > 0 (non è una mossa di status)
            if move_details is not None and move_details['base_power'] > 0:
                move_powers.append(move_details['base_power'])
        
        # Calcola la media, se non ci sono mosse offensive usa 0
        avg_power = np.mean(move_powers) if move_powers else 0
        avg_powers.append(avg_power)
    
    return pd.DataFrame({'p2_avg_move_power': avg_powers})


def avg_move_power_difference(dataset) -> pd.DataFrame: #feature
    """
    Differenza tra la potenza media delle mosse di P1 e P2.
    Valori positivi indicano che P1 usa mosse più potenti in media.
    """
    p1_power = p1_avg_move_power(dataset)
    p2_power = p2_avg_move_power(dataset)
    
    diff = p1_power['p1_avg_move_power'] - p2_power['p2_avg_move_power']
    
    return pd.DataFrame({'avg_move_power_difference': diff})


def p1_offensive_ratio(dataset) -> pd.DataFrame: #feature
    """
    Rapporto tra statistiche offensive (Atk + SpA) e difensive (Def + SpD) per il team P1.
    Ratio > 1: team offensivo
    Ratio < 1: team difensivo
    Ratio ~ 1: team bilanciato
    """
    ratios = []
    
    for game in dataset:
        p1_team = game['p1_team_details']
        
        total_offensive = 0
        total_defensive = 0
        
        for pokemon in p1_team:
            total_offensive += pokemon['base_atk'] + pokemon['base_spa']
            total_defensive += pokemon['base_def'] + pokemon['base_spd']
        
        # Calcola il rapporto, evita divisione per zero
        ratio = total_offensive / total_defensive if total_defensive > 0 else 0
        ratios.append(ratio)
    
    return pd.DataFrame({'p1_offensive_ratio': ratios})


def p2_offensive_ratio(dataset) -> pd.DataFrame: #feature
    """
    Rapporto tra statistiche offensive (Atk + SpA) e difensive (Def + SpD) per il team P2.
    Nota: P2 ha solo informazioni sul lead e sui Pokémon visti durante la battaglia.
    """
    pkmn_database = open_pkmn_database_csv()
    ratios = []
    
    for game in dataset:
        # Trova tutti i Pokémon P2 visti durante la battaglia
        all_turns = pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
        unique_pokemon = all_turns['name'].unique()
        
        total_offensive = 0
        total_defensive = 0
        
        for pkmn_name in unique_pokemon:
            # Ottieni stats dal database
            pkmn_info = pkmn_database[pkmn_database['name'] == pkmn_name]
            
            if len(pkmn_info) > 0:
                pkmn_stats = pkmn_info.iloc[0]
                total_offensive += pkmn_stats['base_atk'] + pkmn_stats['base_spa']
                total_defensive += pkmn_stats['base_def'] + pkmn_stats['base_spd']
        
        # Calcola il rapporto, evita divisione per zero
        ratio = total_offensive / total_defensive if total_defensive > 0 else 0
        ratios.append(ratio)
    
    return pd.DataFrame({'p2_offensive_ratio': ratios})


def offensive_ratio_difference(dataset) -> pd.DataFrame: #feature
    """
    Differenza tra i rapporti offensivi/difensivi di P1 e P2.
    Valori positivi indicano che P1 ha un team più offensivo rispetto a P2.
    """
    p1_ratio = p1_offensive_ratio(dataset)
    p2_ratio = p2_offensive_ratio(dataset)
    
    diff = p1_ratio['p1_offensive_ratio'] - p2_ratio['p2_offensive_ratio']
    
    return pd.DataFrame({'offensive_ratio_difference': diff})


def p1_moved_first_count(dataset) -> pd.DataFrame: #feature
    """
    Conta il numero di turni in cui P1 ha attaccato per primo.
    Determina chi muove prima basandosi su:
    1. Priority della mossa (più alta = muove prima)
    2. Se priority è uguale, usa la Speed del Pokémon (considerando i boost)
    
    Ignora turni dove almeno uno dei due fa uno switch (move_details = None).
    """
    pkmn_database = open_pkmn_database_csv()
    first_move_counts = []
    
    for game in dataset:
        p1_first = 0
        
        for turn in game['battle_timeline']:
            p1_move = turn['p1_move_details']
            p2_move = turn['p2_move_details']
            
            # Se entrambi hanno fatto una mossa (non switch)
            if p1_move is not None and p2_move is not None:
                p1_priority = p1_move['priority']
                p2_priority = p2_move['priority']
                
                # Chi ha priority più alta muove prima
                if p1_priority > p2_priority:
                    p1_first += 1
                elif p1_priority == p2_priority:
                    # Se priority uguale, controlla speed
                    p1_name = turn['p1_pokemon_state']['name']
                    p2_name = turn['p2_pokemon_state']['name']
                    
                    # Ottieni base speed
                    p1_info = pkmn_database[pkmn_database['name'] == p1_name]
                    p2_info = pkmn_database[pkmn_database['name'] == p2_name]
                    
                    if len(p1_info) > 0 and len(p2_info) > 0:
                        p1_base_spe = p1_info.iloc[0]['base_spe']
                        p2_base_spe = p2_info.iloc[0]['base_spe']
                        
                        # Considera i boost di speed
                        p1_spe_boost = turn['p1_pokemon_state']['boosts']['spe']
                        p2_spe_boost = turn['p2_pokemon_state']['boosts']['spe']
                        
                        # Applica i boost (ogni stage = 50% in più o in meno)
                        # Semplificazione: +1 = 1.5x, +2 = 2x, -1 = 0.67x, etc.
                        p1_effective_spe = p1_base_spe * (1 + 0.5 * p1_spe_boost) if p1_spe_boost >= 0 else p1_base_spe / (1 + 0.5 * abs(p1_spe_boost))
                        p2_effective_spe = p2_base_spe * (1 + 0.5 * p2_spe_boost) if p2_spe_boost >= 0 else p2_base_spe / (1 + 0.5 * abs(p2_spe_boost))
                        
                        # Considera paralisi (riduce speed del 75% in Gen 1)
                        if turn['p1_pokemon_state']['status'] == 'par':
                            p1_effective_spe *= 0.25
                        if turn['p2_pokemon_state']['status'] == 'par':
                            p2_effective_spe *= 0.25
                        
                        if p1_effective_spe > p2_effective_spe:
                            p1_first += 1
        
        first_move_counts.append(p1_first)
    
    return pd.DataFrame({'p1_moved_first_count': first_move_counts})


def p2_moved_first_count(dataset) -> pd.DataFrame: #feature
    """
    Conta il numero di turni in cui P2 ha attaccato per primo.
    Usa la stessa logica di p1_moved_first_count.--Feature Utility Code------------------------
    # ottieni i coefficienti
    #coefficients = pd.Series(pipeline.named_steps['classifier'].coef_[0], index=train_df.columns[2::])

    # ordina per importanza
    coefficients = coefficients.abs().sort_values(ascending=False)

    #print("Most useful features:")
    print(coefficients)


    print(train_df.corr())
    #print("Best CV score:", grid.best_score_)
    #print("Best params:", grid.best_params_)

    """
    pkmn_database = open_pkmn_database_csv()
    first_move_counts = []
    
    for game in dataset:
        p2_first = 0
        
        for turn in game['battle_timeline']:
            p1_move = turn['p1_move_details']
            p2_move = turn['p2_move_details']
            
            # Se entrambi hanno fatto una mossa (non switch)
            if p1_move is not None and p2_move is not None:
                p1_priority = p1_move['priority']
                p2_priority = p2_move['priority']
                
                # Chi ha priority più alta muove prima
                if p2_priority > p1_priority:
                    p2_first += 1
                elif p1_priority == p2_priority:
                    # Se priority uguale, controlla speed
                    p1_name = turn['p1_pokemon_state']['name']
                    p2_name = turn['p2_pokemon_state']['name']
                    
                    # Ottieni base speed
                    p1_info = pkmn_database[pkmn_database['name'] == p1_name]
                    p2_info = pkmn_database[pkmn_database['name'] == p2_name]
                    
                    if len(p1_info) > 0 and len(p2_info) > 0:
                        p1_base_spe = p1_info.iloc[0]['base_spe']
                        p2_base_spe = p2_info.iloc[0]['base_spe']
                        
                        # Considera i boost di speed
                        p1_spe_boost = turn['p1_pokemon_state']['boosts']['spe']
                        p2_spe_boost = turn['p2_pokemon_state']['boosts']['spe']
                        
                        # Applica i boost
                        p1_effective_spe = p1_base_spe * (1 + 0.5 * p1_spe_boost) if p1_spe_boost >= 0 else p1_base_spe / (1 + 0.5 * abs(p1_spe_boost))
                        p2_effective_spe = p2_base_spe * (1 + 0.5 * p2_spe_boost) if p2_spe_boost >= 0 else p2_base_spe / (1 + 0.5 * abs(p2_spe_boost))
                        
                        # Considera paralisi
                        if turn['p1_pokemon_state']['status'] == 'par':
                            p1_effective_spe *= 0.25
                        if turn['p2_pokemon_state']['status'] == 'par':
                            p2_effective_spe *= 0.25
                        
                        if p2_effective_spe > p1_effective_spe:
                            p2_first += 1
        
        first_move_counts.append(p2_first)
    
    return pd.DataFrame({'p2_moved_first_count': first_move_counts})


def speed_advantage_ratio(dataset) -> pd.DataFrame: #feature
    """
    Rapporto tra il numero di turni in cui P1 muove prima vs P2.
    
    Ratio > 1: P1 ha vantaggio di speed
    Ratio < 1: P2 ha vantaggio di speed
    Ratio = 1: Speed equilibrata
    
    Formula: (p1_moved_first + 1) / (p2_moved_first + 1)
    Aggiungiamo +1 per evitare divisioni per zero.
    """
    p1_first = p1_moved_first_count(dataset)
    p2_first = p2_moved_first_count(dataset)
    
    # Calcola il rapporto con smoothing (+1) per evitare divisione per zero
    ratio = (p1_first['p1_moved_first_count'] + 1) / (p2_first['p2_moved_first_count'] + 1)
    
    return pd.DataFrame({'speed_advantage_ratio': ratio})


if __name__=="__main__":
    dataset=open_train_json()
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
    pd.set_option('display.max_rows',None)
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
    print(pd.concat([p1_reflect_ratio(dataset),p2_reflect_ratio(dataset),p1_lightscreen_ratio(dataset),p2_lightscreen_ratio(dataset)],axis=1))
    print(pd.concat([p1_reflect_ratio(dataset).sum(),p2_reflect_ratio(dataset).sum(),p1_lightscreen_ratio(dataset).sum(),p2_lightscreen_ratio(dataset).sum()],axis=1))
    
   
    #reflect active
    #lightscreen active