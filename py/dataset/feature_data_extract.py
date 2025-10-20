import pandas as pd
import json
import numpy as np
from typing import List, Dict, Callable
from enum import Enum

class Feature(Enum):
    """Enum con tutte le feature disponibili"""

    P1_MEAN_HP_START = "p1_mean_hp_start"
    P2_MEAN_HP_START = "p2_mean_hp_start"
    LEAD_SPD = "lead_spd"
    MEAN_SPE_START = "mean_spe_start"
    MEAN_SPE_LAST = "mean_spe_last"
    MEAN_HP_LAST = "mean_hp_last"
    MEAN_STATS_START= "mean_stats_start"
    P1_ALIVE_PKMN = "p1_alive_pkmn"
    P2_ALIVE_PKMN = "p2_alive_pkmn"
    WEAKNESS_TEAMS= "weakness_teams"
    
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
    P1_MAX_DEBUFF_RECEIVED = "p1_max_debuff_received" #non sicuro se necessaria
    P2_MAX_DEBUFF_RECEIVED = "p2_max_debuff_received" #non sicuro se necessaria
    

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
        self._extractors[Feature.LEAD_SPD] = lead_spd
        self._extractors[Feature.MEAN_SPE_START] = mean_spe_start
        self._extractors[Feature.MEAN_SPE_LAST] = mean_spe_last
        self._extractors[Feature.MEAN_HP_LAST] = mean_hp_last
        self._extractors[Feature.MEAN_STATS_START] = mean_stats_start
        self._extractors[Feature.P1_ALIVE_PKMN] = p1_alive_pkmn
        self._extractors[Feature.P2_ALIVE_PKMN] = p2_alive_pkmn
        self._extractors[Feature.WEAKNESS_TEAMS] = weakness_teams
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
        self._extractors[Feature.P1_MAX_DEBUFF_RECEIVED] = p1_max_debuff_received
        self._extractors[Feature.P2_MAX_DEBUFF_RECEIVED] = p2_max_debuff_received
        


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

def extract_p1_team_from_game_start_with_boosts(game)-> pd.DataFrame:
    turns=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])
    pkmn_p1_start=turns.drop_duplicates(subset='name',keep='last')[['name','boosts']]

    return pkmn_p1_start

def extract_p1_team_from_game_last_with_boosts(game) -> pd.Series:
    turns=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])
    pkmn_dead_p1=turns[turns['status']=='fnt']['name'].drop_duplicates(keep='last')
    
    team_start_p1=extract_p1_team_from_game_start_with_boosts(game)
    team_remain_p1=team_start_p1[~team_start_p1.isin(pkmn_dead_p1)]
    
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

def extract_p2_team_from_game_start_with_boosts(game)-> pd.DataFrame:
    turns=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
    pkmn_p2_start=turns.drop_duplicates(subset='name',keep='last')[['name','boosts']]

    return pkmn_p2_start

def extract_p2_team_from_game_last_with_boosts(game) -> pd.Series:
    turns=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
    pkmn_dead_p2=turns[turns['status']=='fnt']['name'].drop_duplicates(keep='last')
    
    team_start_p2=extract_p2_team_from_game_start_with_boosts(game)
    team_remain_p2=team_start_p2[~team_start_p2.isin(pkmn_dead_p2)]
    
    return team_remain_p2

def mean_spe_database(pkmn_database) -> float:
    return np.mean(pkmn_database['base_spe'])

def mean_spe_start(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()

    p1_mean_spe=[]
    p2_mean_spe=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_start(game)
        p2_team=extract_p2_team_from_game_start(game)

        p1_team=pkmn_database[pkmn_database['name'].isin(p1_team)]
        p1_known=len(p1_team)
        p1_team=p1_team[['name','base_spe']]
        p1_mean_spe.append((np.sum(p1_team['base_spe'])+ (mean_spe_database(pkmn_database)*(6-p1_known)))/6)
    
        p2_team=pkmn_database[pkmn_database['name'].isin(p2_team)]
        p2_known=len(p2_team)
        p2_team=p2_team[['name','base_spe']]
        p2_mean_spe.append((np.sum(p2_team['base_spe'])+ mean_spe_database(pkmn_database)*(6-p2_known))/6)

    mean_spe_start=pd.DataFrame({'p1_mean_spe_start':p1_mean_spe,'p2_mean_spe_start':p2_mean_spe})
    return mean_spe_start

def mean_spe_last(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()
    
    p1_mean_spe=[]
    p2_mean_spe=[]

    multipliers={-6: 2/8, -5: 2/7, -4: 2/6, -3: 2/5, -2:2/4, -1: 2/3, 0:1, +1: 3/2, 2: 4/2, 3: 5/2, 4: 6/2, 5: 7/2, 6: 8/2 }
    
    for game in dataset:
        p1_team=extract_p1_team_from_game_last_with_boosts(game)
        p2_team=extract_p2_team_from_game_last_with_boosts(game)

        p1_team=p1_team.merge(pkmn_database, how='inner', on='name')
        p1_known=len(extract_p1_team_from_game_start(game))
        p1_team=p1_team[['name','base_spe','boosts']]
        if(len(p1_team)!=0):
            p1_mean_spe.append((np.sum(p1_team['base_spe']*multipliers[p1_team['boosts'][0]['spe']])+ mean_spe_database(pkmn_database)*(6-p1_known))/6)
             #p1_mean_spd.append(np.mean(p1_team['base_spe']))
        else:
            p1_mean_spe.append(0)
       
    
        p2_team=p2_team.merge(pkmn_database, how='inner', on='name')
        p2_known=len(extract_p2_team_from_game_start(game))
        p2_team=p2_team[['name','base_spe','boosts']]
        if(len(p2_team)!=0):
            p2_mean_spe.append((np.sum(p2_team['base_spe']*multipliers[p2_team['boosts'][0]['spe']])+ mean_spe_database(pkmn_database)*(6-p2_known))/6)
            #p2_mean_spd.append(np.mean(p2_team['base_spe']))
        else:
            p2_mean_spe.append(0)


    mean_spe_last=pd.DataFrame({'p1_mean_spe_last':p1_mean_spe,'p2_mean_spe_last':p2_mean_spe})
    mean_spe_last=mean_spe_last.fillna(value=0)
    return mean_spe_last 

def mean_hp_database(pkmn_database) -> float:
    return np.mean(pkmn_database['base_hp'])

def p1_mean_hp_start(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()

    p1_mean_hp=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_start(game)
        p1_known=len(p1_team)    
        p1_team=pkmn_database[pkmn_database['name'].isin(p1_team)]
        p1_team=p1_team[['name','base_hp']]
        p1_mean_hp.append((np.sum(p1_team['base_hp'])+ mean_hp_database(pkmn_database)*(6-p1_known))/6)

    mean_hp_start=pd.DataFrame({'p1_mean_hp_start':p1_mean_hp})
    return mean_hp_start

def p2_mean_hp_start(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()

    p2_mean_hp=[]
    for game in dataset:
        p2_team=extract_p2_team_from_game_start(game)
        p2_team=pkmn_database[pkmn_database['name'].isin(p2_team)]
        p2_team=p2_team[['name','base_hp']]
        p2_mean_hp.append((np.sum(p2_team['base_hp'])+ mean_hp_database(pkmn_database)*(6-len(p2_team)))/6)
    mean_hp_start=pd.DataFrame({'p2_mean_hp_start':p2_mean_hp})
    return mean_hp_start

def mean_hp_last(dataset):
    pkmn_database = open_pkmn_database_csv()
    p1_mean_hp=[]
    p2_mean_hp=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_last(game)
        p2_team=extract_p2_team_from_game_last(game)


        p1_team=pkmn_database[pkmn_database['name'].isin(p1_team)]
        p1_known=len(extract_p1_team_from_game_start(game))
        p1_team=p1_team[['name','base_hp']]
        p1_mean_hp.append((np.sum(p1_team['base_hp'])+ mean_hp_database(pkmn_database)*(6-p1_known))/6)
    
        p2_team=pkmn_database[pkmn_database['name'].isin(p2_team)]
        p2_team=p2_team[['name','base_hp']]
        p2_known=len(extract_p2_team_from_game_start(game))
        p2_mean_hp.append((np.sum(p2_team['base_hp'])+ mean_hp_database(pkmn_database)*(6-p2_known))/6)

    mean_hp_last=pd.DataFrame({'p1_mean_hp_last':p1_mean_hp,'p2_mean_hp_last':p2_mean_hp})
    mean_hp_last=mean_hp_last.fillna(value=0)
    return mean_hp_last

def mean_total_database(pkmn_database) -> float:
    pkmn_database['total']=np.sum(pkmn_database[['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']],axis=1)
    return np.mean(pkmn_database['total'])

def mean_stats_start(dataset) -> pd.DataFrame: #feature
    pkmn_database = open_pkmn_database_csv()

    p1_mean_stats=[]
    p2_mean_stats=[]
    for game in dataset:
        p1_team=extract_p1_team_from_game_start(game).to_frame()
        p2_team=extract_p2_team_from_game_start(game).to_frame()

        p1_team=p1_team.merge(pkmn_database,how='inner',on='name')
        p1_known=len(p1_team)
        p1_team['total']=np.sum(p1_team[['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']],axis=1)
        p1_team=p1_team[['name','total']]
        p1_mean_stats.append((np.sum(p1_team['total'])+ (mean_total_database(pkmn_database)*(6-p1_known)))/6)
    
        p2_team=p2_team.merge(pkmn_database,how='inner',on='name')
        p2_known=len(p2_team)
        p2_team['total']=np.sum(p2_team[['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']],axis=1)
        p2_team=p2_team[['name','total']]
        p2_mean_stats.append((np.sum(p2_team['total'])+ mean_total_database(pkmn_database)*(6-p2_known))/6)

    mean_stats=pd.DataFrame({'p1_mean_stats':p1_mean_stats,'p2_mean_stats':p2_mean_stats})
    return mean_stats

def p1_alive_pkmn(dataset)->pd.Series: #feature
    pkmn_alive_p1=[]
    for game in dataset:
        turns=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])
        pkmn_dead_p1=turns[turns['status']=='fnt']['name'].drop_duplicates(keep='last')
        pkmn_alive_p1.append(6-len(pkmn_dead_p1))
    pkmn_alive_p1=pd.DataFrame(pkmn_alive_p1).rename(columns={0:'p1_pkmn_alive'})
    return pkmn_alive_p1


def p2_alive_pkmn(dataset)->pd.Series: #feature
    pkmn_alive_p2=[]
    for game in dataset:
        turns=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
        pkmn_dead_p2=turns[turns['status']=='fnt']['name'].drop_duplicates(keep='last')
        pkmn_alive_p2.append(6-len(pkmn_dead_p2))
    pkmn_alive_p2=pd.DataFrame(pkmn_alive_p2).rename(columns={0:'p2_pkmn_alive'})
    return pkmn_alive_p2
   

def all_pokemon_round(player: int,json):
    if player==1:
        return set([elem['p1_pokemon_state']['name'] for elem in json['battle_timeline']])
    elif player==2:
        return set([elem['p2_pokemon_state']['name'] for elem in json['battle_timeline']])




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
    Se P1 non perde nessun Pokémon, restituisce il numero totale di turni.
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
            faint_turn = len(game['battle_timeline']) + 1
        
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

def weakness_teams_not_opt(dataset) -> pd.DataFrame: #feature
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
        #if game['battle_id']==0:
            #weakness_p1=[]
            p1_team_types=extract_types_from_team_p1(game)
            p1_team_types=p1_team_types.merge(pkmn_db_weak,how='inner',on='name')
            #print(p1_team_types)
            #print(p1_team_types,"\n")
            sw_1=set(sum(p1_team_types['weaknesses'],[]))
            #print(sw_1)
            
            #weakness_p1.append(len(sw_1))
            weak_games_p1.append(len(sw_1))

            #weakness_p2=[]
            p2_team_types=extract_types_from_team_p2(game)
            p2_team_types=p2_team_types.merge(pkmn_db_weak,how='inner',on='name')
            #print(p2_team_types,"\n")
            #print(p2_team_types)
            sw_2=set(sum(p2_team_types['weaknesses'],[]))
            #print(sw_2)
            #weakness_p2.append(len(sw_2))
            weak_games_p2.append(len(sw_2))
            
            

    return pd.DataFrame({'weakness_start_p1':weak_games_p1,'weakness_start_p2':weak_games_p2})
        

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
    print(weakness_teams(dataset))