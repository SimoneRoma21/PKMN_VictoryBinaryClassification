import pandas as pd
import json
import numpy as np


def open_train_json():
    list = []
    with open("../../data/train.jsonl", "r") as f:
        for line in f:
            list.append(json.loads(line))
    return list

def open_pkmn_database_csv():
    #opening pkmn database csv
    pkmn_db=pd.read_csv("../../data/pkmn_database.csv")
    pkmn_db=pkmn_db.drop("Unnamed: 0",axis=1)
    return pkmn_db
    
def extract_all_pokemon_p1_teams(dataset):
    #extracting all p1 teams
    db_pkmn_p1= pd.DataFrame([team for game in dataset for team in game['p1_team_details']]) 
    db_pkmn_p1.drop_duplicates(subset=['name'],inplace=True)
    return db_pkmn_p1

def extract_all_pokemon_p2_seen(dataset):
    #extracting all p2 seens pokemons
    db_pkmn_p2_battles=pd.DataFrame([elem['p2_pokemon_state']['name'] for game in dataset for elem in game['battle_timeline']])
    db_pkmn_p2_battles.drop_duplicates(inplace=True)
    db_pkmn_p2_battles.rename(columns={0:'name'},inplace=True)
    return db_pkmn_p2_battles

def extract_all_pokemon_p2_lead(dataset,duplicates):
    # getting all p2 leads
    db_pkmn_p2_lead=pd.DataFrame([game['p2_lead_details'] for game in dataset ])
    if not(duplicates): # admitting duplicates or not
        db_pkmn_p2_lead.drop_duplicates(subset=['name'],inplace=True)
    return db_pkmn_p2_lead

def extract_all_pokemon_p2(dataset):
 
    # picking all pokemons seen in all battles of p2 
    db_pkmn_p2_battles=extract_all_pokemon_p2_seen(dataset)
    # picking all pokemon leads of p2 (has to be subset of db_pkmn_p2_battles)
    db_pkmn_p2_lead=extract_all_pokemon_p2_lead(dataset,False)
    # merging the two dataset
    db_pkmn_p2=db_pkmn_p2_lead.merge(db_pkmn_p2_battles, how='inner', on=['name'])

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
    pd.DataFrame.to_csv(db_pkmn,"../../data/pkmn_database.csv")


def moves_database():
    pass


def extract_lead_velocity(dataset): # feature

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

    
def extract_p1_team_from_game_start(game):
    return pd.DataFrame(game['p1_team_details'])

def extract_p1_team_from_game_last(game):
    turns=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])
    pkmn_dead_p1=turns[turns['status']=='fnt']['name']
    
    team_start_p1=extract_p1_team_from_game_start(game)
    team_remain_p1=team_start_p1[~team_start_p1['name'].isin(pkmn_dead_p1)]
    
    return team_remain_p1
    
def extract_hp_adv(dict):
    pkmn_p1_dict=dict['p1_team_details']
    pkmn_p2=[]
    
    pkmn_p1=[(pkmn['name'],pkmn['level'],pkmn['base_hp']) for pkmn in pkmn_p1_dict]
    ##print(pkmn_p1)
    pass



def all_pokemon_round(player: int,json):
    if player==1:
        return set([elem['p1_pokemon_state']['name'] for elem in json['battle_timeline']])
    elif player==2:
        return set([elem['p2_pokemon_state']['name'] for elem in json['battle_timeline']])




if __name__=="__main__":
    dataset=open_train_json()
    #print(dataset[0]['battle_timeline'])
    #extract_lead_velocity(dataset)
    #print(pkmn_database(dataset))
    #print(extract_all_pokemon_p2_seen(dataset))
    #pkmn_database(dataset)

    #open_pkmn_database_csv()

    extract_p1_team_from_game_last(dataset[0])
    