import pandas as pd

from . import extract_utilities as ext_u

def p1_frozen_pkmn(dataset)-> pd.DataFrame: #feature
    '''
    Counts the number of alive pokemon of p1 that are currently frozen
    per each game and return a dataframe of the counts per game
    '''
    p1_count=[]
    for game in dataset:
        # extracting p1 team
        p1_team=ext_u.extract_p1_team_from_game_last_with_stats(game)
        
        # appending the count if some pokemon is alive in the team
        # 0 otherwise
        if len(p1_team)!=0:
            p1_count.append(len(p1_team[p1_team['status']=='frz']))
        else:
            p1_count.append(0)
        
    return pd.DataFrame({'p1_frozen_pkmn':p1_count})

def p2_frozen_pkmn(dataset)-> pd.DataFrame: #feature
    '''
    Counts the number of alive pokemon of p2 that are currently frozen
    per each game and return a dataframe of the counts per game
    '''
    p2_count=[]
    for game in dataset:
        # extracting p2 team
        p2_team=ext_u.extract_p2_team_from_game_last_with_stats(game)
        
        # appending the count if some pokemon is alive in the team
        # 0 otherwise
        if len(p2_team)!=0:
            p2_count.append(len(p2_team[p2_team['status']=='frz']))
        else:
            p2_count.append(0)
        
    return pd.DataFrame({'p2_frozen_pkmn':p2_count})
    
def p1_paralized_pkmn(dataset)-> pd.DataFrame: #feature
    '''
    Counts the number of alive pokemon of p1 that are currently paralized
    per each game and return a dataframe of the counts per game
    '''
    p1_count=[]
    for game in dataset:
        # extracting p1 team
        p1_team=ext_u.extract_p1_team_from_game_last_with_stats(game)
        
        # appending the count if some pokemon is alive in the team
        # 0 otherwise
        if len(p1_team)!=0:
            p1_count.append(len(p1_team[p1_team['status']=='par']))
        else:
            p1_count.append(0)
        
    return pd.DataFrame({'p1_paralized_pkmn':p1_count})

def p2_paralized_pkmn(dataset)-> pd.DataFrame: #feature
    '''
    Counts the number of alive pokemon of p2 that are currently paralized
    per each game and return a dataframe of the counts per game
    '''
    p2_count=[]
    for game in dataset:
        # extracting p2 team
        p2_team=ext_u.extract_p2_team_from_game_last_with_stats(game)
        
        # appending the count if some pokemon is alive in the team
        # 0 otherwise
        if len(p2_team)!=0:
            p2_count.append(len(p2_team[p2_team['status']=='par']))
        else:
            p2_count.append(0)
        
    return pd.DataFrame({'p2_paralized_pkmn':p2_count})
    
def p1_sleep_pkmn(dataset)-> pd.DataFrame: #feature
    '''
    Counts the number of alive pokemon of p1 that are currently asleep
    per each game and return a dataframe of the counts per game
    '''
    p1_count=[]
    for game in dataset:
        # extracting p1 team
        p1_team=ext_u.extract_p1_team_from_game_last_with_stats(game)
        
        # appending the count if some pokemon is alive in the team
        # 0 otherwise
        if len(p1_team)!=0:
            p1_count.append(len(p1_team[p1_team['status']=='slp']))
        else:
            p1_count.append(0)
        
    return pd.DataFrame({'p1_sleep_pkmn':p1_count})

def p2_sleep_pkmn(dataset)-> pd.DataFrame: #feature
    '''
    Counts the number of alive pokemon of p2 that are currently asleep
    per each game and return a dataframe of the counts per game
    '''
    p2_count=[]
    for game in dataset:
        # extracting p2 team
        p2_team=ext_u.extract_p2_team_from_game_last_with_stats(game)
        
        # appending the count if some pokemon is alive in the team
        # 0 otherwise
        if len(p2_team)!=0:
            p2_count.append(len(p2_team[p2_team['status']=='slp']))
        else:
            p2_count.append(0)
        
    return pd.DataFrame({'p2_sleep_pkmn':p2_count})

def p1_poison_pkmn(dataset)-> pd.DataFrame: #feature
    '''
    Counts the number of alive pokemon of p1 that are currently poisoned
    per each game and return a dataframe of the counts per game
    '''
    p1_count=[]
    for game in dataset:
        # extracting p1 team
        p1_team=ext_u.extract_p1_team_from_game_last_with_stats(game)
        
        # appending the count if some pokemon is alive in the team
        # 0 otherwise
        if len(p1_team)!=0:
            p1_count.append(len(p1_team[p1_team['status']=='psn']))
        else:
            p1_count.append(0)
        
    return pd.DataFrame({'p1_poison_pkmn':p1_count})
 
def p2_poison_pkmn(dataset)-> pd.DataFrame: #feature
    '''
    Counts the number of alive pokemon of p2 that are currently poisoned
    per each game and return a dataframe of the counts per game
    '''
    p2_count=[]
    for game in dataset:
        # extracting p2 team
        p2_team=ext_u.extract_p2_team_from_game_last_with_stats(game)
        
        # appending the count if some pokemon is alive in the team
        # 0 otherwise
        if len(p2_team)!=0:
            p2_count.append(len(p2_team[p2_team['status']=='psn']))
        else:
            p2_count.append(0)
        
    return pd.DataFrame({'p2_poison_pkmn':p2_count})
    
def p1_burned_pkmn(dataset)-> pd.DataFrame: #feature
    '''
    Counts the number of alive pokemon of p1 that are currently burned
    per each game and return a dataframe of the counts per game
    '''
    p1_count=[]
    for game in dataset:
        # extracting p1 team
        p1_team=ext_u.extract_p1_team_from_game_last_with_stats(game)
        
        # appending the count if some pokemon is alive in the team
        # 0 otherwise
        if len(p1_team)!=0:
            p1_count.append(len(p1_team[p1_team['status']=='brn']))
        else:
            p1_count.append(0)
        
    return pd.DataFrame({'p1_burned_pkmn':p1_count})

def p2_burned_pkmn(dataset)-> pd.DataFrame: #feature
    '''
    Counts the number of alive pokemon of p2 that are currently burned
    per each game and return a dataframe of the counts per game
    '''
    p2_count=[]
    for game in dataset:
        # extracting p2 team
        p2_team=ext_u.extract_p2_team_from_game_last_with_stats(game)

        # appending the count if some pokemon is alive in the team
        # 0 otherwise
        if len(p2_team)!=0:
            p2_count.append(len(p2_team[p2_team['status']=='brn']))
        else:
            p2_count.append(0)
        
    return pd.DataFrame({'p2_burned_pkmn':p2_count})
