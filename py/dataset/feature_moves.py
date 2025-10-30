import pandas as pd
import numpy as np


def p1_pokemon_reflect(dataset)->pd.DataFrame: #feature
    '''
    Count the number of alive pokemon of p1 that have the move reflect
    per each games and return a dataframe of the counts per game
    '''
    p1_count=[]
    for game in dataset:
        #getting information about states and moves of pokemons
        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        #checking if at least one pkmn is alive
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='reflect']
            turns=turns[turns['status']!='fnt']
            p1_count.append(len(turns)) # appending the occurrence
        else:
            p1_count.append(0) #otherwise, no pkmn, occurences are 0 per game g
    return pd.DataFrame({'p1_pkmn_reflect':p1_count})

def p2_pokemon_reflect(dataset)->pd.DataFrame: #feature
    '''
    Count the number of alive pokemon of p2 that have the move reflect
    per each games and return a dataframe of the counts per game
    '''
    p2_count=[]
    for game in dataset:
        #getting information about states and moves of pokemons
        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        #checking if at least one pkmn is alive
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='reflect']
            turns=turns[turns['status']!='fnt']
            p2_count.append(len(turns))# appending the occurrence
        else:
            p2_count.append(0) #otherwise, no pkmn, occurences are 0 per game g
    return pd.DataFrame({'p2_pkmn_reflect':p2_count})

def p1_pokemon_rest(dataset)->pd.DataFrame: #feature
    '''
    Count the number of alive pokemon of p1 that have the move rest
    per each games and return a dataframe of the counts per game
    '''
    p1_count=[]
    for game in dataset:
        #getting information about states and moves of pokemons
        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        #checking if at least one pkmn is alive
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='rest']
            turns=turns[turns['status']!='fnt']
            p1_count.append(len(turns)) # appending the occurrence
        else:
            p1_count.append(0) #otherwise, no pkmn, occurences are 0 per game g
    return pd.DataFrame({'p1_pkmn_rest':p1_count})

def p2_pokemon_rest(dataset)->pd.DataFrame: #feature
    '''
    Count the number of alive pokemon of p2 that have the move rest
    per each games and return a dataframe of the counts per game
    '''
    p2_count=[]
    for game in dataset:
        #getting information about states and moves of pokemons
        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        #checking if at least one pkmn is alive
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='rest']
            turns=turns[turns['status']!='fnt']
            p2_count.append(len(turns)) # appending the occurrence
        else:
            p2_count.append(0) #otherwise, no pkmn, occurences are 0 per game g
    return pd.DataFrame({'p2_pkmn_rest':p2_count})

def p1_pokemon_explosion(dataset)->pd.DataFrame: #feature
    '''
    Count the number of alive pokemon of p1 that have the move explosion and selfdestruct
    per each games and return a dataframe of the counts per game
    '''
    p1_count=[]
    for game in dataset:
        #getting information about states and moves of pokemons
        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        #checking if at least one pkmn is alive
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=pd.concat([turns[turns['name']=='explosion'], turns[turns['name']=='selfdestruct']],axis=0)
            
            p1_count.append(len(turns)) # appending the occurrence
        else:
            p1_count.append(0) #otherwise, no pkmn, occurences are 0 per game g
    return pd.DataFrame({'p1_pkmn_explosions':p1_count})

def p2_pokemon_explosion(dataset)->pd.DataFrame: #feature
    '''
    Count the number of alive pokemon of p2 that have the move explosion and selfdestruct
    per each games and return a dataframe of the counts per game
    '''
    p2_count=[]
    for game in dataset:
        #getting information about states and moves of pokemons
        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        #checking if at least one pkmn is alive
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=pd.concat([turns[turns['name']=='explosion'], turns[turns['name']=='selfdestruct']],axis=0)
            
            p2_count.append(len(turns))# appending the occurrence
        else:
            p2_count.append(0)#otherwise, no pkmn, occurences are 0 per game g
    return pd.DataFrame({'p2_pkmn_explosions':p2_count})

def p1_pokemon_thunderwave(dataset)->pd.DataFrame: #feature
    '''
    Count the number of alive pokemon of p1 that have the move thunderwave
    per each games and return a dataframe of the counts per game
    '''
    p1_count=[]
    for game in dataset:
        # getting information about states and moves of pokemons
        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        # checking if at least one pkmn is alive
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='thunderwave']
            turns=turns[turns['status']!='fnt']
            p1_count.append(len(turns)) # appending the occurrence
        else:
            p1_count.append(0) # otherwise, no pkmn, occurences are 0 per game g
    return pd.DataFrame({'p1_pkmn_thunderwave':p1_count})

def p2_pokemon_thunderwave(dataset)->pd.DataFrame: #feature
    '''
    Count the number of alive pokemon of p2 that have the move thunderwave
    per each games and return a dataframe of the counts per game
    '''
    p2_count=[]
    for game in dataset:
        # getting information about states and moves of pokemons
        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        # checking if at least one pkmn is alive
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='thunderwave']
            turns=turns[turns['status']!='fnt']
            p2_count.append(len(turns)) # appending the occurrence
        else:
            p2_count.append(0) # otherwise, no pkmn, occurences are 0 per game g
    return pd.DataFrame({'p2_pkmn_thunderwave':p2_count})

def p1_pokemon_recover(dataset)->pd.DataFrame: #feature
    '''
    Count the number of alive pokemon of p1 that have the move recover
    per each games and return a dataframe of the counts per game
    '''
    p1_count=[]
    for game in dataset:
        # getting information about states and moves of pokemons
        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        # checking if at least one pkmn is alive
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='recover']
            turns=turns[turns['status']!='fnt']
            p1_count.append(len(turns)) # appending the occurrence
        else:
            p1_count.append(0) # otherwise, no pkmn, occurences are 0 per game g
    return pd.DataFrame({'p1_pkmn_recover':p1_count})

def p2_pokemon_recover(dataset)->pd.DataFrame: #feature
    '''
    Count the number of alive pokemon of p2 that have the move recover
    per each games and return a dataframe of the counts per game
    '''
    p2_count=[]
    for game in dataset:
        # getting information about states and moves of pokemons
        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        # checking if at least one pkmn is alive
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='recover']
            turns=turns[turns['status']!='fnt']
            p2_count.append(len(turns)) # appending the occurrence
        else:
            p2_count.append(0) # otherwise, no pkmn, occurences are 0 per game g
    return pd.DataFrame({'p2_pkmn_recover':p2_count})

def p1_pokemon_toxic(dataset)->pd.DataFrame: #feature
    '''
    Count the number of alive pokemon of p1 that have the move toxic
    per each games and return a dataframe of the counts per game
    '''
    p1_count=[]
    for game in dataset:
        # getting information about states and moves of pokemons
        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        # checking if at least one pkmn is alive
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='toxic']
            turns=turns[turns['status']!='fnt']
            p1_count.append(len(turns)) # appending the occurrence
        else:
            p1_count.append(0) # otherwise, no pkmn, occurences are 0 per game g
    return pd.DataFrame({'p1_pkmn_toxic':p1_count})

def p2_pokemon_toxic(dataset)->pd.DataFrame: #feature
    '''
    Count the number of alive pokemon of p2 that have the move toxic
    per each games and return a dataframe of the counts per game
    '''
    p2_count=[]
    for game in dataset:
        # getting information about states and moves of pokemons
        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        # checking if at least one pkmn is alive
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='toxic']
            turns=turns[turns['status']!='fnt']
            p2_count.append(len(turns)) # appending the occurrence
        else:
            p2_count.append(0) # otherwise, no pkmn, occurences are 0 per game g
    return pd.DataFrame({'p2_pkmn_toxic':p2_count})

def p1_pokemon_firespin(dataset)->pd.DataFrame: #feature
    '''
    Count the number of alive pokemon of p1 that have the move firespin
    per each games and return a dataframe of the counts per game
    '''
    p1_count=[]
    for game in dataset:
        # getting information about states and moves of pokemons
        turns_p1_state=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p1_moves=pd.DataFrame([turn['p1_move_details'] if turn['p1_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        # checking if at least one pkmn is alive
        if len(turns_p1_state)!=0 and len(turns_p1_moves)!=0:
            turns=pd.concat([turns_p1_state,turns_p1_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='firespin']
            turns=turns[turns['status']!='fnt']
            p1_count.append(len(turns)) # appending the occurrence
        else:
            p1_count.append(0) # otherwise, no pkmn, occurences are 0 per game g
    return pd.DataFrame({'p1_pkmn_firespin':p1_count})

def p2_pokemon_firespin(dataset)->pd.DataFrame: #feature
    '''
    Count the number of alive pokemon of p2 that have the move firespin
    per each games and return a dataframe of the counts per game
    '''
    p2_count=[]
    for game in dataset:
        # getting information about states and moves of pokemons
        turns_p2_state=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])[['name','status']].rename(columns={'name':'pkmn_name'})
        turns_p2_moves=pd.DataFrame([turn['p2_move_details'] if turn['p2_move_details']!=None else {'name':'None'} for turn in game['battle_timeline']])['name']
        # checking if at least one pkmn is alive
        if len(turns_p2_state)!=0 and len(turns_p2_moves)!=0:
            turns=pd.concat([turns_p2_state,turns_p2_moves],axis=1).drop_duplicates(subset='pkmn_name',keep='last')
            turns=turns[turns['name']=='firespin']
            turns=turns[turns['status']!='fnt']
            p2_count.append(len(turns)) # appending the occurrence
        else:
            p2_count.append(0) # otherwise, no pkmn, occurences are 0 per game g
    return pd.DataFrame({'p2_pkmn_firespin':p2_count})

def p1_reflect_ratio(dataset)->pd.DataFrame: #feature
    '''
    Counts the ratio of turns in which p1 had the reflect effect active.
    The ratio is calcuted as n_turns_reflect/30. Results are returned
    as a dataframe.
    '''
    p1_count=[]
    for game in dataset:
        #getting all turns of p1
        p1_timeline=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])
        #counting turns of reflect
        p1_timeline['n_reflects']=p1_timeline['effects'].apply(lambda x: x[0].count('reflect'))
        #appending ratio
        p1_count.append(np.sum(p1_timeline['n_reflects'])/30)
    return pd.DataFrame({'p1_reflect_ratio':p1_count})

def p2_reflect_ratio(dataset)->pd.DataFrame: #feature
    '''
    Counts the ratio of turns in which p2 had the reflect effect active.
    The ratio is calcuted as n_turns_reflect/30. Results are returned
    as a dataframe.
    '''
    p2_count=[]
    for game in dataset:
        #getting all turns of p2
        p2_timeline=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
        #counting turns of reflect
        p2_timeline['n_reflects']=p2_timeline['effects'].apply(lambda x: x[0].count('reflect'))
        #appending ratio
        p2_count.append(np.sum(p2_timeline['n_reflects'])/30)
    return pd.DataFrame({'p2_reflect_ratio':p2_count})

def p1_lightscreen_ratio(dataset)->pd.DataFrame: #feature
    '''
    Counts the ratio of turns in which p1 had the lightscreen active.
    The ratio is calcuted as n_turns_lightscreen/30. Results are returned
    as a dataframe.
    '''
    p1_count=[]
    for game in dataset:
        #getting all turns of p1
        p1_timeline=pd.DataFrame([turn['p1_pokemon_state'] for turn in game['battle_timeline']])
        #counting turns of lightscreen
        p1_timeline['n_lightscreens']=p1_timeline['effects'].apply(lambda x: x[0].count('lightscreen'))
        #appending ratio
        p1_count.append(np.sum(p1_timeline['n_lightscreens'])/30)
    return pd.DataFrame({'p1_lightscreens_ratio':p1_count})

def p2_lightscreen_ratio(dataset)->pd.DataFrame: #feature
    '''
    Counts the ratio of turns in which p2 had the lightscreen active.
    The ratio is calcuted as n_turns_lightscreen/30. Results are returned
    as a dataframe.
    '''
    p2_count=[]
    for game in dataset:
        #getting all turns of p2
        p2_timeline=pd.DataFrame([turn['p2_pokemon_state'] for turn in game['battle_timeline']])
        #counting turns of lightscreen
        p2_timeline['n_lightscreens']=p2_timeline['effects'].apply(lambda x: x[0].count('lightscreen'))
        #appending ratio
        p2_count.append(np.sum(p2_timeline['n_lightscreens'])/30)
    return pd.DataFrame({'p2_lightscreens_ratio':p2_count})
