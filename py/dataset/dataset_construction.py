import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Callable, Union
from enum import Enum

from .feature_data_extract import Feature, FeatureRegistry



def p1_mean_hp(data: List[Dict]) -> np.ndarray:
    """Calcola la media dei punti salute per il team del player 1 per tutte le battaglie"""
    return np.array([np.mean([mon.get('base_hp', 0) for mon in battle.get('p1_team_details', [])]) 
                     for battle in data])


class FeaturePipeline:
    """
    Pipeline per l'estrazione di feature.
    Basta passare una lista di feature desiderate.
    
    Example:
        pipeline = FeaturePipeline([
            Feature.P1_MEAN_HP,
            Feature.P1_MEAN_ATK,
            Feature.P2_LEAD_SPE
        ])
        df = pipeline.extract_features(data)
        
    O con stringhe:
        pipeline = FeaturePipeline([
            "p1_mean_hp",
            "p1_mean_atk",
            "speed_ratio"
        ])
    """
    
    def __init__(self, features: List[Feature]):
        self.registry = FeatureRegistry()
        self.features = features
    
    
    def extract_features(self, data: List[Dict], show_progress: bool = True) -> pd.DataFrame:
        """
        Estrae le feature specificate dai dati
        
        Args:
            data: Lista di battaglie (dict)
            show_progress: Mostra progress bar
            
        Returns:
            DataFrame con le feature estratte
        """
        # Inizializza il dataframe con i metadati
        df_dict = {
            'battle_id': [battle.get('battle_id') for battle in data]
        }
        
        # Aggiungi player_won se presente
        if data and 'player_won' in data[0]:
            df_dict['player_won'] = [int(battle['player_won']) for battle in data]
        
        # Estrai ogni feature richiesta (genera colonne complete)
        iterator = tqdm(self.features, desc="Extracting features") if show_progress else self.features
        
        columns = []
        for feature in iterator:
            extractor = self.registry.get_extractor(feature)
            if extractor:
                # L'extractor ora genera tutta la colonna
                columns.append(extractor(data))

        df = pd.concat([pd.DataFrame(df_dict)] + columns, axis=1)
        return df

    def list_selected_features(self):
        """Stampa le feature selezionate"""
        print("Selected Features:")
        print("-" * 40)
        for f in self.features:
            print(f"  • {f.value}")
        print(f"\nTotal: {len(self.features)} features")

