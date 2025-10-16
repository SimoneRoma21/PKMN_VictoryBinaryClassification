import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Callable, Union
from enum import Enum

from .feature_data_extract import (
    p1_mean_hp_start,
    p2_mean_hp_start,
    lead_spd,
    mean_spd_start,
    mean_spd_last,
    mean_hp_last,
    p1_alive_pkmn,
    p2_alive_pkmn
)

class Feature(Enum):
    """Enum con tutte le feature disponibili"""

    P1_MEAN_HP_START = "p1_mean_hp_start"
    P2_MEAN_HP_START = "p2_mean_hp_start"
    LEAD_SPD = "lead_spd"
    MEAN_SPD_START = "mean_spd_start"
    MEAN_SPD_LAST = "mean_spd_last"
    MEAN_HP_LAST = "mean_hp_last"
    P1_ALIVE_PKMN = "p1_alive_pkmn"
    P2_ALIVE_PKMN = "p2_alive_pkmn"


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
        self._extractors[Feature.MEAN_SPD_START] = mean_spd_start
        self._extractors[Feature.MEAN_SPD_LAST] = mean_spd_last
        self._extractors[Feature.MEAN_HP_LAST] = mean_hp_last
        self._extractors[Feature.P1_ALIVE_PKMN] = p1_alive_pkmn
        self._extractors[Feature.P2_ALIVE_PKMN] = p2_alive_pkmn

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

