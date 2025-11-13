import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Callable, Union
from enum import Enum
import os

from .feature_data_extract import Feature, FeatureRegistry


class FeaturePipeline:
    """
    Feature extraction pipeline.
    Pass a list of desired features to extract them from battle data.
    
    Features can be cached to disk for faster subsequent loads.
    
    Example:
        Using Feature enums:
       
        pipeline = FeaturePipeline([
            Feature.P1_MEAN_HP,
            Feature.P1_MEAN_ATK,
            Feature.P2_LEAD_SPE
        ])
        df = pipeline.extract_features(data)
       
        
    """
    
    def __init__(self, features: List[Feature], cache_dir: str = "../data/feature_cache"):
        """
        Initialize the feature pipeline.
        
        Args:
            features: List of Feature enums 
            cache_dir: Directory path where cached features will be stored (default: "../data/feature_cache")
        """
        self.registry = FeatureRegistry()
        self.features = features
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    
    def extract_features(self, data: List[Dict], show_progress: bool = True, force_extraction: bool = False) -> pd.DataFrame:
        """
        Extract specified features from battle data.
        
        This method will attempt to load features from cache if available.
        If not cached or force_extraction is True, features will be extracted
        from the raw data and saved to cache.
        
        Args:
            data: List of battle dictionaries
            show_progress: If True, display progress bar during extraction (default: True)
            force_extraction: If True, force re-extraction even if cached files exist (default: False)
            
        Returns:
            DataFrame containing extracted features with columns:
                - battle_id: Unique identifier for each battle
                - player_won: Target variable (if present in data)
                - [feature columns]: One column per requested feature
        """
        # Initialize dataframe with metadata
        df_dict = {
            'battle_id': [battle.get('battle_id') for battle in data]
        }
        
        # Add player_won if present
        if data and 'player_won' in data[0]:
            df_dict['player_won'] = [int(battle['player_won']) for battle in data]
        
        # Extract each requested feature (generates complete columns)
        iterator = tqdm(self.features, desc="Extracting features") if show_progress else self.features
        
        columns = []
        for feature in iterator:
            # Generate cache file name
            cache_file = os.path.join(self.cache_dir, f"{feature.value}.csv")
            
            # Check if cache file exists and extraction is not forced
            if os.path.exists(cache_file) and not force_extraction:
                if show_progress:
                    print(f"Loading {feature.value} from cache...")
                # Load data from CSV
                cached_df = pd.read_csv(cache_file, index_col=0)
                columns.append(cached_df)
            else:
                # Extract the feature
                if show_progress:
                    print(f"Extracting {feature.value}...")
                extractor = self.registry.get_extractor(feature)
                if extractor:
                    # The extractor now generates the entire column
                    feature_df = extractor(data)
                    columns.append(feature_df)
                    
                    # Save to CSV file
                    feature_df.to_csv(cache_file)
                    if show_progress:
                        print(f"Cached {feature.value} to {cache_file}")

        df = pd.concat([pd.DataFrame(df_dict)] + columns, axis=1)
        return df

    def list_selected_features(self):
        """Print the selected features."""
        print("Selected Features:")
        print("-" * 40)
        for f in self.features:
            print(f"  • {f.value}")
        print(f"\nTotal: {len(self.features)} features")


