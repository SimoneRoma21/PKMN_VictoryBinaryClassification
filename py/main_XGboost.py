import json
import pandas as pd
from dataset.dataset_construction import Feature, FeaturePipeline
from dataset.csv_utilities import *
from dataset.extract_utilities import *
from ModelTrainer import ModelTrainer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel


def main():
    #---------------Feature Extraction Code------------------------
    selected_features = [

        #----Feature Base Stats Pokemon----#
        Feature.MEAN_SPE_LAST, #*
        
        Feature.MEAN_HP_LAST, #*
       
        Feature.P1_FINAL_TEAM_HP, #*
        Feature.P2_FINAL_TEAM_HP, #*

        Feature.MEAN_ATK_LAST, #* 
        Feature.MEAN_DEF_LAST, #*
        Feature.MEAN_SPA_LAST, #*
        Feature.MEAN_SPD_LAST, #*
        Feature.MEAN_STATS_LAST, #*
        Feature.MEAN_CRIT, #*

        #---Feature Infos During Battle----#
        Feature.P1_ALIVE_PKMN, #*
        Feature.P2_ALIVE_PKMN, #*
        
        Feature.P1_SWITCHES_COUNT, #*
        Feature.P2_SWITCHES_COUNT, #*
        Feature.P1_AVG_HP_WHEN_SWITCHING, #*
        Feature.P2_AVG_HP_WHEN_SWITCHING, #*
        Feature.P1_MAX_DEBUFF_RECEIVED,
        Feature.P2_MAX_DEBUFF_RECEIVED,
        Feature.P1_AVG_MOVE_POWER, #*
        Feature.P2_AVG_MOVE_POWER, #*
        Feature.AVG_MOVE_POWER_DIFFERENCE, #*
        Feature.P1_OFFENSIVE_RATIO, #*
        Feature.P2_OFFENSIVE_RATIO, #*
        Feature.OFFENSIVE_RATIO_DIFFERENCE, #*
        Feature.P1_MOVED_FIRST_COUNT, #*
        Feature.P2_MOVED_FIRST_COUNT, #*
        Feature.SPEED_ADVANTAGE_RATIO, #*

       
        
        #----Feature Status of Pokemons----#
        Feature.P1_FROZEN_PKMN, #*
        Feature.P2_FROZEN_PKMN, #*
        Feature.P1_PARALIZED_PKMN, #*
        Feature.P2_PARALIZED_PKMN, #*
        Feature.P1_SLEEP_PKMN, #*
        Feature.P2_SLEEP_PKMN, #*
        Feature.P1_POISON_PKMN, #*
        Feature.P2_POISON_PKMN, #* 
        Feature.P1_BURNED_PKMN, #*
        Feature.P2_BURNED_PKMN, #*
        
        #----Feature Pokemon Moves----#
        Feature.P1_PKMN_REFLECT, #*
        Feature.P2_PKMN_REFLECT, #*
        Feature.P1_PKMN_REST, #*
        Feature.P2_PKMN_REST, #*
        Feature.P1_PKMN_EXPLOSION, #*
        Feature.P2_PKMN_EXPLOSION, #*
        Feature.P1_PKMN_THUNDERWAVE, #*
        Feature.P2_PKMN_THUNDERWAVE, #*
        Feature.P1_PKMN_RECOVER, #*
        Feature.P2_PKMN_RECOVER, #*
        Feature.P1_PKMN_TOXIC, #*
        Feature.P2_PKMN_TOXIC, #*
        Feature.P1_PKMN_FIRESPIN, #*
        Feature.P2_PKMN_FIRESPIN, #*
        
        

        #----Feature Weaknesses of Teams / Team Composition----#
        #Feature.WEAKNESS_TEAMS_START, 
        #Feature.WEAKNESS_TEAMS_LAST, 
        #Feature.ADVANTAGE_WEAK_START, 
        #Feature.ADVANTAGE_WEAK_LAST, 
        #Feature.P1_PSY_PKMN,
        #Feature.P2_PSY_PKMN
       
]
    feature_pipeline = FeaturePipeline(selected_features)

    train_file_path = '../data/train.jsonl'
    test_file_path = '../data/test.jsonl'
    train_out_path="predict_csv/train_features_extracted.csv"

    print("Loading training data...")
    train_data = []
    with open(train_file_path, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))

    # Estrai le feature train_set
    print("\nExtracting features from training data...")
    train_df = feature_pipeline.extract_features(train_data)
    print("\nTraining features preview:")
    print(train_df.head())
    # Salva il dataset in un file CSV
    train_df.to_csv(train_out_path, index=False)

    #---------------Model Training and Evaluation Code------------------------
    # Remove row 4877 from the train dataset
    train_df = train_df.drop(index=4877)

    X_train = train_df.drop(['battle_id', 'player_won'], axis=1)
    y_train = train_df['player_won']

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Pipeline with XGBoost
    pipeline = Pipeline([
    ('classifier', XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder = False))
    # ('classifier', XGBClassifier(eval_metric='logloss',random_state=42, colsample_bytree= 0.8, gamma = 0, 
    #                       learning_rate=0.05, max_depth=3, min_child_weight=5, n_estimators=600, reg_alpha=0, reg_lambda=2, subsample=0.8)),
    ])
    # Grid Search for XGBoost
    # param_grid = { # Too many hyperparameters
    # 'classifier__n_estimators': [200, 400, 600],
    # 'classifier__learning_rate': [0.01, 0.05, 0.1],
    # 'classifier__max_depth': [3, 5, 7],
    # 'classifier__min_child_weight': [1, 3, 5],
    # 'classifier__gamma': [0, 0.1, 0.3],
    # 'classifier__subsample': [0.7, 0.8, 1.0],
    # 'classifier__colsample_bytree': [0.7, 0.8, 1.0],
    # 'classifier__reg_alpha': [0, 0.1, 0.5],
    # 'classifier__reg_lambda': [1, 2, 3],
    # }

    param_grid = {
    'classifier__n_estimators': [200,400,600],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [3, 6],
    'classifier__min_child_weight': [1, 5],
    'classifier__gamma': [0, 0.3],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0],
    'classifier__reg_alpha': [0, 0.1],
    'classifier__reg_lambda': [1, 2],
    }

    grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    # scoring='accuracy',
    cv=5, 
    verbose=2,
    n_jobs=-1
    )

    trainer = ModelTrainer(grid)
    # trainer = ModelTrainer(pipeline)
    trainer.train(X_tr, y_tr)
    trainer.evaluate(X_val, y_val)

    #---------------Feature Utility Code GRID------------------------

    # # Get the best model
    # best_model = grid.best_estimator_.named_steps['classifier']

    # # Get the coefficients
    # importances = pd.Series(best_model.feature_importances_, index=train_df.columns[2::])

    # # Sort by importance
    # importances = importances.sort_values(ascending=False)

    # print("Most useful features:")
    # pd.set_option('display.max_rows', None)
    # print(importances)

    # print(train_df.corr())
    # print("Best CV score:", grid.best_score_)
    # print("Best params:", grid.best_params_)

    # ------------------ Evaluate on Test Set -----------------

    evaluate_test_set(trainer, selected_features, test_file_path)

def evaluate_test_set(trainer: ModelTrainer, feature_list: list, test_file_path: str):

    feature_pipeline = FeaturePipeline(feature_list, cache_dir="../data/test_feature_cache")

    print("\nLoading test data...")
    test_data = []
    with open(test_file_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))

    # Estrai le feature del test_set
    print("\nExtracting features from test data...")
    test_df = feature_pipeline.extract_features(test_data, show_progress=True)

    X_test = test_df.drop(['battle_id'], axis=1, errors='ignore')

    # Predici sul test set
    predictions = trainer.predict(X_test)

    submission = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': predictions
    })
    submission.to_csv('predict_csv/predictions_XGBoost.csv', index=False)

if __name__ == "__main__":
    main()

    #Best params: {'classifier__colsample_bytree': 0.8, 'classifier__gamma': 0, 'classifier__learning_rate': 0.05, 'classifier__max_depth': 3, 'classifier__min_child_weight': 5, 'classifier__n_estimators': 600, 'classifier__reg_alpha': 0, 'classifier__reg_lambda': 2, 'classifier__subsample': 0.8}
