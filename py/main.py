import json
import pandas as pd
from dataset.dataset_construction import Feature, FeaturePipeline
from dataset.csv_utilities import *
from dataset.extract_utilities import *
from ModelTrainer import ModelTrainer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    #---------------Feature Extraction Code------------------------
    selected_features = [

        #----Feature Base Stats Pokemon----#
        #Feature.P1_MEAN_HP_START, #*
        #Feature.P2_MEAN_HP_START, 
        #Feature.MEAN_HP_DIFFERENCE_START,
        #Feature.LEAD_SPD,
        #Feature.MEAN_SPE_START,  
        #Feature.MEAN_ATK_START,  
        #Feature.MEAN_DEF_START,  
        #Feature.MEAN_SPA_START,  
        #Feature.MEAN_SPD_START,  
        #Feature.P1_MEAN_SPE_START,
        #Feature.P2_MEAN_SPE_START,
        #Feature.MEAN_SPE_DIFFERENCE_START,
        #Feature.MEAN_STATS_START, 
        #Feature.MEAN_SPE_LAST, #*
        #Feature.P1_MEAN_SPE_LAST,
        #Feature.P2_MEAN_SPE_LAST,
        #Feature.MEAN_SPE_DIFFERENCE_LAST,
        #Feature.MEAN_HP_LAST, #*
        #Feature.P1_MEAN_HP_LAST,
        #Feature.P2_MEAN_HP_LAST,
        #Feature.MEAN_HP_DIFFERENCE_LAST,
        Feature.P1_FINAL_TEAM_HP, #*
        Feature.P2_FINAL_TEAM_HP, #*
        #Feature.FINAL_TEAM_HP_DIFFERENCE, #*
        Feature.MEAN_ATK_LAST, #* 
        #Feature.MEAN_DEF_LAST, #*
        Feature.MEAN_SPA_LAST, #*
        #Feature.MEAN_SPD_LAST, #*
        Feature.MEAN_STATS_LAST, #*
        Feature.MEAN_CRIT, #*
        
        # Sum versions of mean_*_last features
        #Feature.SUM_HP_LAST,
        #Feature.P1_SUM_HP_LAST,
        #Feature.P2_SUM_HP_LAST,
        #Feature.SUM_SPE_LAST,
        #Feature.P1_SUM_SPE_LAST,
        #Feature.P2_SUM_SPE_LAST,
        #Feature.SUM_ATK_LAST,
        #Feature.P1_SUM_ATK_LAST,
        #Feature.P2_SUM_ATK_LAST,
        #Feature.SUM_DEF_LAST,
        #Feature.P1_SUM_DEF_LAST,
        #Feature.P2_SUM_DEF_LAST,
        #Feature.SUM_SPA_LAST,
        #Feature.P1_SUM_SPA_LAST,
        #Feature.P2_SUM_SPA_LAST,
        #Feature.SUM_SPD_LAST,
        #Feature.P1_SUM_SPD_LAST,
        #Feature.P2_SUM_SPD_LAST,
        #Feature.SUM_STATS_LAST,
        #Feature.P1_SUM_STATS_LAST,
        #Feature.P2_SUM_STATS_LAST,

        #---Feature Infos During Battle----#
        #Feature.P1_ALIVE_PKMN, #*
        #Feature.P2_ALIVE_PKMN, #*
        #Feature.ALIVE_PKMN_DIFFERENCE, #*
        #Feature.P1_PKMN_STAB, 
        #Feature.P2_PKMN_STAB, 
        Feature.P1_SWITCHES_COUNT, #*
        Feature.P2_SWITCHES_COUNT, #*
        #Feature.SWITCHES_DIFFERENCE, #*
        #Feature.P1_STATUS_INFLICTED, 
        #Feature.P2_STATUS_INFLICTED, 
        #Feature.STATUS_INFLICTED_DIFFERENCE, 
        
        #Feature.P1_FIRST_FAINT_TURN,
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
        Feature.SPEED_ADVANTAGE_RATIO, #
       
        
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
        #Feature.P1_REFLECT_RATIO,
        #Feature.P2_REFLECT_RATIO,
        #Feature.P1_LIGHTSCREEN_RATIO,
        #Feature.P2_LIGHTSCREEN_RATIO,
        

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
    #test_out_path="test_features_extracted.csv"

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
    # Carica il train e test set da csv con le feature estratte
    # print(f"\nLoading train_set from {train_out_path}...")
    # train_df = pd.read_csv(train_out_path)
    # Rimuovi la riga 4877 dal dataset
    train_df = train_df.drop(index=4877)
    # test_df = pd.read_csv(test_out_path)
    X_train = train_df.drop(['battle_id', 'player_won'], axis=1)
    y_train = train_df['player_won']
    # X_test = test_df.drop(['battle_id'], axis=1, errors='ignore')


    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=210978)
    #X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # Crea una pipeline con normalizzazione e modello
    print("\nCreating pipeline with MinMaxScaler and LogisticRegression...")
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),#
        #('scaler',StandardScaler()),
        #('scaler',RobustScaler()),
        #('pca', PCA(n_components=0.95)),
        #('classifier', LogisticRegression(random_state=42, max_iter=1000,penalty='l2',solver='liblinear',C=100)) #*
        #('classifier', LogisticRegression(random_state=42, max_iter=1000,penalty='l1',solver='liblinear',C=100)) #*
        #('classifier', LogisticRegressionCV(random_state=42, max_iter=1000,penalty='l2',solver='liblinear',Cs=100)) 
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, penalty='l1',solver='liblinear',C=10))
        #('classifier', LogisticRegressionCV(random_state=210978, max_iter=1000,penalty='l2',solver='liblinear',Cs=100))
        #('classifier',RandomForestClassifier(n_estimators=50))
        #('classifier', LogisticRegression(random_state=210978, max_iter=1000,penalty='l1',solver='liblinear',C=10))
        #('classifier', LogisticRegression(random_state=42, max_iter=1000,penalty='l1',solver='liblinear',C=10))
        #('classifier', LogisticRegressionCV(random_state=42, max_iter=1000, penalty='l2',solver='liblinear',Cs=1))
        #('classifier', LogisticRegression(random_state=42, max_iter=2000)),
        #('classifier',LogisticRegressionCV(random_state=42, max_iter=2000)),
        #('classifier',KNeighborsClassifier(n_neighbors=30,p=1,weights='distance'))
        #('classifier',RandomForestClassifier(max_depth=4,random_state=42))
    ])


    # Addestra e valuta
    #Grid Search per Logistic Regression 
    
    params={
         'classifier__C':[0.01,0.1,1,10,100],
         'classifier__penalty':['l1','l2'],
         'classifier__solver':['liblinear','saga'],
         'classifier__max_iter':[1000,2000]
    }
    

    '''
    params= {
    'classifier__n_neighbors': range(1, 31),
    'classifier__weights': ['uniform', 'distance'],
    'classifier__p': [1, 2]
}
    '''
    grid=GridSearchCV(pipeline,params,cv=5)

    # Pipeline with XGBoost
    # pipeline = Pipeline([
    # ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    # ])
    # Grid Search per XGBoost
    # param_grid = {
    # 'classifier__n_estimators': [100, 200, 300, 500],        # numero di alberi
    # 'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],     # tasso di apprendimento
    # 'classifier__max_depth': [3, 5, 7, 9],                   # profondità massima degli alberi
    # 'classifier__min_child_weight': [1, 3, 5, 7],            # peso minimo di un nodo figlio
    # 'classifier__gamma': [0, 0.1, 0.3, 0.5],                 # regolarizzazione (riduce overfitting)
    # 'classifier__subsample': [0.6, 0.8, 1.0],                # percentuale di campioni per ogni albero
    # 'classifier__colsample_bytree': [0.6, 0.8, 1.0],         # percentuale di feature per ogni albero
    # 'classifier__reg_alpha': [0, 0.01, 0.1, 1],              # L1 regularization (lasso)
    # 'classifier__reg_lambda': [0.5, 1, 2],                   # L2 regularization (ridge)
    # }
    # param_grid = {
    # 'classifier__n_estimators': [100, 300],
    # 'classifier__learning_rate': [0.05, 0.1],
    # 'classifier__max_depth': [3, 6],
    # 'classifier__min_child_weight': [1, 5],
    # 'classifier__gamma': [0, 0.3],
    # 'classifier__subsample': [0.8, 1.0],
    # 'classifier__colsample_bytree': [0.8, 1.0],
    # 'classifier__reg_alpha': [0, 0.1],
    # 'classifier__reg_lambda': [1, 2],
    # }

    #grid = GridSearchCV(
    #estimator=pipeline,
    #param_grid=param_grid,
    #scoring='accuracy',  # o 'f1', 'roc_auc', ecc.
    #cv=5, 
    #verbose=2,
    #n_jobs=-1
    #)

    #trainer = ModelTrainer(grid)
    trainer = ModelTrainer(pipeline)
    trainer.train(X_tr, y_tr)
    trainer.evaluate(X_val, y_val)
    


    # #---------------Feature Utility Code------------------------
    # # ottieni i coefficienti
    #coefficients = pd.Series(pipeline.named_steps['classifier'].coef_[0], index=train_df.columns[2::])

    # # ordina per importanza
    #coefficients = coefficients.abs().sort_values(ascending=False)

    #print("Most useful features:")
    #pd.set_option('display.max_rows', None)
    #print(coefficients)

    # ------------------ Evaluate on Test Set -----------------

    evaluate_test_set(trainer, selected_features, test_file_path)

    #---------------Feature Utility Code GRID------------------------

    # # ottieni il classificatore addestrato dal grid search
    #best_model = grid.best_estimator_.named_steps['classifier']

    # ottieni l'importanza delle feature
    #importances = pd.Series(best_model.coef_[0], index=train_df.columns[2::])

    # # ordina per importanza
    #importances = importances.sort_values(ascending=False)

    # print("Most useful features:")
    pd.set_option('display.max_rows', None)
    #print(importances)

    #print("Best CV score:", grid.best_score_)
    #print("Best params:", grid.best_params_)

    # -------------- Correlation matrix -------------
    corr=train_df.corr()
    
    #print(corr)

    mask = corr.abs() > 0.5
    filtered=corr.where(mask).dropna(axis=0,how='all').dropna(axis=1,how='all')
    plt.figure(figsize=(12, 12))
    sns.heatmap(filtered, cmap="coolwarm", center=0, annot=True)
    plt.title("Feature Correlation Matrix", fontsize=14)
    plt.show()
    
    # ------------------ Feature selection -----------------

    #
    # Creiamo il selettore basato su XGBoost
    # Puoi cambiare threshold="mean" oppure threshold=0.005 per essere più selettivo
    # selector = SelectFromModel(best_model, threshold=0.005, prefit=True)

    # #Applica la selezione delle feature
    # X = train_df.iloc[:, 2:]  # le tue feature (partendo dalla colonna 2)
    # X_selected = selector.transform(X)

    # print(f"\nNumero di feature iniziali: {X.shape[1]}")
    # print(f"Numero di feature selezionate: {X_selected.shape[1]}")

    # #Recupera i nomi delle feature selezionate
    # selected_features = X.columns[selector.get_support()]
    # print("\nFeature selezionate:")
    # print(selected_features.tolist())

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
    submission.to_csv('predict_csv/predictions.csv', index=False)

if __name__ == "__main__":
    main()