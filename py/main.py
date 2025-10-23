import json
import pandas as pd
from dataset.dataset_construction import Feature, FeaturePipeline
from ModelTrainer import ModelTrainer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

def save_features(train_out_path,test_out_path):
    selected_features = [
        Feature.P1_MEAN_HP_START, #*
        Feature.P2_MEAN_HP_START, #*
        Feature.MEAN_HP_DIFFERENCE_START,
        Feature.MEAN_HP_LAST, #*
        #Feature.P1_MEAN_HP_LAST
        #Feature.P2_MEAN_HP_LAST
        #Feature.MEAN_HP_DIFFERENCE_LAST
        Feature.MEAN_SPE_START,  #*
        #Feature.P1_MEAN_SPE_START
        #Feature.P2_MEAN_SPE_START
        #Feature.MEAN_SPE_DIFFERENCE_START
        Feature.MEAN_SPE_LAST,  #*
        #Feature.P1_MEAN_SPE_LAST
        #Feature.P2_MEAN_SPE_LAST
        #Feature.MEAN_SPE_DIFFERENCE_LAST
        Feature.P1_ALIVE_PKMN, #*
        Feature.P2_ALIVE_PKMN, #*
        Feature.ALIVE_PKMN_DIFFERENCE,
        #Feature.WEAKNESS_TEAMS_START, #*
        #Feature.WEAKNESS_TEAMS_LAST, #*
        #Feature.ADVANTAGE_WEAK_START, #*
        #Feature.ADVANTAGE_WEAK_LAST, #*
        Feature.MEAN_STATS_START, #*
        Feature.MEAN_STATS_LAST, #*
        #
        # Feature.TOTAL_TURNS, 
        Feature.P1_SWITCHES_COUNT,
        Feature.P2_SWITCHES_COUNT,
        Feature.P1_STATUS_INFLICTED, #*
        Feature.P2_STATUS_INFLICTED, #*
        Feature.SWITCHES_DIFFERENCE,
        Feature.STATUS_INFLICTED_DIFFERENCE, #*
        Feature.P1_FINAL_TEAM_HP, #*
        Feature.P2_FINAL_TEAM_HP, #*
        Feature.FINAL_TEAM_HP_DIFFERENCE, #*
        Feature.P1_FIRST_FAINT_TURN,
        Feature.P1_AVG_HP_WHEN_SWITCHING,
        Feature.P2_AVG_HP_WHEN_SWITCHING,
        Feature.P1_MAX_DEBUFF_RECEIVED,
        Feature.P2_MAX_DEBUFF_RECEIVED,
        Feature.P1_AVG_MOVE_POWER, #*
        Feature.P2_AVG_MOVE_POWER, #*
        Feature.AVG_MOVE_POWER_DIFFERENCE, #*
        Feature.P1_OFFENSIVE_RATIO,
        Feature.P2_OFFENSIVE_RATIO,
        Feature.OFFENSIVE_RATIO_DIFFERENCE,
        Feature.P1_MOVED_FIRST_COUNT,
        Feature.P2_MOVED_FIRST_COUNT,
        Feature.SPEED_ADVANTAGE_RATIO
#
    ]#

    pipeline = FeaturePipeline(selected_features)
    # Carica i dati
    train_file_path = '../data/train.jsonl'
    test_file_path = '../data/test.jsonl'

    print("Loading training data...")
    train_data = []
    with open(train_file_path, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))

    # Estrai le feature train_set
    print("\nExtracting features from training data...")
    train_df = pipeline.extract_features(train_data)
    print("\nTraining features preview:")
    print(train_df.head())
    # Salva il dataset in un file CSV
    train_df.to_csv(train_out_path, index=False)

    print("Loading test data...")
    test_data = []
    with open(test_file_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    # Estrai le feature test_set
    print("\nExtracting features from test data...")
    test_df = pipeline.extract_features(test_data)
    print("\nTest features preview:")
    print(test_df.head())
    # Salva il dataset in un file CSV
    test_df.to_csv(test_out_path, index=False)


def main():
    #---------------Feature Extraction Code------------------------
    train_out_path="train_features_extracted.csv"
    test_out_path="test_features_extracted.csv"
    # Uncomment to extract and save features
    # save_features(train_out_path,test_out_path) 

    #---------------Model Training and Evaluation Code------------------------
    # Carica il train e test set da csv con le feature estratte
    print(f"\nLoading train_set from {train_out_path}...")
    train_df = pd.read_csv(train_out_path)
    # Rimuovi la riga 4877 dal dataset
    train_df = train_df.drop(index=4877)
    test_df = pd.read_csv(test_out_path)

    X_train = train_df.drop(['battle_id', 'player_won'], axis=1)
    y_train = train_df['player_won']
    X_test = test_df.drop(['battle_id'], axis=1, errors='ignore')


    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Crea una pipeline con normalizzazione e modello
    print("\nCreating pipeline with MinMaxScaler and LogisticRegression...")
    pipeline = Pipeline([
        #  ('scaler', MinMaxScaler()),
        ('scaler',StandardScaler()),
        # ('scaler',RobustScaler()),
         ('classifier', LogisticRegressionCV(random_state=42, max_iter=1000,solver='liblinear',Cs=1))
        #('classifier', LogisticRegression(random_state=42, max_iter=2000)),
        #('classifier',LogisticRegressionCV(random_state=42, max_iter=2000)),
    ])
    # pipeline with RandomForestClassifier
    # pipeline = Pipeline([
    #     ('classifier', RandomForestClassifier(n_estimators=4, random_state=42))
    # ])

    # Addestra e valuta

    params={
        'classifier__Cs':[0.01,0.1,1,10,100],
        'classifier__solver':['liblinear','saga'],
        'classifier__max_iter':[1000,2000]
    }
    grid=GridSearchCV(pipeline,params,cv=5)

    trainer = ModelTrainer(grid)
    # trainer = ModelTrainer(pipeline)
    trainer.train(X_tr, y_tr)
    trainer.evaluate(X_val, y_val)
    
    # Predici sul test set
    predictions = trainer.predict(X_test)
    
    submission = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': predictions
    })
    submission.to_csv('predictions.csv', index=False)



    #---------------Feature Utility Code------------------------
    # ottieni i coefficienti
    #coefficients = pd.Series(pipeline.named_steps['classifier'].coef_[0], index=train_df.columns[2::])

    # ordina per importanza
    #coefficients = coefficients.abs().sort_values(ascending=False)

    #print("Most useful features:")
    #print(coefficients)


    print(train_df.corr())
    # print("Best CV score:", grid.best_score_)
    # print("Best params:", grid.best_params_)

if __name__ == "__main__":
    main()