import json
import pandas as pd
from dataset.dataset_construction import Feature, FeaturePipeline
from ModelTrainer import ModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main():

    selected_features = [
        Feature.P1_MEAN_HP_START,
        Feature.P2_MEAN_HP_START,
        Feature.MEAN_SPD_START,
        # Feature.MEAN_SPD_LAST,
        # Feature.MEAN_HP_LAST,
        Feature.P1_ALIVE_PKMN,
        Feature.P2_ALIVE_PKMN
    ]

    pipeline = FeaturePipeline(selected_features)

     # Carica i dati
    train_file_path = '../data/train.jsonl'
    test_file_path = '../data/test.jsonl'

    print("Loading training data...")
    train_data = []
    with open(train_file_path, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    print("Loading test data...")
    test_data = []
    with open(test_file_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # Estrai le feature
    print("\nExtracting features from training data...")
    train_df = pipeline.extract_features(train_data)
    
    print("Extracting features from test data...")
    test_df = pipeline.extract_features(test_data)
    
    print("\nTraining features preview:")
    print(train_df.head())

    X_train = train_df.drop(['battle_id', 'player_won'], axis=1)
    y_train = train_df['player_won']
    X_test = test_df.drop(['battle_id'], axis=1, errors='ignore')


    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    
    # Addestra e valuta
    model = LogisticRegression(random_state=42, max_iter=2000)
    trainer = ModelTrainer(model)
    trainer.train(X_tr, y_tr)
    trainer.evaluate(X_val, y_val)
    
    # Predici sul test set
    predictions = trainer.predict(X_test)
    
    submission = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': predictions
    })
    submission.to_csv('predictions.csv', index=False)


if __name__ == "__main__":
    main()