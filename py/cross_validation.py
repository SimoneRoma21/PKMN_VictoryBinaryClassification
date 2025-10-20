import json
import pandas as pd
from dataset.dataset_construction import Feature, FeaturePipeline
from ModelTrainer import ModelTrainer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score
import numpy as np


def save_features(train_out_path,test_out_path):
    selected_features = [
        Feature.P1_MEAN_HP_START,
        Feature.P2_MEAN_HP_START,
        Feature.MEAN_SPE_START, 
        Feature.MEAN_SPE_LAST, 
        Feature.MEAN_HP_LAST, 
        Feature.P1_ALIVE_PKMN,
        Feature.P2_ALIVE_PKMN,
        Feature.MEAN_STATS_START,
        # Feature.TOTAL_TURNS, 
        Feature.P1_SWITCHES_COUNT,
        Feature.P2_SWITCHES_COUNT,
        Feature.P1_STATUS_INFLICTED,
        Feature.P2_STATUS_INFLICTED,
        Feature.SWITCHES_DIFFERENCE,
        Feature.STATUS_INFLICTED_DIFFERENCE,
        Feature.P1_FINAL_TEAM_HP,
        Feature.P2_FINAL_TEAM_HP,
        Feature.FINAL_TEAM_HP_DIFFERENCE,
        Feature.P1_FIRST_FAINT_TURN,
        Feature.P1_AVG_HP_WHEN_SWITCHING,
        Feature.P1_MAX_DEBUFF_RECEIVED,
        Feature.P2_MAX_DEBUFF_RECEIVED,
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

    # Estrai le feature train_set
    print("\nExtracting features from training data...")
    train_df = pipeline.extract_features(train_data)
    print("\nTraining features preview:")
    print(train_df.head())
    # Salva il dataset in un file CSV
    train_df.to_csv(train_out_path, index=False)

    # print("Loading test data...")
    # test_data = []
    # with open(test_file_path, 'r') as f:
    #     for line in f:
    #         test_data.append(json.loads(line))
    # # Estrai le feature test_set
    # print("\nExtracting features from test data...")
    # test_df = pipeline.extract_features(test_data)
    # print("\nTest features preview:")
    # print(test_df.head())
    # # Salva il dataset in un file CSV
    # test_df.to_csv(test_out_path, index=False)


def main(train_out_path,test_out_path):
    # Carica il train e test set da csv con le feature estratte
    print(f"\nLoading train_set from {train_out_path}...")
    train_df = pd.read_csv(train_out_path)
    #Rimuovi la riga 4877 dal dataset
    # train_df = train_df.drop(index=4877)
    # test_df = pd.read_csv(test_out_path)

    X_train = train_df.drop(['battle_id', 'player_won'],axis=1)
    y_train = train_df['player_won']
    # Param grid per fine tuning
    param_grid = { 
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], # Regola la regolarizzazione di lambda
        'penalty': ['l1', 'l2'], # Tipo di regolarizzazione l1 e l2
        'solver': ['liblinear'] # Supporta i 2 penalty
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    best_score = {"mean_acc": 0, "mean f1": 0, "mean prec": 0, "mean rec": 0,"cm":0}
    best_param = None
    # Cross validation
    for C in param_grid['C']:
        for penalty in param_grid['penalty']:
            for solver in param_grid['solver']:
                fold_accuracies = []
                fold_f1s = []
                fold_precisions = []
                fold_recalls = []
                fold_cm=[]
                for train_idx, val_idx in kfold.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    # Scaler
                    scaler = StandardScaler()
                    X_tr_scal = scaler.fit_transform(X_tr)
                    X_val_scal = scaler.transform(X_val)

                    #Addestra il modello
                    model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=2000, random_state=42)

                    model.fit(X_tr_scal, y_tr)
                    preds = model.predict(X_val_scal)
                    acc = accuracy_score(y_val, preds)
                    f1 = f1_score(y_val, preds, zero_division=0)
                    prec = precision_score(y_val, preds, zero_division=0)
                    rec = recall_score(y_val, preds, zero_division=0)
                    cm = confusion_matrix(y_val, preds)                    
                    
                    fold_accuracies.append(acc)
                    fold_f1s.append(f1)
                    fold_precisions.append(prec)
                    fold_recalls.append(rec)
                    fold_cm.append(cm)
                mean_acc = np.mean(fold_accuracies)
                mean_f1 = np.mean(fold_f1s)
                mean_prec = np.mean(fold_precisions)
                mean_rec = np.mean(fold_recalls)
                print(f"C={C}, penalty={penalty}: mean acc = {mean_acc:.4f}, mean f1 = {mean_f1:.4f}, mean prec = {mean_prec:.4f}, mean rec = {mean_rec:.4f}")

                # Se migliora, aggiorna i migliori
                if mean_acc> best_score.get("mean_acc"):
                    best_score = {"mean_acc": mean_acc, "mean f1": mean_f1, "mean prec": mean_prec, "mean rec": mean_prec,"cm":cm}
                    best_params = {'C': C, 'penalty': penalty, 'solver': solver}

    print("\nBest params found:", best_params)
    print("Best CV accuracy:", best_score)

    print("Confusion Matrix")
    print(best_score.get("cm"))
    print(f"\nTrue Negatives:  {cm[0][0]:4d}  |  False Positives: {cm[0][1]:4d}")
    print(f"False Negatives: {cm[1][0]:4d}  |  True Positives:  {cm[1][1]:4d}")


    # # Testa il modello con i migliori iperparametri sul test set
    # best_model = grid_search.best_estimator_
    # test_accuracy = best_model.score(X_test, Y_test)
    # print(f"Test set accuracy with best hyperparameters: {test_accuracy}")
    
    # submission = pd.DataFrame({
    #     'battle_id': test_df['battle_id'],
    #     'player_won': predictions
    # })
    # submission.to_csv('predictions.csv', index=False)


if __name__ == "__main__":
    train_out_path="train_features_extracted.csv"
    test_out_path="train_features_extracted.csv"
    # save_features(train_out_path,test_out_path)
    main(train_out_path,test_out_path)