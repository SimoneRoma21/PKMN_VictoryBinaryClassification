import json
import pandas as pd
from py.dataset.dataset_construction import Feature, FeaturePipeline
from py.ModelTrainer import ModelTrainer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score
import numpy as np

def main(train_out_path,test_out_path):
    # Carica il train e test set da csv con le feature estratte
    print(f"\nLoading train_set from {train_out_path}...")
    train_df = pd.read_csv(train_out_path)
    #Rimuovi la riga 4877 dal dataset
    # train_df = train_df.drop(index=4877)
    test_df = pd.read_csv(test_out_path)

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
                    # scaler = StandardScaler()
                    scaler = RobustScaler()
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

    final_model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], solver=best_params['solver'], max_iter=2000, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    scaler = RobustScaler()
    X_tr_scal = scaler.fit_transform(X_tr)
    X_val_scal = scaler.transform(X_val)
    final_model.fit(X_tr_scal, y_tr)
    val_preds = final_model.predict(X_val_scal)
    print("\nFinal model evaluation on validation set:")
    print(classification_report(y_val, val_preds))

    print("Confusion Matrix")
    print(best_score.get("cm"))
    print(f"\nTrue Negatives:  {cm[0][0]:4d}  |  False Positives: {cm[0][1]:4d}")
    print(f"False Negatives: {cm[1][0]:4d}  |  True Positives:  {cm[1][1]:4d}")

    #Predici sul test_set
    X_test = test_df.drop(['battle_id'], axis=1, errors='ignore')
    X_test_scal = scaler.transform(X_test)
    test_preds = final_model.predict(X_test_scal)
    # Salva le predizioni
    submission = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': test_preds
    })
    submission.to_csv('predictions_cross_val.csv', index=False)





if __name__ == "__main__":
    train_out_path="train_features_extracted.csv"
    test_out_path="test_features_extracted.csv"
    main(train_out_path,test_out_path)