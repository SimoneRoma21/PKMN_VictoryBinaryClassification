import json
import pandas as pd
from dataset.dataset_construction import Feature, FeaturePipeline
from dataset.csv_utilities import *
from dataset.extract_utilities import *
from ModelTrainer import ModelTrainer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier


def main():
    # ---------------Feature Extraction Code------------------------
    selected_features = [
        # ------Stats Features---------#
        Feature.P1_FINAL_TEAM_HP,
        Feature.P2_FINAL_TEAM_HP,
        Feature.MEAN_SPE_LAST,
        Feature.MEAN_HP_LAST,
        Feature.MEAN_ATK_LAST,
        Feature.MEAN_SPA_LAST,
        Feature.MEAN_STATS_LAST,
        Feature.MEAN_CRIT,
        # ------Ratio on Stats Features--------#
        Feature.HP_BULK_RATIO,
        Feature.SPE_ATK_RATIO,
        Feature.OFF_DEF_RATIO,
        Feature.OFF_SPAD_RATIO,
        # -------Differential Features on Stats---#
        Feature.HP_TREND_DIFF,
        Feature.ATK_TREND_DIFF,
        Feature.SPA_TREND_DIFF,
        Feature.SPE_TREND_DIFF,
        # ---Feature Infos During Battle----#
        Feature.P1_ALIVE_PKMN,
        Feature.P2_ALIVE_PKMN,
        Feature.P1_SWITCHES_COUNT,
        Feature.P2_SWITCHES_COUNT,
        Feature.P1_AVG_HP_WHEN_SWITCHING,
        Feature.P2_AVG_HP_WHEN_SWITCHING,
        Feature.P1_MAX_DEBUFF_RECEIVED,
        Feature.P2_MAX_DEBUFF_RECEIVED,
        Feature.P1_AVG_MOVE_POWER,
        Feature.P2_AVG_MOVE_POWER,
        Feature.AVG_MOVE_POWER_DIFFERENCE,
        Feature.P1_OFFENSIVE_RATIO,
        Feature.P2_OFFENSIVE_RATIO,
        Feature.OFFENSIVE_RATIO_DIFFERENCE,
        Feature.P1_MOVED_FIRST_COUNT,
        Feature.P2_MOVED_FIRST_COUNT,
        Feature.SPEED_ADVANTAGE_RATIO,
        # ----Feature Status of Pokemons----#
        Feature.P1_FROZEN_PKMN,
        Feature.P2_FROZEN_PKMN,
        Feature.P1_PARALIZED_PKMN,
        Feature.P2_PARALIZED_PKMN,
        Feature.P1_SLEEP_PKMN,
        Feature.P2_SLEEP_PKMN,
        Feature.P1_POISON_PKMN,
        Feature.P2_POISON_PKMN,
        Feature.P1_BURNED_PKMN,
        Feature.P2_BURNED_PKMN,
        # ----Feature Pokemon Moves----#
        Feature.P1_PKMN_REFLECT,
        Feature.P2_PKMN_REFLECT,
        Feature.P1_PKMN_REST,
        Feature.P2_PKMN_REST,
        Feature.P1_PKMN_EXPLOSION,
        Feature.P2_PKMN_EXPLOSION,
        Feature.P1_PKMN_THUNDERWAVE,
        Feature.P2_PKMN_THUNDERWAVE,
        Feature.P1_PKMN_RECOVER,
        Feature.P2_PKMN_RECOVER,
        Feature.P1_PKMN_TOXIC,
        Feature.P2_PKMN_TOXIC,
        Feature.P1_PKMN_FIRESPIN,
        Feature.P2_PKMN_FIRESPIN,
    ]
    feature_pipeline = FeaturePipeline(selected_features)

    train_file_path = "../data/train.jsonl"
    test_file_path = "../data/test.jsonl"
    train_out_path = "predict_csv/train_features_extracted.csv"

    print("Loading training data...")
    train_data = []
    with open(train_file_path, "r") as f:
        for line in f:
            train_data.append(json.loads(line))

    # Extract features from train_set
    print("\nExtracting features from training data...")
    train_df = feature_pipeline.extract_features(train_data)
    print("\nTraining features preview:")
    print(train_df.head())
    # Save dataset in CSV
    train_df.to_csv(train_out_path, index=False)

    # ---------------Model Training and Evaluation Code------------------------

    # Remove row 4877 from the train dataset
    train_df = train_df.drop(index=4877)

    X_train = train_df.drop(["battle_id", "player_won"], axis=1)
    y_train = train_df["player_won"]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )


    # Pipelines: Scaling only the linear models
    pipe_lr = Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    C=1,
                    penalty="l1",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    pipe_sgd = Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "sgd",
                SGDClassifier(
                    loss="modified_huber",
                    penalty="l1",
                    alpha=0.001,
                    learning_rate="adaptive",
                    eta0=0.01,
                    max_iter=1000,
                    tol=0.0001,
                    random_state=42,
                ),
            ),
        ]
    )

    # Tree and Boosting
    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        colsample_bytree=0.8,
        gamma=0.3,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=5,
        n_estimators=400,
        reg_alpha=0,
        reg_lambda=2,
        subsample=0.8,
    )

    rf = RandomForestClassifier(
        random_state=42,
        bootstrap=True,
        max_depth=12,
        max_features="sqrt",
        min_samples_leaf=3,
        min_samples_split=5,
        n_estimators=300,
    )

    # Base estimators
    base_estimators = [("lr", pipe_lr), ("xgb", xgb), ("rf", rf), ("sgd", pipe_sgd)]

    # Meta model with gridSearchCV for the hyperparameters
    meta = LogisticRegression(random_state=42)

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )
    param_grid = {
        "final_estimator__C": [0.01, 0.1, 1, 10, 100],
        "final_estimator__penalty": ["l1", "l2"],
        "final_estimator__solver": ["liblinear", "saga"],
        "final_estimator__max_iter": [1000, 2000],
    }

    gs = GridSearchCV(stack, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)

    trainer = ModelTrainer(gs)
    trainer.train(X_tr, y_tr)
    trainer.evaluate(X_val, y_val)

    print("Best CV score:", gs.best_score_)
    print("Best params:", gs.best_params_)

    # ------------------ Evaluate on Test Set -----------------

    evaluate_test_set(trainer, selected_features, test_file_path)


def evaluate_test_set(trainer: ModelTrainer, feature_list: list, test_file_path: str):

    feature_pipeline = FeaturePipeline(
        feature_list, cache_dir="../data/test_feature_cache"
    )

    print("\nLoading test data...")
    test_data = []
    with open(test_file_path, "r") as f:
        for line in f:
            test_data.append(json.loads(line))

    # Extract features from test set
    print("\nExtracting features from test data...")
    test_df = feature_pipeline.extract_features(test_data, show_progress=True)

    X_test = test_df.drop(["battle_id"], axis=1, errors="ignore")

    # Predict on test set
    predictions = trainer.predict(X_test)

    submission = pd.DataFrame(
        {"battle_id": test_df["battle_id"], "player_won": predictions}
    )
    submission.to_csv(
        "predict_csv/predictions_ModelStacking_Submission3.csv", index=False
    )

if __name__ == "__main__":
    main()
