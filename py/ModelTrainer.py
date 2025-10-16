import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

class ModelTrainer:
    """Classe per addestrare e valutare modelli"""
    
    def __init__(self, model=None):
        self.model = model or RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_importance = None
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Addestra il modello"""
        print("Training model...")
        self.model.fit(X, y)
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Print accuracy on training set
        train_accuracy = accuracy_score(y, self.model.predict(X))
        print(f"Training Accuracy: {train_accuracy:.4f}")

        print("✓ Training complete!")
        return self
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        """Valuta il modello"""
        y_pred = self.model.predict(X)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        print(f"\nAccuracy: {accuracy_score(y, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        if self.feature_importance is not None:
            print("\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10).to_string(index=False))
        
        return y_pred
    
    def predict(self, X: pd.DataFrame):
        """Fa predizioni"""
        return self.model.predict(X)