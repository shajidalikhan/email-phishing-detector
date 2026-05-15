# XGBoost Training Script
from xgboost import XGBClassifier

def train_xgboost(X_train, y_train):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model