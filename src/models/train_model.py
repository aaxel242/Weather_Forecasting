import joblib
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_classifier_model_LR(features, labels, split_factor=0.8, model_path="src/models/trained_classifier_lr.pkl"):
    # Convertimos labels a entero para evitar problemas de tipos
    labels = labels.astype(int)
    
    split = int(len(features) * split_factor)
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = labels.iloc[:split], labels.iloc[split:]

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])

    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    predictions = pd.Series(model.predict(X_test), index=y_test.index)
    return model, predictions

def train_classifier_model_RF(features, labels, split_factor=0.8, model_path="src/models/trained_classifier_rf.pkl"):
    # Convertimos labels a entero para evitar problemas de tipos
    labels = labels.astype(int)
    
    split = int(len(features) * split_factor)
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = labels.iloc[:split], labels.iloc[split:]

    model = Pipeline([
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])

    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    predictions = pd.Series(model.predict(X_test), index=y_test.index)
    return model, predictions

def train_models (features, labels, model_path_rf, model_path_lr):

    model_rf, y_pred_rf = train_classifier_model_RF(features, labels, model_path=model_path_rf)
    model_lr, y_pred_lr = train_classifier_model_LR(features, labels, model_path=model_path_lr)

    return model_rf, y_pred_rf, model_lr, y_pred_lr