import pandas as pd
import pickle
import optuna
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

df = pd.read_csv('cs_clean.csv')
X = df.drop(columns=['bomb_planted'])
y = df['bomb_planted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nГиперпараметры для логистической регрессии\n")
param_grid_lr = {
    'C': [0.1, 1, 10], 
    'penalty': ['l2'], 
    'solver': ['lbfgs', 'saga'], 
    'max_iter': [2000]
}

grid_search_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_lr = grid_search_lr.best_estimator_

print("\nGradientBoosting через Optuna\n")
def objective_gb_clf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1).mean()

study_gb_clf = optuna.create_study(direction='maximize')
study_gb_clf.optimize(objective_gb_clf, n_trials=15)
best_gb = GradientBoostingClassifier(**study_gb_clf.best_params, random_state=42)

print("\nCatBoostClassifier через Optuna\n")
def objective_cb_clf(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'verbose': False,
        'random_seed': 42,
        'eval_metric': 'F1'
    }
    model = CatBoostClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1).mean()

study_cb_clf = optuna.create_study(direction='maximize')
study_cb_clf.optimize(objective_cb_clf, n_trials=15)

best_cb_params = {
    'iterations': study_cb_clf.best_params['iterations'],
    'depth': study_cb_clf.best_params['depth'],
    'learning_rate': study_cb_clf.best_params['learning_rate'],
    'verbose': False,
    'random_seed': 42,
    'eval_metric': 'F1'
}
best_cb = CatBoostClassifier(**best_cb_params)

print("\nBaggingClassifier через Optuna\n")
def objective_bagging_clf(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 20, 500),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'estimator__max_depth': trial.suggest_int('estimator__max_depth', 5, 35),
        'estimator__criterion': trial.suggest_categorical('estimator__criterion', ['gini', 'entropy'])
    }
    base_model = DecisionTreeClassifier(random_state=42)
    model = BaggingClassifier(estimator=base_model, random_state=42, n_jobs=-1)
    model.set_params(**param)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()

study_bag_clf = optuna.create_study(direction='maximize')
study_bag_clf.optimize(objective_bagging_clf, n_trials=30)

best_bag_clf = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42)
best_bag_clf.set_params(**study_bag_clf.best_params)

print("\nStackingClassifier через Optuna\n")
def objective_stacking_clf(trial):
    gb_params = {
        'n_estimators': trial.suggest_int('gbr__n_estimators', 50, 300),
        'max_depth': trial.suggest_int('gbr__max_depth', 3, 10),
        'learning_rate': trial.suggest_float('gbr__learning_rate', 0.01, 0.1)
    }
    bag_params = {
        'n_estimators': trial.suggest_int('bag__n_estimators', 10, 100),
        'estimator__max_depth': trial.suggest_int('bag__max_depth', 5, 20)
    }
    
    base_models = [
        ('gbc', GradientBoostingClassifier(random_state=42, **gb_params)),
        ('bag', BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=bag_params['estimator__max_depth']), 
                                  n_estimators=bag_params['n_estimators'], 
                                  random_state=42))
    ]
    meta_model = LogisticRegression(C=trial.suggest_float('meta_C', 0.01, 10.0))
    
    stack = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)
    return cross_val_score(stack, X_train, y_train, cv=3, scoring='f1').mean()

study_stack_clf = optuna.create_study(direction='maximize')
study_stack_clf.optimize(objective_stacking_clf, n_trials=15)

bp = study_stack_clf.best_params
best_stack = StackingClassifier(
    estimators=[
        ('gbc', GradientBoostingClassifier(n_estimators=bp['gbr__n_estimators'], 
                                           max_depth=bp['gbr__max_depth'], 
                                           learning_rate=bp['gbr__learning_rate'], 
                                           random_state=42)),
        ('bag', BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=bp['bag__max_depth']), 
                                  n_estimators=bp['bag__n_estimators'], 
                                  random_state=42))
    ],
    final_estimator=LogisticRegression(C=bp['meta_C'])
)

print("\nОбучение всех ансамблей и нейросети:\n")

models = {
    "ML1_LogisticRegression": best_lr,
    "ML2_GradientBoosting": best_gb,
    "ML3_CatBoost": best_cb,
    "ML4_RandomForest": best_bag_clf,
    "ML5_Stacking": best_stack,
    "ML6_NeuralNetwork": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

print("\nМетрики качества F1")
for name, model in models.items():
    if name != "ML1_LogisticRegression":
        model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    print(f"{name} F1: {f1:.4f}")
    
    if "CatBoost" in name:
        model.save_model(f"{name}.cbm")
    else:
        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)