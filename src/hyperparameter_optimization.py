#!/usr/bin/env python3
"""
Hyperparameter Optimization for NeurIPS 2025 Polymer Prediction
===============================================================

This script implements automated hyperparameter optimization using Optuna
for the polymer prediction baseline model.

Usage:
    python src/hyperparameter_optimization.py --data_dir data/ --n_trials 100

Author: Enhanced optimization for competition baseline
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import joblib
import json
from datetime import datetime
import warnings

from baseline_model import PolymerPredictor, load_competition_data, identify_columns

warnings.filterwarnings('ignore')


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for polymer prediction models.
    """
    
    def __init__(self, X_train, y_train, cv_folds=5, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
    def objective_xgboost(self, trial):
        """Objective function for XGBoost hyperparameter optimization."""
        
        # Define hyperparameter search space
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'verbosity': 0,
            'random_state': self.random_state,
            'n_jobs': -1,
            
            # Hyperparameters to optimize
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        }
        
        # Cross-validation
        model = xgb.XGBRegressor(**params)
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=self.kfold, scoring='neg_root_mean_squared_error', n_jobs=1
        )
        
        return -cv_scores.mean()  # Return RMSE (minimize)
    
    def objective_lightgbm(self, trial):
        """Objective function for LightGBM hyperparameter optimization."""
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': self.random_state,
            'n_jobs': -1,
            
            # Hyperparameters to optimize
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-8, 10.0, log=True),
        }
        
        # Cross-validation
        model = lgb.LGBMRegressor(**params)
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=self.kfold, scoring='neg_root_mean_squared_error', n_jobs=1
        )
        
        return -cv_scores.mean()
    
    def objective_catboost(self, trial):
        """Objective function for CatBoost hyperparameter optimization."""
        
        params = {
            'objective': 'RMSE',
            'verbose': False,
            'random_state': self.random_state,
            'thread_count': -1,
            
            # Hyperparameters to optimize
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        }
        
        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
        elif params['bootstrap_type'] == 'Bernoulli':
            params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
        
        # Cross-validation
        model = CatBoostRegressor(**params)
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=self.kfold, scoring='neg_root_mean_squared_error', n_jobs=1
        )
        
        return -cv_scores.mean()
    
    def optimize(self, model_type='xgboost', n_trials=100, timeout=None):
        """
        Run hyperparameter optimization.
        
        Parameters:
        -----------
        model_type : str
            Type of model to optimize ('xgboost', 'lightgbm', 'catboost')
        n_trials : int
            Number of optimization trials
        timeout : int, optional
            Timeout in seconds
        
        Returns:
        --------
        optuna.Study
            Optimization study object
        """
        
        # Select objective function
        if model_type == 'xgboost':
            objective = self.objective_xgboost
        elif model_type == 'lightgbm':
            objective = self.objective_lightgbm
        elif model_type == 'catboost':
            objective = self.objective_catboost
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        print(f"Starting {model_type} optimization with {n_trials} trials...")
        
        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        print(f"\\nOptimization completed!")
        print(f"Best RMSE: {study.best_value:.6f}")
        print(f"Best parameters: {study.best_params}")
        
        return study


def train_optimized_model(X_train, y_train, X_test, best_params, model_type='xgboost', cv_folds=5, random_state=42):
    """
    Train model with optimized hyperparameters using cross-validation.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    best_params : dict
        Optimized hyperparameters
    model_type : str
        Model type
    cv_folds : int
        Number of CV folds
    random_state : int
        Random state
    
    Returns:
    --------
    tuple
        CV scores, OOF predictions, test predictions, trained models
    """
    
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    cv_scores = []
    models = []
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    
    print(f"Training {model_type} with optimized parameters...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"Fold {fold + 1}/{cv_folds}")
        
        # Split data
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train model
        if model_type == 'xgboost':
            model = xgb.XGBRegressor(**best_params, random_state=random_state, n_jobs=-1)
        elif model_type == 'lightgbm':
            model = lgb.LGBMRegressor(**best_params, random_state=random_state, n_jobs=-1)
        elif model_type == 'catboost':
            model = CatBoostRegressor(**best_params, random_state=random_state, thread_count=-1, verbose=False)
        
        model.fit(X_fold_train, y_fold_train)
        
        # Predictions
        val_pred = model.predict(X_fold_val)
        test_pred = model.predict(X_test)
        
        # Store results
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / cv_folds
        models.append(model)
        
        # Calculate metrics
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
        cv_scores.append(fold_rmse)
        
        print(f"  RMSE: {fold_rmse:.6f}")
    
    # Overall performance
    cv_rmse = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
    
    print(f"\\nCV RMSE: {cv_rmse:.6f} ± {cv_std:.6f}")
    print(f"OOF RMSE: {oof_rmse:.6f}")
    
    return cv_scores, oof_predictions, test_predictions, models


def main():
    """Main optimization pipeline"""
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for Polymer Prediction')
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='models/', help='Output directory')
    parser.add_argument('--model_type', type=str, default='xgboost', 
                       choices=['xgboost', 'lightgbm', 'catboost'], help='Model type to optimize')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--n_features', type=int, default=200, help='Number of features to select')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("HYPERPARAMETER OPTIMIZATION FOR POLYMER PREDICTION")
    print("=" * 60)
    
    # Load data and prepare features (reuse baseline model pipeline)
    print("Loading data and preparing features...")
    train_df, test_df = load_competition_data(args.data_dir)
    if train_df is None:
        print("Failed to load data. Exiting.")
        return
    
    smiles_col, target_col = identify_columns(train_df)
    print(f"Using SMILES column: {smiles_col}")
    print(f"Using target column: {target_col}")
    
    # Prepare features using baseline model
    model = PolymerPredictor(random_state=args.random_state)
    
    print("Generating molecular features...")
    train_features = model.calculate_molecular_descriptors(train_df[smiles_col].tolist(), 'comprehensive')
    test_features = model.calculate_molecular_descriptors(test_df[smiles_col].tolist(), 'comprehensive')
    
    print("Feature engineering...")
    train_features_eng = model.engineer_features(train_features)
    test_features_eng = model.engineer_features(test_features)
    
    print("Preprocessing features...")
    X_train, X_test = model.preprocess_features(train_features_eng, test_features_eng)
    
    y_train = train_df[target_col]
    X_train_final, X_test_final = model.select_features(X_train, y_train, X_test, args.n_features)
    
    print(f"Final feature shapes: Train {X_train_final.shape}, Test {X_test_final.shape}")
    
    # Run optimization
    print("\\n" + "="*50)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*50)
    
    optimizer = HyperparameterOptimizer(
        X_train_final, y_train, 
        cv_folds=args.cv_folds, 
        random_state=args.random_state
    )
    
    study = optimizer.optimize(
        model_type=args.model_type,
        n_trials=args.n_trials,
        timeout=args.timeout
    )
    
    # Train final model with best parameters
    print("\\n" + "="*50)
    print("TRAINING FINAL MODEL")
    print("="*50)
    
    cv_scores, oof_predictions, test_predictions, models = train_optimized_model(
        X_train_final, y_train, X_test_final,
        study.best_params, args.model_type,
        args.cv_folds, args.random_state
    )
    
    # Save results
    print("\\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save optimization study
    study_path = output_dir / f'optimization_study_{args.model_type}_{timestamp}.pkl'
    joblib.dump(study, study_path)
    print(f"Optimization study saved to {study_path}")
    
    # Save optimized model
    optimized_model_artifacts = {
        'models': models,
        'best_params': study.best_params,
        'cv_scores': cv_scores,
        'oof_predictions': oof_predictions,
        'test_predictions': test_predictions,
        'feature_selector': model.feature_selector,
        'selected_features': model.selected_features.tolist(),
        'model_type': args.model_type,
        'optimization_score': study.best_value
    }
    
    model_path = output_dir / f'optimized_{args.model_type}_model_{timestamp}.pkl'
    joblib.dump(optimized_model_artifacts, model_path)
    print(f"Optimized model saved to {model_path}")
    
    # Save optimization results
    optimization_results = {
        'timestamp': timestamp,
        'model_type': args.model_type,
        'n_trials': args.n_trials,
        'best_score': study.best_value,
        'best_params': study.best_params,
        'cv_scores': cv_scores,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'n_features': len(model.selected_features),
        'smiles_column': smiles_col,
        'target_column': target_col
    }
    
    results_path = output_dir / f'optimization_results_{args.model_type}_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(optimization_results, f, indent=2, default=str)
    print(f"Optimization results saved to {results_path}")
    
    # Create submission
    from baseline_model import create_submission
    submissions_dir = Path('submissions')
    submissions_dir.mkdir(exist_ok=True)
    
    submission_path = submissions_dir / f'optimized_{args.model_type}_submission_{timestamp}.csv'
    create_submission(test_df, test_predictions, submission_path)
    
    # Print optimization insights
    print("\\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Model type: {args.model_type}")
    print(f"Trials completed: {len(study.trials)}")
    print(f"Best RMSE: {study.best_value:.6f}")
    print(f"Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print(f"\\nFinal CV performance: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    print(f"\\nFiles saved:")
    print(f"  • Study: {study_path}")
    print(f"  • Model: {model_path}")
    print(f"  • Results: {results_path}")
    print(f"  • Submission: {submission_path}")
    print("="*60)


if __name__ == "__main__":
    main()