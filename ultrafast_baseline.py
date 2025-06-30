#!/usr/bin/env python3
"""
Ultra-fast Baseline - Test the XGBoost infinite value fix
========================================================

This baseline uses minimal features to test the core algorithm fix quickly.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


def ultrafast_features(smiles_list):
    """Create minimal features."""
    features = []
    for smiles in smiles_list:
        feat = {
            'length': len(smiles),
            'carbon_count': smiles.count('C'),
            'star_count': smiles.count('*'),
            'complexity': len(set(smiles))
        }
        features.append(feat)
    return pd.DataFrame(features)


def robust_train_target(X_train, y_target, target_name, cv_folds=2):
    """Test the robust training with XGBoost fallback."""
    
    print(f"\nTraining {target_name}...")
    
    # Handle NaN values in target
    valid_mask = y_target.notna()
    valid_count = valid_mask.sum()
    
    print(f"  Valid samples: {valid_count}/{len(y_target)}")
    
    if valid_count < 10:
        print(f"  Using mean prediction")
        mean_val = y_target.mean() if valid_count > 0 else 0.0
        return None, mean_val
    
    # Get valid data
    X_valid = X_train[valid_mask]
    y_valid = y_target[valid_mask]
    
    # Comprehensive cleaning
    print("  Cleaning data...")
    X_clean = X_valid.replace([np.inf, -np.inf], np.nan).fillna(0)
    y_clean = y_valid.replace([np.inf, -np.inf], np.nan)
    
    # Remove problematic rows
    final_mask = np.isfinite(X_clean.values).all(axis=1) & np.isfinite(y_clean.values)
    X_final = X_clean[final_mask]
    y_final = y_clean[final_mask]
    
    print(f"  After cleaning: {len(X_final)} samples")
    
    if len(X_final) < 5:
        print(f"  Too few samples, using mean")
        return None, y_target.mean()
    
    # Cross-validation with fallback
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_final)):
        print(f"    Fold {fold + 1}/{cv_folds}")
        
        X_fold_train = X_final.iloc[train_idx]
        X_fold_val = X_final.iloc[val_idx]
        y_fold_train = y_final.iloc[train_idx]
        y_fold_val = y_final.iloc[val_idx]
        
        # Additional cleaning per fold
        X_fold_train = X_fold_train.replace([np.inf, -np.inf], 0).fillna(0)
        X_fold_val = X_fold_val.replace([np.inf, -np.inf], 0).fillna(0)
        y_fold_train = y_fold_train.replace([np.inf, -np.inf], np.nan).fillna(y_fold_train.median())
        y_fold_val = y_fold_val.replace([np.inf, -np.inf], np.nan).fillna(y_fold_val.median())
        
        # Verify cleanliness
        if not (np.isfinite(X_fold_train.values).all() and np.isfinite(y_fold_train.values).all()):
            print(f"      Data still contains inf/nan, skipping fold")
            continue
        
        # Try XGBoost with fallback
        try:
            model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            model.fit(X_fold_train, y_fold_train)
            print(f"      XGBoost succeeded")
        except Exception as e:
            print(f"      XGBoost failed: {str(e)[:50]}...")
            print(f"      Using RandomForest fallback")
            model = RandomForestRegressor(
                n_estimators=20,
                max_depth=5,
                random_state=42,
                n_jobs=1
            )
            model.fit(X_fold_train, y_fold_train)
        
        # Validate
        val_pred = model.predict(X_fold_val)
        val_score = np.sqrt(mean_squared_error(y_fold_val, val_pred))
        cv_scores.append(val_score)
        print(f"      RMSE: {val_score:.6f}")
    
    if not cv_scores:
        print(f"  All folds failed, using mean")
        return None, y_target.mean()
    
    avg_score = np.mean(cv_scores)
    print(f"  Average CV RMSE: {avg_score:.6f}")
    
    # Train final model on all clean data
    try:
        final_model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        final_model.fit(X_final, y_final)
        print(f"  Final XGBoost model trained")
    except Exception:
        print(f"  Final XGBoost failed, using RandomForest")
        final_model = RandomForestRegressor(
            n_estimators=20,
            max_depth=5,
            random_state=42,
            n_jobs=1
        )
        final_model.fit(X_final, y_final)
    
    return final_model, avg_score


def main():
    print("=" * 60)
    print("ULTRA-FAST BASELINE - TESTING XGBOOST FIX")
    print("=" * 60)
    
    # Load data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Create minimal features
    print("\nCreating minimal features...")
    train_features = ultrafast_features(train_df['SMILES'])
    test_features = ultrafast_features(test_df['SMILES'])
    
    print(f"Features: {train_features.shape[1]} columns")
    
    # Train models
    print("\nTesting robust training...")
    models = {}
    predictions = {}
    
    for target in targets:
        if target in train_df.columns:
            model, score = robust_train_target(train_features, train_df[target], target)
            
            if model is not None:
                pred = model.predict(test_features)
                models[target] = model
                predictions[target] = pred
                print(f"  ✅ {target}: Model trained, score={score:.6f}")
            else:
                predictions[target] = np.full(len(test_features), score)
                models[target] = None
                print(f"  ⚠️ {target}: Used mean prediction={score:.6f}")
        else:
            predictions[target] = np.zeros(len(test_features))
            print(f"  ❌ {target}: Missing column")
    
    # Create submission
    print("\nCreating submission...")
    submission = pd.DataFrame({'id': test_df['id']})
    for target in targets:
        submission[target] = predictions[target]
    
    print("Submission preview:")
    print(submission)
    
    # Save submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'submissions/ultrafast_baseline_{timestamp}.csv'
    
    import os
    os.makedirs('submissions', exist_ok=True)
    submission.to_csv(submission_path, index=False)
    
    print(f"\n" + "=" * 60)
    print("✅ ULTRA-FAST BASELINE COMPLETE!")
    print("=" * 60)
    print("Results:")
    for target in targets:
        if models[target] is not None:
            print(f"  • {target}: Model trained successfully")
        else:
            print(f"  • {target}: Used fallback strategy")
    
    print(f"\nSubmission: {submission_path}")
    print("🚀 XGBoost infinite value handling tested!")


if __name__ == "__main__":
    main()