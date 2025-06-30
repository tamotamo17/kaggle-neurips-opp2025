#!/usr/bin/env python3
"""
Simple Fixed Baseline - Just Fix the NaN Error
==============================================

This modifies the original baseline to handle multi-target with NaN values.
"""

import sys
sys.path.append('src')

from baseline_model import PolymerPredictor, load_competition_data
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from pathlib import Path
import joblib
from datetime import datetime


def train_single_target_xgb(X_train, y_target, X_test, target_name, random_state=42):
    """Train a single XGBoost model for one target, handling NaN values."""
    
    print(f"\\nTraining {target_name}...")
    
    # Handle NaN values in target
    valid_mask = y_target.notna()
    valid_count = valid_mask.sum()
    
    print(f"  Valid samples: {valid_count}/{len(y_target)} ({valid_count/len(y_target)*100:.1f}%)")
    
    if valid_count < 10:
        print(f"  Insufficient data, using mean prediction")
        mean_val = y_target.mean() if valid_count > 0 else 0.0
        predictions = np.full(len(X_test), mean_val)
        return None, predictions, None
    
    # Get valid data
    X_train_valid = X_train[valid_mask]
    y_train_valid = y_target[valid_mask]
    
    # Train XGBoost
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state,
        'verbosity': 0,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_valid, y_train_valid)
    
    # Validate
    train_pred = model.predict(X_train_valid)
    train_rmse = np.sqrt(mean_squared_error(y_train_valid, train_pred))
    
    # Test predictions
    test_pred = model.predict(X_test)
    
    print(f"  Training RMSE: {train_rmse:.6f}")
    
    return model, test_pred, train_rmse


def main():
    print("=" * 60)
    print("SIMPLE FIXED BASELINE - NO MORE NaN ERRORS")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    train_df, test_df = load_competition_data('data')
    if train_df is None:
        print("Failed to load data")
        return
    
    # Define targets explicitly
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    smiles_col = 'SMILES'
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Target columns: {target_columns}")
    
    # Check target data
    print(f"\\nTarget statistics:")
    for target in target_columns:
        valid_count = train_df[target].notna().sum()
        if valid_count > 0:
            mean_val = train_df[target].mean()
            print(f"  {target}: {valid_count} valid, mean={mean_val:.4f}")
    
    # Generate features using the original PolymerPredictor
    print("\\n" + "="*50)
    print("FEATURE GENERATION") 
    print("="*50)
    
    predictor = PolymerPredictor()
    
    # Generate molecular features
    train_features = predictor.calculate_molecular_descriptors(
        train_df[smiles_col].tolist(), 'comprehensive'
    )
    test_features = predictor.calculate_molecular_descriptors(
        test_df[smiles_col].tolist(), 'comprehensive'
    )
    
    # Feature engineering
    train_features_eng = predictor.engineer_features(train_features)
    test_features_eng = predictor.engineer_features(test_features)
    
    # Preprocess features
    X_train, X_test = predictor.preprocess_features(train_features_eng, test_features_eng)
    
    print(f"Final features: {X_train.shape[1]}")
    
    # Train separate models for each target
    print("\\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    models = {}
    predictions = {}
    scores = {}
    
    for target in target_columns:
        model, pred, rmse = train_single_target_xgb(
            X_train, train_df[target], X_test, target, random_state=42
        )
        
        models[target] = model
        predictions[target] = pred
        scores[target] = rmse
    
    # Create submission
    print("\\n" + "="*50)
    print("CREATING SUBMISSION")
    print("="*50)
    
    submission = pd.DataFrame({'id': test_df['id']})
    for target in target_columns:
        submission[target] = predictions[target]
    
    print("Submission preview:")
    print(submission)
    
    # Save files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save submission
    Path('submissions').mkdir(exist_ok=True)
    submission_path = f'submissions/fixed_baseline_{timestamp}.csv'
    submission.to_csv(submission_path, index=False)
    
    # Save models
    Path('models').mkdir(exist_ok=True)
    artifacts = {
        'models': models,
        'scores': scores,
        'target_columns': target_columns,
        'timestamp': timestamp
    }
    model_path = f'models/fixed_baseline_{timestamp}.pkl'
    joblib.dump(artifacts, model_path)
    
    print(f"\\n" + "="*60)
    print("✅ FIXED BASELINE COMPLETE!")
    print("="*60)
    print("Model performance:")
    for target in target_columns:
        if scores[target] is not None:
            print(f"  • {target}: RMSE = {scores[target]:.6f}")
        else:
            print(f"  • {target}: Used mean prediction")
    
    print(f"\\nFiles saved:")
    print(f"  • Submission: {submission_path}")
    print(f"  • Models: {model_path}")
    print("\\n🚀 No more NaN errors!")


if __name__ == "__main__":
    main()