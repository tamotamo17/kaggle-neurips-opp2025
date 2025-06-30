#!/usr/bin/env python3
"""
Working Baseline - Simple approach that definitely works
=====================================================

This baseline uses only basic Python libraries to avoid dependency issues.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def simple_features(smiles_list):
    """Create very simple string-based features."""
    features = []
    
    for smiles in smiles_list:
        feat = {
            'length': len(smiles),
            'carbon_count': smiles.count('C') + smiles.count('c'),
            'oxygen_count': smiles.count('O') + smiles.count('o'),
            'nitrogen_count': smiles.count('N') + smiles.count('n'),
            'star_count': smiles.count('*'),
            'ring_count': smiles.count('c'),
            'branch_count': smiles.count('('),
            'double_bond_count': smiles.count('='),
            'ring_digit_count': sum(1 for c in smiles if c.isdigit()),
            'complexity': len(set(smiles))
        }
        features.append(feat)
    
    return pd.DataFrame(features)


def simple_predict(X_train, y_target, X_test):
    """Simple prediction using mean/median strategies."""
    valid_mask = pd.notna(y_target)
    valid_count = valid_mask.sum()
    
    if valid_count < 5:
        # Use zero if insufficient data
        return np.zeros(len(X_test))
    
    # Use mean of valid values
    mean_val = y_target[valid_mask].mean()
    return np.full(len(X_test), mean_val)


def main():
    print("=" * 60)
    print("WORKING BASELINE - SIMPLE AND RELIABLE")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Show target info
    print("\nTarget statistics:")
    for target in targets:
        if target in train_df.columns:
            valid = train_df[target].notna().sum()
            mean_val = train_df[target].mean() if valid > 0 else 0
            print(f"  {target}: {valid}/{len(train_df)} valid, mean={mean_val:.4f}")
    
    # Create simple features
    print("\nCreating simple features...")
    train_features = simple_features(train_df['SMILES'])
    test_features = simple_features(test_df['SMILES'])
    
    print(f"Features: {train_features.shape[1]} columns")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = {}
    
    for target in targets:
        if target in train_df.columns:
            pred = simple_predict(train_features, train_df[target], test_features)
            predictions[target] = pred
            print(f"  {target}: mean prediction = {np.mean(pred):.4f}")
        else:
            predictions[target] = np.zeros(len(test_features))
            print(f"  {target}: missing, using zeros")
    
    # Create submission
    print("\nCreating submission...")
    submission = pd.DataFrame({'id': test_df['id']})
    for target in targets:
        submission[target] = predictions[target]
    
    print("Submission preview:")
    print(submission.head())
    
    # Save submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'submissions/working_baseline_{timestamp}.csv'
    
    # Create submissions directory if it doesn't exist
    import os
    os.makedirs('submissions', exist_ok=True)
    
    submission.to_csv(submission_path, index=False)
    
    print(f"\n" + "=" * 60)
    print("✅ WORKING BASELINE COMPLETE!")
    print("=" * 60)
    print(f"Submission saved: {submission_path}")
    print(f"Shape: {submission.shape}")
    print("Ready for Kaggle!")
    print("=" * 60)


if __name__ == "__main__":
    main()