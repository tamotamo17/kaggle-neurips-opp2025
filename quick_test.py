#!/usr/bin/env python3
"""
Quick test version to verify the pipeline works
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 50)
    print("QUICK PIPELINE TEST")
    print("=" * 50)
    
    # Load data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Target columns
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Quick features - just use string length as a proxy
    train_features = pd.DataFrame({
        'smiles_length': train_df['SMILES'].str.len(),
        'smiles_star_count': train_df['SMILES'].str.count('\\*'),
        'smiles_ring_count': train_df['SMILES'].str.count('c'),
        'smiles_carbon_count': train_df['SMILES'].str.count('C'),
    })
    
    test_features = pd.DataFrame({
        'smiles_length': test_df['SMILES'].str.len(),
        'smiles_star_count': test_df['SMILES'].str.count('\\*'),
        'smiles_ring_count': test_df['SMILES'].str.count('c'),
        'smiles_carbon_count': test_df['SMILES'].str.count('C'),
    })
    
    print(f"Features: {train_features.shape[1]}")
    
    # Train simple models for each target
    predictions = {}
    
    for target in targets:
        valid_mask = train_df[target].notna()
        valid_count = valid_mask.sum()
        
        print(f"\\n{target}: {valid_count} valid samples")
        
        if valid_count > 10:
            X = train_features[valid_mask]
            y = train_df[target][valid_mask]
            
            # Simple model
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Test prediction
            pred = model.predict(test_features)
            predictions[target] = pred
            
            print(f"  Model trained, predictions: {pred}")
        else:
            predictions[target] = np.zeros(len(test_features))
            print(f"  Insufficient data, using zeros")
    
    # Create submission
    submission = pd.DataFrame({'id': test_df['id']})
    for target in targets:
        submission[target] = predictions[target]
    
    print(f"\\nSubmission:")
    print(submission)
    
    # Save
    submission.to_csv('submissions/quick_test_submission.csv', index=False)
    print(f"\\n✅ Quick test complete! Submission saved.")

if __name__ == "__main__":
    main()