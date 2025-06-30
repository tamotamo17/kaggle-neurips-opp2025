#!/usr/bin/env python3
"""
Fast Baseline for NeurIPS 2025 Polymer Prediction
=================================================

Optimized version that runs quickly while still using molecular descriptors.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from datetime import datetime
import joblib

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors

warnings.filterwarnings('ignore')


def calculate_fast_descriptors(smiles_list, max_samples=1000):
    """Calculate essential molecular descriptors quickly."""
    
    # Limit samples for speed during development
    if len(smiles_list) > max_samples:
        print(f"Sampling {max_samples} molecules for speed...")
        sample_indices = np.random.choice(len(smiles_list), max_samples, replace=False)
        smiles_sample = [smiles_list[i] for i in sample_indices]
    else:
        smiles_sample = smiles_list
        sample_indices = None
    
    # Essential descriptors only
    descriptors = {
        'MolWt': Descriptors.MolWt,
        'LogP': Descriptors.MolLogP,
        'TPSA': Descriptors.TPSA,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'NumAromaticRings': Descriptors.NumAromaticRings,
        'HeavyAtomCount': Descriptors.HeavyAtomCount,
    }
    
    results = []
    print(f"Calculating descriptors for {len(smiles_sample)} molecules...")
    
    for smiles in smiles_sample:
        try:
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is not None:
                mol_desc = {}
                for name, func in descriptors.items():
                    try:
                        val = func(mol)
                        mol_desc[name] = val if not (np.isnan(val) or np.isinf(val)) else 0.0
                    except:
                        mol_desc[name] = 0.0
                
                # Add simple features
                mol_desc['NumAtoms'] = mol.GetNumAtoms()
                mol_desc['NumRings'] = mol.GetRingInfo().NumRings()
                mol_desc['SMILES_Length'] = len(smiles)
                mol_desc['Star_Count'] = smiles.count('*')
                
                results.append(mol_desc)
            else:
                # Invalid SMILES
                mol_desc = {name: 0.0 for name in descriptors.keys()}
                mol_desc.update({'NumAtoms': 1, 'NumRings': 0, 'SMILES_Length': len(smiles), 'Star_Count': smiles.count('*')})
                results.append(mol_desc)
                
        except Exception as e:
            print(f"Error with SMILES {smiles}: {e}")
            mol_desc = {name: 0.0 for name in descriptors.keys()}
            mol_desc.update({'NumAtoms': 1, 'NumRings': 0, 'SMILES_Length': len(smiles), 'Star_Count': smiles.count('*')})
            results.append(mol_desc)
    
    feature_df = pd.DataFrame(results)
    
    # If we sampled, create full dataframe with basic features for others
    if sample_indices is not None:
        print("Creating basic features for remaining molecules...")
        basic_features = []
        for i, smiles in enumerate(smiles_list):
            if i not in sample_indices:
                basic_feat = {name: 0.0 for name in descriptors.keys()}
                basic_feat.update({
                    'NumAtoms': 1,
                    'NumRings': 0,
                    'SMILES_Length': len(smiles),
                    'Star_Count': smiles.count('*')
                })
                basic_features.append(basic_feat)
        
        # Combine sampled descriptors with basic features
        basic_df = pd.DataFrame(basic_features)
        full_indices = list(range(len(smiles_list)))
        
        # Create full dataframe in correct order
        full_results = []
        sample_idx = 0
        basic_idx = 0
        
        for i in range(len(smiles_list)):
            if sample_indices is not None and i in sample_indices:
                full_results.append(results[sample_idx])
                sample_idx += 1
            else:
                full_results.append(basic_features[basic_idx])
                basic_idx += 1
        
        feature_df = pd.DataFrame(full_results)
    
    print(f"Generated {feature_df.shape[1]} features")
    return feature_df


def train_fast_models(X_train, y_train, X_test, targets):
    """Train fast models for each target."""
    
    print(f"Training models for {len(targets)} targets...")
    
    models = {}
    predictions = {}
    scores = {}
    
    for target in targets:
        print(f"\\nTraining {target}...")
        
        # Get valid samples
        valid_mask = y_train[target].notna()
        valid_count = valid_mask.sum()
        
        if valid_count < 10:
            print(f"  Insufficient data ({valid_count} samples), using mean")
            mean_val = y_train[target].mean() if valid_count > 0 else 0.0
            predictions[target] = np.full(len(X_test), mean_val)
            models[target] = None
            scores[target] = None
            continue
        
        X_valid = X_train[valid_mask]
        y_valid = y_train[target][valid_mask]
        
        # Choose model based on data size
        if valid_count < 100:
            model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
        else:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        
        # Train
        model.fit(X_valid, y_valid)
        
        # Quick validation
        train_pred = model.predict(X_valid)
        train_score = np.sqrt(mean_squared_error(y_valid, train_pred))
        
        # Test prediction
        test_pred = model.predict(X_test)
        
        models[target] = model
        predictions[target] = test_pred
        scores[target] = train_score
        
        print(f"  Training RMSE: {train_score:.6f}")
        print(f"  Predictions: [{test_pred[0]:.4f}, {test_pred[1]:.4f}, {test_pred[2]:.4f}]")
    
    return models, predictions, scores


def main():
    print("=" * 60)
    print("FAST BASELINE FOR NEURIPS 2025 POLYMER PREDICTION")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"Targets: {targets}")
    
    # Show target statistics
    print("\\nTarget statistics:")
    y_train = train_df[targets]
    for target in targets:
        valid_count = y_train[target].notna().sum()
        if valid_count > 0:
            mean_val = y_train[target].mean()
            print(f"  {target}: {valid_count}/{len(y_train)} valid ({valid_count/len(y_train)*100:.1f}%), mean={mean_val:.4f}")
        else:
            print(f"  {target}: No valid values")
    
    # Generate features
    print("\\n" + "="*40)
    print("FEATURE GENERATION")
    print("="*40)
    
    # Use sampling for speed on large training set
    train_features = calculate_fast_descriptors(train_df['SMILES'].tolist(), max_samples=2000)
    test_features = calculate_fast_descriptors(test_df['SMILES'].tolist())
    
    print(f"Feature shapes: Train {train_features.shape}, Test {test_features.shape}")
    
    # Train models
    print("\\n" + "="*40)
    print("MODEL TRAINING")
    print("="*40)
    
    models, predictions, scores = train_fast_models(train_features, y_train, test_features, targets)
    
    # Create submission
    print("\\n" + "="*40)
    print("CREATING SUBMISSION")
    print("="*40)
    
    submission = pd.DataFrame({'id': test_df['id']})
    for target in targets:
        submission[target] = predictions[target]
    
    print("Submission preview:")
    print(submission)
    
    # Save files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save submission
    Path('submissions').mkdir(exist_ok=True)
    submission_path = f'submissions/fast_baseline_{timestamp}.csv'
    submission.to_csv(submission_path, index=False)
    
    # Save models
    Path('models').mkdir(exist_ok=True)
    model_artifacts = {
        'models': models,
        'scores': scores,
        'targets': targets,
        'timestamp': timestamp
    }
    model_path = f'models/fast_baseline_models_{timestamp}.pkl'
    joblib.dump(model_artifacts, model_path)
    
    print(f"\\n" + "="*60)
    print("✅ FAST BASELINE COMPLETE!")
    print("="*60)
    print("Results summary:")
    for target in targets:
        if scores[target] is not None:
            print(f"  • {target}: Training RMSE = {scores[target]:.6f}")
        else:
            print(f"  • {target}: Used mean/zero prediction")
    
    print(f"\\nFiles saved:")
    print(f"  • Submission: {submission_path}")
    print(f"  • Models: {model_path}")
    print("\\n🚀 Ready for Kaggle submission!")


if __name__ == "__main__":
    main()