#!/usr/bin/env python3
"""
Final Working Baseline for NeurIPS 2025 Polymer Prediction
==========================================================

This version properly handles missing values and multi-target prediction.
"""

import argparse
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib
from tqdm import tqdm
import json
from datetime import datetime

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors

warnings.filterwarnings('ignore')


def calculate_molecular_descriptors(smiles_list):
    """Calculate molecular descriptors from SMILES strings."""
    
    # Basic descriptors that work across RDKit versions
    basic_descriptors = {
        'MolWt': Descriptors.MolWt,
        'LogP': Descriptors.MolLogP,
        'NumHDonors': Descriptors.NumHDonors,
        'NumHAcceptors': Descriptors.NumHAcceptors,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'NumAromaticRings': Descriptors.NumAromaticRings,
        'TPSA': Descriptors.TPSA,
        'HeavyAtomCount': Descriptors.HeavyAtomCount,
        'BertzCT': Descriptors.BertzCT,
        'Chi0v': Descriptors.Chi0v,
        'Chi1v': Descriptors.Chi1v,
        'Kappa1': Descriptors.Kappa1,
        'Kappa2': Descriptors.Kappa2,
        'LabuteASA': Descriptors.LabuteASA,
        'NumHeteroatoms': Descriptors.NumHeteroatoms
    }
    
    results = []
    print(f"Calculating molecular descriptors for {len(smiles_list)} molecules...")
    
    for smiles in tqdm(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is not None:
                mol_descriptors = {}
                
                # Calculate basic descriptors
                for desc_name, desc_func in basic_descriptors.items():
                    try:
                        value = desc_func(mol)
                        if np.isinf(value) or np.isnan(value):
                            value = 0.0
                        mol_descriptors[desc_name] = value
                    except Exception:
                        mol_descriptors[desc_name] = 0.0
                
                # Add ring and atom counts
                mol_descriptors['NumRings'] = mol.GetRingInfo().NumRings()
                mol_descriptors['NumAtoms'] = mol.GetNumAtoms()
                mol_descriptors['NumBonds'] = mol.GetNumBonds()
                
                # Basic molecular ratios
                mol_descriptors['MolWt_per_Atom'] = mol_descriptors['MolWt'] / max(mol_descriptors['NumAtoms'], 1)
                mol_descriptors['TPSA_per_Atom'] = mol_descriptors['TPSA'] / max(mol_descriptors['NumAtoms'], 1)
                
                results.append(mol_descriptors)
            else:
                # Invalid SMILES - fill with zeros
                mol_descriptors = {desc: 0.0 for desc in basic_descriptors.keys()}
                mol_descriptors.update({
                    'NumRings': 0, 'NumAtoms': 1, 'NumBonds': 0,
                    'MolWt_per_Atom': 0, 'TPSA_per_Atom': 0
                })
                results.append(mol_descriptors)
                
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            # Fill with zeros for error cases
            mol_descriptors = {desc: 0.0 for desc in basic_descriptors.keys()}
            mol_descriptors.update({
                'NumRings': 0, 'NumAtoms': 1, 'NumBonds': 0,
                'MolWt_per_Atom': 0, 'TPSA_per_Atom': 0
            })
            results.append(mol_descriptors)
    
    feature_df = pd.DataFrame(results)
    print(f"Generated {feature_df.shape[1]} molecular descriptors")
    return feature_df


def train_single_target_models(X_train, y_train, X_test, target_columns, cv_folds=3, random_state=42):
    """Train separate XGBoost models for each target (handles missing values better)."""
    
    print(f"Training separate models for {len(target_columns)} targets")
    
    # XGBoost parameters
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
    
    all_models = {}
    all_cv_scores = {}
    all_predictions = {}
    
    for target in target_columns:
        print(f"\\nTraining model for {target}...")
        
        # Get valid samples for this target
        valid_mask = y_train[target].notna()
        valid_count = valid_mask.sum()
        
        if valid_count < 10:
            print(f"  Skipping {target}: only {valid_count} valid samples")
            all_predictions[target] = np.zeros(len(X_test))
            all_cv_scores[target] = []
            continue
        
        X_train_valid = X_train[valid_mask]
        y_train_valid = y_train[target][valid_mask]
        
        print(f"  Using {valid_count} valid samples")
        
        # Cross-validation for this target
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = []
        test_preds = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_valid)):
            # Split data
            X_fold_train = X_train_valid.iloc[train_idx]
            X_fold_val = X_train_valid.iloc[val_idx]
            y_fold_train = y_train_valid.iloc[train_idx]
            y_fold_val = y_train_valid.iloc[val_idx]
            
            # Train model
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_fold_train, y_fold_train)
            
            # Validate
            val_pred = model.predict(X_fold_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
            cv_scores.append(rmse)
            
            # Test prediction
            test_pred = model.predict(X_test)
            test_preds.append(test_pred)
        
        # Average CV score and test predictions
        avg_cv_score = np.mean(cv_scores)
        avg_test_pred = np.mean(test_preds, axis=0)
        
        print(f"  CV RMSE: {avg_cv_score:.6f}")
        
        # Train final model on all valid data
        final_model = xgb.XGBRegressor(**xgb_params)
        final_model.fit(X_train_valid, y_train_valid)
        final_test_pred = final_model.predict(X_test)
        
        # Store results
        all_models[target] = final_model
        all_cv_scores[target] = cv_scores
        all_predictions[target] = final_test_pred
    
    # Summary
    print(f"\\n" + "="*50)
    print("CROSS-VALIDATION SUMMARY:")
    for target in target_columns:
        if all_cv_scores[target]:
            scores = all_cv_scores[target]
            print(f"  {target}: {np.mean(scores):.6f} ± {np.std(scores):.6f}")
        else:
            print(f"  {target}: No valid data")
    print("="*50)
    
    return all_models, all_cv_scores, all_predictions


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Final NeurIPS 2025 Polymer Prediction Baseline')
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='models/', help='Output directory')
    parser.add_argument('--cv_folds', type=int, default=3, help='Number of CV folds')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    submissions_dir = Path('submissions')
    submissions_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("NeurIPS 2025 Open Polymer Prediction - Final Baseline")
    print("=" * 60)
    
    # Load data
    data_dir = Path(args.data_dir)
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    
    if not train_path.exists() or not test_path.exists():
        print(f"❌ Data files not found in {data_dir}")
        return
    
    print(f"Loading data from {data_dir}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Target columns
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    print(f"Target columns: {target_columns}")
    
    # Show target statistics
    print(f"\\nTarget statistics:")
    y_train = train_df[target_columns]
    for target in target_columns:
        valid_count = y_train[target].notna().sum()
        if valid_count > 0:
            mean_val = y_train[target].mean()
            print(f"  {target}: {valid_count}/{len(y_train)} valid ({valid_count/len(y_train)*100:.1f}%), mean={mean_val:.4f}")
        else:
            print(f"  {target}: No valid values")
    
    # Generate features
    print("\\n" + "="*50)
    print("FEATURE GENERATION")
    print("="*50)
    
    train_features = calculate_molecular_descriptors(train_df['SMILES'].tolist())
    test_features = calculate_molecular_descriptors(test_df['SMILES'].tolist())
    
    # Remove constant features
    constant_cols = []
    for col in train_features.columns:
        if train_features[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"Removing {len(constant_cols)} constant features")
        train_features = train_features.drop(columns=constant_cols)
        test_features = test_features.drop(columns=constant_cols)
    
    print(f"Final feature count: {train_features.shape[1]}")
    
    # Train models
    print("\\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    
    models, cv_scores, predictions = train_single_target_models(
        train_features, y_train, test_features, target_columns, 
        args.cv_folds, args.random_state
    )
    
    # Create submission
    print("\\n" + "="*50)
    print("CREATING SUBMISSION")
    print("="*50)
    
    submission_df = pd.DataFrame({'id': test_df['id']})
    
    for target in target_columns:
        submission_df[target] = predictions[target]
    
    # Save submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = submissions_dir / f'neurips_baseline_submission_{timestamp}.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Submission saved: {submission_path}")
    print("\\nSubmission preview:")
    print(submission_df)
    
    # Save models
    model_artifacts = {
        'models': models,
        'cv_scores': cv_scores,
        'target_columns': target_columns,
        'feature_columns': list(train_features.columns),
        'timestamp': timestamp
    }
    
    model_path = output_dir / f'neurips_models_{timestamp}.pkl'
    joblib.dump(model_artifacts, model_path)
    print(f"\\nModels saved: {model_path}")
    
    # Final summary
    print("\\n" + "="*60)
    print("✅ BASELINE TRAINING COMPLETE!")
    print("="*60)
    print("Key results:")
    for target in target_columns:
        if cv_scores[target]:
            rmse = np.mean(cv_scores[target])
            print(f"  • {target}: CV RMSE = {rmse:.6f}")
        else:
            print(f"  • {target}: No valid training data")
    
    print(f"\\nFiles created:")
    print(f"  • Submission: {submission_path}")
    print(f"  • Models: {model_path}")
    print("\\n🚀 Ready for Kaggle submission!")


if __name__ == "__main__":
    main()