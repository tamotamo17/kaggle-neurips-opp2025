#!/usr/bin/env python3
"""
Fixed Baseline Model for NeurIPS 2025 Polymer Prediction
=======================================================

This script fixes the multi-target prediction and RDKit compatibility issues.
"""

import argparse
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.multioutput import MultiOutputRegressor
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
        'NumSaturatedRings': Descriptors.NumSaturatedRings,
        'NumAliphaticRings': Descriptors.NumAliphaticRings,
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
                mol_descriptors = {'SMILES': smiles}
                
                # Calculate basic descriptors
                for desc_name, desc_func in basic_descriptors.items():
                    try:
                        value = desc_func(mol)
                        if np.isinf(value) or np.isnan(value):
                            value = 0.0
                        mol_descriptors[desc_name] = value
                    except Exception:
                        mol_descriptors[desc_name] = 0.0
                
                # Add custom descriptors
                mol_descriptors['NumRings'] = mol.GetRingInfo().NumRings()
                mol_descriptors['NumAtoms'] = mol.GetNumAtoms()
                mol_descriptors['NumBonds'] = mol.GetNumBonds()
                
                # Safe calculation of FractionCsp3
                try:
                    mol_descriptors['FractionCsp3'] = Descriptors.FractionCsp3(mol)
                except AttributeError:
                    mol_descriptors['FractionCsp3'] = 0.0
                
                # Basic molecular properties
                mol_descriptors['MolWt_per_Atom'] = mol_descriptors['MolWt'] / (mol_descriptors['NumAtoms'] + 1e-6)
                mol_descriptors['TPSA_per_Atom'] = mol_descriptors['TPSA'] / (mol_descriptors['NumAtoms'] + 1e-6)
                mol_descriptors['LogP_per_Atom'] = mol_descriptors['LogP'] / (mol_descriptors['NumAtoms'] + 1e-6)
                mol_descriptors['HeavyAtom_Ratio'] = mol_descriptors['HeavyAtomCount'] / (mol_descriptors['NumAtoms'] + 1e-6)
                
                results.append(mol_descriptors)
            else:
                # Invalid SMILES - fill with zeros
                invalid_mol = {'SMILES': smiles}
                for desc_name in basic_descriptors.keys():
                    invalid_mol[desc_name] = 0.0
                # Add zero values for custom descriptors
                for key in ['NumRings', 'NumAtoms', 'NumBonds', 'FractionCsp3', 
                           'MolWt_per_Atom', 'TPSA_per_Atom', 'LogP_per_Atom', 'HeavyAtom_Ratio']:
                    invalid_mol[key] = 0.0
                results.append(invalid_mol)
                
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            continue
    
    feature_df = pd.DataFrame(results)
    print(f"Generated {feature_df.shape[1]-1} molecular descriptors")
    return feature_df


def preprocess_features(X_train, X_test):
    """Remove constant and highly correlated features."""
    print(f"Initial shapes: Train {X_train.shape}, Test {X_test.shape}")
    
    # Remove SMILES column
    if 'SMILES' in X_train.columns:
        X_train = X_train.drop('SMILES', axis=1)
        X_test = X_test.drop('SMILES', axis=1)
    
    # Remove constant features
    constant_features = []
    for col in X_train.columns:
        if X_train[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        print(f"Removing {len(constant_features)} constant features")
        X_train = X_train.drop(columns=constant_features)
        X_test = X_test.drop(columns=constant_features)
    
    # Remove highly correlated features
    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    
    if to_drop:
        print(f"Removing {len(to_drop)} highly correlated features")
        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop)
    
    # Handle infinite and NaN values
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"After preprocessing: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test


def train_multi_target_model(X_train, y_train, X_test, cv_folds=5, random_state=42):
    """Train multi-target XGBoost model."""
    
    print(f"Training multi-target model for {y_train.shape[1]} targets")
    print(f"Target columns: {list(y_train.columns)}")
    
    # XGBoost parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # Create multi-output regressor
    model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
    
    # Cross-validation
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    cv_scores = {target: [] for target in y_train.columns}
    oof_predictions = pd.DataFrame(index=y_train.index, columns=y_train.columns)
    test_predictions = pd.DataFrame(index=range(len(X_test)), columns=y_train.columns)
    
    print("Starting cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"\\nFold {fold + 1}/{cv_folds}")
        
        # Split data
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train model
        model.fit(X_fold_train, y_fold_train)
        
        # Validation predictions
        val_pred = model.predict(X_fold_val)
        val_pred_df = pd.DataFrame(val_pred, columns=y_train.columns, index=val_idx)
        oof_predictions.loc[val_idx] = val_pred_df.values
        
        # Test predictions
        test_pred = model.predict(X_test)
        test_pred_df = pd.DataFrame(test_pred, columns=y_train.columns)
        
        if fold == 0:
            test_predictions = test_pred_df / cv_folds
        else:
            test_predictions += test_pred_df / cv_folds
        
        # Calculate metrics for each target
        for target in y_train.columns:
            if not y_fold_val[target].isna().all():  # Only if target has valid values
                valid_mask = ~y_fold_val[target].isna()
                if valid_mask.sum() > 0:
                    rmse = np.sqrt(mean_squared_error(
                        y_fold_val[target][valid_mask], 
                        val_pred_df[target][valid_mask]
                    ))
                    cv_scores[target].append(rmse)
                    print(f"  {target} RMSE: {rmse:.6f}")
    
    # Overall CV performance
    print(f"\\n" + "="*50)
    print(f"CROSS-VALIDATION RESULTS:")
    for target in y_train.columns:
        if cv_scores[target]:
            mean_rmse = np.mean(cv_scores[target])
            std_rmse = np.std(cv_scores[target])
            print(f"  {target}: {mean_rmse:.6f} ± {std_rmse:.6f}")
        else:
            print(f"  {target}: No valid data")
    print(f"="*50)
    
    # Train final model on all data
    print("\\nTraining final model on all data...")
    final_model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
    
    # Only use targets with valid data
    valid_targets = []
    for target in y_train.columns:
        if not y_train[target].isna().all():
            valid_targets.append(target)
    
    if valid_targets:
        final_model.fit(X_train, y_train[valid_targets])
        final_predictions = final_model.predict(X_test)
        final_predictions_df = pd.DataFrame(final_predictions, columns=valid_targets)
        
        # Fill missing targets with zeros
        for target in y_train.columns:
            if target not in valid_targets:
                final_predictions_df[target] = 0.0
        
        # Reorder columns to match original order
        final_predictions_df = final_predictions_df[y_train.columns]
    else:
        print("Warning: No valid targets found. Using zeros for all predictions.")
        final_predictions_df = pd.DataFrame(0.0, index=range(len(X_test)), columns=y_train.columns)
    
    return cv_scores, oof_predictions, final_predictions_df, final_model


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Fixed NeurIPS 2025 Polymer Prediction Baseline')
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='models/', help='Output directory')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    submissions_dir = Path('submissions')
    submissions_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("NeurIPS 2025 Open Polymer Prediction - Fixed Baseline")
    print("=" * 60)
    
    # Load data
    data_dir = Path(args.data_dir)
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    
    if not train_path.exists() or not test_path.exists():
        print(f"❌ Data files not found in {data_dir}")
        print("Please ensure train.csv and test.csv are in the data directory")
        return
    
    print(f"Loading data from {data_dir}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Train columns: {list(train_df.columns)}")
    
    # Identify target columns (all except id and SMILES)
    target_columns = [col for col in train_df.columns if col not in ['id', 'SMILES']]
    print(f"Target columns: {target_columns}")
    
    # Generate molecular features
    print("\\n" + "="*50)
    print("FEATURE GENERATION")
    print("="*50)
    
    train_features = calculate_molecular_descriptors(train_df['SMILES'].tolist())
    test_features = calculate_molecular_descriptors(test_df['SMILES'].tolist())
    
    # Preprocess features
    X_train, X_test = preprocess_features(train_features, test_features)
    
    # Prepare targets
    y_train = train_df[target_columns]
    
    # Show target statistics
    print(f"\\nTarget statistics:")
    for target in target_columns:
        valid_count = y_train[target].notna().sum()
        if valid_count > 0:
            print(f"  {target}: {valid_count} valid values, mean={y_train[target].mean():.4f}")
        else:
            print(f"  {target}: No valid values")
    
    # Train model
    print("\\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    
    cv_scores, oof_predictions, test_predictions, model = train_multi_target_model(
        X_train, y_train, X_test, args.cv_folds, args.random_state
    )
    
    # Create submission
    print("\\n" + "="*50)
    print("CREATING SUBMISSION")
    print("="*50)
    
    submission_df = pd.DataFrame({
        'id': test_df['id']
    })
    
    # Add predictions for each target
    for target in target_columns:
        submission_df[target] = test_predictions[target]
    
    # Save submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = submissions_dir / f'fixed_baseline_submission_{timestamp}.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Submission saved to {submission_path}")
    print(f"\\nSubmission preview:")
    print(submission_df.head())
    
    # Save model artifacts
    model_artifacts = {
        'model': model,
        'cv_scores': cv_scores,
        'target_columns': target_columns,
        'feature_columns': list(X_train.columns),
        'timestamp': timestamp
    }
    
    model_path = output_dir / f'fixed_baseline_model_{timestamp}.pkl'
    joblib.dump(model_artifacts, model_path)
    print(f"Model saved to {model_path}")
    
    print("\\n" + "="*60)
    print("BASELINE TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()