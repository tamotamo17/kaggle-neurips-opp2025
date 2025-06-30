#!/usr/bin/env python3
"""
NeurIPS 2025 Open Polymer Prediction - Enhanced Baseline Model
==============================================================

This script implements a comprehensive baseline model for the competition using:
- RDKit for molecular descriptor generation
- XGBoost for machine learning
- Advanced feature engineering
- Robust cross-validation
- Automated submission generation

Usage:
    python src/baseline_model.py --data_dir data/ --output_dir models/

Author: Enhanced baseline based on public competition approaches
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import joblib
from tqdm import tqdm
import json
from datetime import datetime

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt, MolLogP, NumHDonors, NumHAcceptors

warnings.filterwarnings('ignore')


class PolymerPredictor:
    """
    Enhanced baseline model for polymer property prediction.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = []
        self.feature_selector = None
        self.selected_features = None
        self.cv_scores = []
        self.oof_predictions = None
        self.feature_importance = None
        
        # XGBoost parameters optimized for molecular property prediction
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'random_state': random_state,
            'n_jobs': -1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'early_stopping_rounds': 100,
            'verbose': False
        }
    
    def calculate_molecular_descriptors(self, smiles_list, descriptor_set='comprehensive'):
        """
        Calculate molecular descriptors from SMILES strings using RDKit.
        
        Parameters:
        -----------
        smiles_list : list
            List of SMILES strings
        descriptor_set : str
            Set of descriptors to calculate ('basic', 'comprehensive', 'all')
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with molecular descriptors
        """
        
        # Define descriptor sets
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
            'NumAliphaticRings': Descriptors.NumAliphaticRings
        }
        
        comprehensive_descriptors = {
            **basic_descriptors,
            'BertzCT': Descriptors.BertzCT,
            'Ipc': Descriptors.Ipc,
            'Chi0v': Descriptors.Chi0v,
            'Chi1v': Descriptors.Chi1v,
            'Chi2v': Descriptors.Chi2v,
            'Chi3v': Descriptors.Chi3v,
            'Chi4v': Descriptors.Chi4v,
            'Kappa1': Descriptors.Kappa1,
            'Kappa2': Descriptors.Kappa2,
            'Kappa3': Descriptors.Kappa3,
            'LabuteASA': Descriptors.LabuteASA,
            'BalabanJ': Descriptors.BalabanJ,
            'PEOE_VSA1': Descriptors.PEOE_VSA1,
            'PEOE_VSA2': Descriptors.PEOE_VSA2,
            'PEOE_VSA3': Descriptors.PEOE_VSA3,
            'SMR_VSA1': Descriptors.SMR_VSA1,
            'SMR_VSA2': Descriptors.SMR_VSA2,
            'SMR_VSA3': Descriptors.SMR_VSA3,
            'SlogP_VSA1': Descriptors.SlogP_VSA1,
            'SlogP_VSA2': Descriptors.SlogP_VSA2,
            'SlogP_VSA3': Descriptors.SlogP_VSA3
        }
        
        # Select descriptor set
        if descriptor_set == 'basic':
            descriptors = basic_descriptors
        elif descriptor_set == 'comprehensive':
            descriptors = comprehensive_descriptors
        else:  # 'all'
            # Get all available descriptors
            descriptors = {name: getattr(Descriptors, name) for name in dir(Descriptors) 
                          if not name.startswith('_') and callable(getattr(Descriptors, name))}
        
        results = []
        
        print(f"Calculating {len(descriptors)} descriptors for {len(smiles_list)} molecules...")
        
        for smiles in tqdm(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                
                if mol is not None:
                    # Calculate descriptors
                    mol_descriptors = {'SMILES': smiles}
                    
                    for desc_name, desc_func in descriptors.items():
                        try:
                            value = desc_func(mol)
                            # Handle infinite or NaN values
                            if np.isinf(value) or np.isnan(value):
                                value = 0.0
                            mol_descriptors[desc_name] = value
                        except Exception:
                            mol_descriptors[desc_name] = 0.0
                    
                    # Add custom descriptors
                    mol_descriptors['NumRings'] = mol.GetRingInfo().NumRings()
                    mol_descriptors['NumAtoms'] = mol.GetNumAtoms()
                    mol_descriptors['NumBonds'] = mol.GetNumBonds()
                    mol_descriptors['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
                    # Check if FractionCsp3 exists (different RDKit versions)
                    try:
                        mol_descriptors['FractionCsp3'] = Descriptors.FractionCsp3(mol)
                    except AttributeError:
                        mol_descriptors['FractionCsp3'] = 0.0
                    
                    # Lipinski's Rule of Five
                    mol_descriptors['Lipinski_Violations'] = (
                        (mol_descriptors['MolWt'] > 500) +
                        (mol_descriptors['LogP'] > 5) +
                        (mol_descriptors['NumHDonors'] > 5) +
                        (mol_descriptors['NumHAcceptors'] > 10)
                    )
                    
                    results.append(mol_descriptors)
                else:
                    # Invalid SMILES
                    invalid_mol = {'SMILES': smiles}
                    for desc_name in descriptors.keys():
                        invalid_mol[desc_name] = 0.0
                    results.append(invalid_mol)
                    
            except Exception as e:
                print(f"Error processing SMILES {smiles}: {e}")
                continue
        
        feature_df = pd.DataFrame(results)
        print(f"Generated {feature_df.shape[1]-1} molecular descriptors")
        
        return feature_df
    
    def engineer_features(self, features_df):
        """
        Apply advanced feature engineering techniques.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame with molecular descriptors
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with engineered features
        """
        df = features_df.copy()
        
        # Remove SMILES column for feature engineering
        if 'SMILES' in df.columns:
            df = df.drop('SMILES', axis=1)
        
        # Create ratio features
        df['MolWt_per_Atom'] = df['MolWt'] / (df['NumAtoms'] + 1e-6)
        df['TPSA_per_Atom'] = df['TPSA'] / (df['NumAtoms'] + 1e-6)
        df['LogP_per_Atom'] = df['LogP'] / (df['NumAtoms'] + 1e-6)
        df['HeavyAtom_Ratio'] = df['HeavyAtomCount'] / (df['NumAtoms'] + 1e-6)
        df['Aromatic_Ratio'] = df['NumAromaticRings'] / (df['NumRings'] + 1e-6)
        df['Rotatable_Ratio'] = df['NumRotatableBonds'] / (df['NumBonds'] + 1e-6)
        df['HBond_Ratio'] = (df['NumHDonors'] + df['NumHAcceptors']) / (df['NumAtoms'] + 1e-6)
        
        # Create interaction features
        df['LogP_TPSA'] = df['LogP'] * df['TPSA']
        df['MolWt_LogP'] = df['MolWt'] * df['LogP']
        df['Rings_Aromatic'] = df['NumRings'] * df['NumAromaticRings']
        df['HBond_TPSA'] = (df['NumHDonors'] + df['NumHAcceptors']) * df['TPSA']
        
        # Create polynomial features for key descriptors
        key_features = ['MolWt', 'LogP', 'TPSA', 'NumRotatableBonds']
        for feat in key_features:
            if feat in df.columns:
                df[f'{feat}_squared'] = df[feat] ** 2
                df[f'{feat}_sqrt'] = np.sqrt(np.abs(df[feat]))
                df[f'{feat}_log'] = np.log1p(np.abs(df[feat]))
        
        # Create binned features
        df['MolWt_bin'] = pd.cut(df['MolWt'], bins=10, labels=False)
        df['LogP_bin'] = pd.cut(df['LogP'], bins=10, labels=False)
        df['TPSA_bin'] = pd.cut(df['TPSA'], bins=10, labels=False)
        
        # Handle infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        print(f"Feature engineering complete. New shape: {df.shape}")
        return df
    
    def preprocess_features(self, X_train, X_test, n_features=200):
        """
        Preprocess features: remove constant/correlated features and select best ones.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        n_features : int
            Number of features to select
        
        Returns:
        --------
        tuple
            Processed training and test features
        """
        print(f"Initial feature shapes: Train {X_train.shape}, Test {X_test.shape}")
        
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
        
        print(f"After feature removal: Train {X_train.shape}, Test {X_test.shape}")
        
        return X_train, X_test
    
    def select_features(self, X_train, y_train, X_test, n_features=200):
        """
        Select best features using statistical tests.
        Handle multi-target data with missing values.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.DataFrame or pd.Series
            Training target(s) - can be multi-target with NaN values
        X_test : pd.DataFrame
            Test features
        n_features : int
            Number of features to select
        
        Returns:
        --------
        tuple
            Selected training and test features
        """
        n_features = min(n_features, X_train.shape[1])
        
        # Handle multi-target data with missing values
        if isinstance(y_train, pd.DataFrame):
            # Multi-target case: use the target with most valid data
            valid_counts = y_train.notna().sum()
            best_target = valid_counts.idxmax()
            y_for_selection = y_train[best_target]
            print(f"Using {best_target} for feature selection ({valid_counts[best_target]} valid samples)")
        else:
            # Single target case
            y_for_selection = y_train
        
        # Remove rows with NaN in the target used for selection
        valid_mask = y_for_selection.notna()
        X_train_valid = X_train[valid_mask]
        y_train_valid = y_for_selection[valid_mask]
        
        print(f"Feature selection using {len(y_train_valid)} valid samples out of {len(y_train)}")
        
        if len(y_train_valid) < 10:
            print(f"Warning: Only {len(y_train_valid)} valid samples for feature selection")
            print("Using all features (no selection)")
            self.selected_features = X_train.columns
            return X_train, X_test
        
        # Perform feature selection on valid data
        self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
        X_train_selected = self.feature_selector.fit_transform(X_train_valid, y_train_valid)
        
        # Apply same selection to full datasets
        X_test_selected = self.feature_selector.transform(X_test)
        X_train_full_selected = self.feature_selector.transform(X_train)
        
        # Get selected feature names
        self.selected_features = X_train.columns[self.feature_selector.get_support()]
        print(f"Selected {len(self.selected_features)} features using statistical tests")
        
        # Convert back to DataFrame
        X_train_final = pd.DataFrame(X_train_full_selected, columns=self.selected_features, index=X_train.index)
        X_test_final = pd.DataFrame(X_test_selected, columns=self.selected_features, index=X_test.index)
        
        return X_train_final, X_test_final
    
    def train_cv(self, X_train, y_train, cv_folds=5):
        """
        Train model using cross-validation with robust infinite value handling.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        cv_folds : int
            Number of CV folds
        
        Returns:
        --------
        tuple
            CV scores and out-of-fold predictions
        """
        # Comprehensive data cleaning before CV
        print("Performing comprehensive data cleaning for CV...")
        
        # Clean features
        X_clean = X_train.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(0)
        
        # Clean target
        y_clean = y_train.copy()
        y_clean = y_clean.replace([np.inf, -np.inf], np.nan)
        
        # Remove any remaining problematic rows
        valid_mask = (
            np.isfinite(X_clean.values).all(axis=1) & 
            np.isfinite(y_clean.values)
        )
        
        X_final = X_clean[valid_mask]
        y_final = y_clean[valid_mask]
        
        print(f"After cleaning: {len(X_final)}/{len(X_train)} samples remain")
        
        if len(X_final) < 10:
            print("Too few samples after cleaning, returning dummy results")
            return [0.5], np.zeros(len(X_train))
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        self.cv_scores = []
        self.models = []
        self.oof_predictions = np.zeros(len(X_final))
        
        print("Starting cross-validation training...")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_final)):
            print(f"\\nFold {fold + 1}/{cv_folds}")
            
            # Split data
            X_fold_train, X_fold_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
            y_fold_train, y_fold_val = y_final.iloc[train_idx], y_final.iloc[val_idx]
            
            # Aggressive per-fold cleaning
            print(f"    Cleaning fold data...")
            
            # Replace infinite values multiple times to catch edge cases
            for i in range(3):  # Multiple passes
                X_fold_train = X_fold_train.replace([np.inf, -np.inf], np.nan)
                X_fold_val = X_fold_val.replace([np.inf, -np.inf], np.nan)
                
            # Fill NaN with median/mean, not zero for better stability
            for col in X_fold_train.columns:
                median_val = X_fold_train[col].median()
                if not np.isfinite(median_val):
                    median_val = 0.0
                X_fold_train[col] = X_fold_train[col].fillna(median_val)
                X_fold_val[col] = X_fold_val[col].fillna(median_val)
            
            # Clean targets
            y_fold_train = y_fold_train.replace([np.inf, -np.inf], np.nan)
            y_fold_val = y_fold_val.replace([np.inf, -np.inf], np.nan)
            
            target_median = y_fold_train.median()
            if not np.isfinite(target_median):
                target_median = 0.0
                
            y_fold_train = y_fold_train.fillna(target_median)
            y_fold_val = y_fold_val.fillna(target_median)
            
            # Clip extremely large values
            X_fold_train = X_fold_train.clip(-1e6, 1e6)
            X_fold_val = X_fold_val.clip(-1e6, 1e6)
            y_fold_train = y_fold_train.clip(-1e6, 1e6)
            y_fold_val = y_fold_val.clip(-1e6, 1e6)
            
            # Final verification with detailed error reporting
            inf_check = np.isinf(X_fold_train.values)
            if inf_check.any():
                print(f"    ❌ X_train still has {inf_check.sum()} infinite values")
                # Replace any remaining with 0
                X_fold_train = pd.DataFrame(
                    np.where(np.isinf(X_fold_train.values), 0, X_fold_train.values),
                    columns=X_fold_train.columns,
                    index=X_fold_train.index
                )
            
            inf_check = np.isinf(X_fold_val.values)
            if inf_check.any():
                print(f"    ❌ X_val still has {inf_check.sum()} infinite values")
                X_fold_val = pd.DataFrame(
                    np.where(np.isinf(X_fold_val.values), 0, X_fold_val.values),
                    columns=X_fold_val.columns,
                    index=X_fold_val.index
                )
            
            if np.isinf(y_fold_train.values).any():
                print(f"    ❌ y_train still has infinite values")
                y_fold_train = pd.Series(
                    np.where(np.isinf(y_fold_train.values), target_median, y_fold_train.values),
                    index=y_fold_train.index
                )
            
            if np.isinf(y_fold_val.values).any():
                print(f"    ❌ y_val still has infinite values")
                y_fold_val = pd.Series(
                    np.where(np.isinf(y_fold_val.values), target_median, y_fold_val.values),
                    index=y_fold_val.index
                )
            
            # Final verification
            assert np.isfinite(X_fold_train.values).all(), f"Fold {fold}: X_train STILL contains inf/nan after aggressive cleaning!"
            assert np.isfinite(X_fold_val.values).all(), f"Fold {fold}: X_val STILL contains inf/nan after aggressive cleaning!"
            assert np.isfinite(y_fold_train.values).all(), f"Fold {fold}: y_train STILL contains inf/nan after aggressive cleaning!"
            assert np.isfinite(y_fold_val.values).all(), f"Fold {fold}: y_val STILL contains inf/nan after aggressive cleaning!"
            
            print(f"    ✅ Fold data cleaned successfully")
            
            # Train model with fallback
            try:
                model = xgb.XGBRegressor(**self.xgb_params)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False
                )
            except Exception as e:
                print(f"  XGBoost failed: {str(e)[:100]}...")
                print(f"  Trying RandomForest...")
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(
                        n_estimators=50,
                        max_depth=8,
                        random_state=self.random_state,
                        n_jobs=1  # Reduce parallelism to avoid issues
                    )
                    model.fit(X_fold_train, y_fold_train)
                    print(f"  ✅ RandomForest succeeded")
                except Exception as e2:
                    print(f"  RandomForest also failed: {str(e2)[:100]}...")
                    print(f"  Using simple Linear Regression fallback...")
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model.fit(X_fold_train, y_fold_train)
                    print(f"  ✅ Linear Regression succeeded")
            
            # Predictions
            val_pred = model.predict(X_fold_val)
            self.oof_predictions[val_idx] = val_pred
            self.models.append(model)
            
            # Calculate metrics
            fold_rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
            fold_mae = mean_absolute_error(y_fold_val, val_pred)
            fold_r2 = r2_score(y_fold_val, val_pred)
            
            self.cv_scores.append(fold_rmse)
            
            print(f"  RMSE: {fold_rmse:.6f}")
            print(f"  MAE: {fold_mae:.6f}")
            print(f"  R²: {fold_r2:.6f}")
        
        # Calculate overall performance
        cv_rmse = np.mean(self.cv_scores)
        cv_std = np.std(self.cv_scores)
        oof_rmse = np.sqrt(mean_squared_error(y_final, self.oof_predictions))
        oof_mae = mean_absolute_error(y_final, self.oof_predictions)
        oof_r2 = r2_score(y_final, self.oof_predictions)
        
        print(f"\\n" + "="*50)
        print(f"CROSS-VALIDATION RESULTS:")
        print(f"  CV RMSE: {cv_rmse:.6f} ± {cv_std:.6f}")
        print(f"  OOF RMSE: {oof_rmse:.6f}")
        print(f"  OOF MAE: {oof_mae:.6f}")
        print(f"  OOF R²: {oof_r2:.6f}")
        print(f"="*50)
        
        # Return full-sized OOF array for compatibility
        full_oof = np.zeros(len(X_train))
        if len(X_final) > 0:
            full_oof[valid_mask] = self.oof_predictions
        
        return self.cv_scores, full_oof
    
    def predict(self, X_test):
        """
        Make predictions on test data using ensemble of CV models.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        
        Returns:
        --------
        np.ndarray
            Test predictions
        """
        if not self.models:
            raise ValueError("Models not trained yet. Call train_cv first.")
        
        test_predictions = np.zeros(len(X_test))
        
        for model in self.models:
            test_pred = model.predict(X_test)
            test_predictions += test_pred / len(self.models)
        
        return test_predictions
    
    def get_feature_importance(self):
        """
        Calculate average feature importance across all CV models.
        
        Returns:
        --------
        pd.DataFrame
            Feature importance DataFrame
        """
        if not self.models:
            raise ValueError("Models not trained yet.")
        
        self.feature_importance = pd.DataFrame({
            'feature': self.selected_features,
            'importance': np.mean([model.feature_importances_ for model in self.models], axis=0)
        }).sort_values('importance', ascending=False)
        
        return self.feature_importance
    
    def save_model(self, filepath):
        """
        Save trained model and artifacts.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to save the model
        """
        model_artifacts = {
            'models': self.models,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features.tolist() if self.selected_features is not None else None,
            'cv_scores': self.cv_scores,
            'oof_predictions': self.oof_predictions,
            'feature_importance': self.feature_importance,
            'xgb_params': self.xgb_params,
            'random_state': self.random_state
        }
        
        joblib.dump(model_artifacts, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model and artifacts.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to load the model from
        """
        model_artifacts = joblib.load(filepath)
        
        self.models = model_artifacts['models']
        self.feature_selector = model_artifacts['feature_selector']
        self.selected_features = pd.Index(model_artifacts['selected_features'])
        self.cv_scores = model_artifacts['cv_scores']
        self.oof_predictions = model_artifacts['oof_predictions']
        self.feature_importance = model_artifacts['feature_importance']
        self.xgb_params = model_artifacts['xgb_params']
        self.random_state = model_artifacts['random_state']
        
        print(f"Model loaded from {filepath}")


def load_competition_data(data_dir):
    """Load competition datasets"""
    data_dir = Path(data_dir)
    
    try:
        # Try common file names
        train_files = ['train.csv', 'training.csv', 'train_data.csv']
        test_files = ['test.csv', 'testing.csv', 'test_data.csv']
        
        train_df = None
        test_df = None
        
        for file in train_files:
            if (data_dir / file).exists():
                train_df = pd.read_csv(data_dir / file)
                print(f"Loaded training data from {file}")
                break
        
        for file in test_files:
            if (data_dir / file).exists():
                test_df = pd.read_csv(data_dir / file)
                print(f"Loaded test data from {file}")
                break
        
        if train_df is None or test_df is None:
            print("Could not find data files. Creating dummy data for demonstration...")
            # Create dummy data
            dummy_smiles = ['CCO', 'CC(C)O', 'CCCCCCCCCCCCCCCCCC(=O)O', 'c1ccccc1', 'CC(C)(C)c1ccc(cc1)O']
            
            train_df = pd.DataFrame({
                'id': range(100),
                'smiles': np.random.choice(dummy_smiles, 100),
                'target': np.random.normal(0, 1, 100)
            })
            
            test_df = pd.DataFrame({
                'id': range(100, 150),
                'smiles': np.random.choice(dummy_smiles, 50)
            })
            
            print("Created dummy data for demonstration.")
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        return train_df, test_df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def identify_columns(train_df):
    """Identify SMILES and target columns"""
    # Identify SMILES column
    smiles_col = None
    for col in train_df.columns:
        if 'smiles' in col.lower() or 'molecule' in col.lower():
            smiles_col = col
            break
    
    if smiles_col is None:
        smiles_col = train_df.select_dtypes(include=['object']).columns[0]
    
    # Identify target column
    target_col = None
    for col in train_df.columns:
        if any(term in col.lower() for term in ['target', 'property', 'value', 'y', 'label']):
            target_col = col
            break
    
    if target_col is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        target_col = numeric_cols[-1]
    
    return smiles_col, target_col


def create_submission(test_df, predictions, submission_path):
    """Create submission file"""
    # Identify ID column
    id_col = None
    for col in test_df.columns:
        if 'id' in col.lower():
            id_col = col
            break
    
    if id_col is None:
        submission_df = pd.DataFrame({
            'id': range(len(predictions)),
            'prediction': predictions
        })
    else:
        submission_df = pd.DataFrame({
            id_col: test_df[id_col],
            'prediction': predictions
        })
    
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    return submission_df


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='NeurIPS 2025 Polymer Prediction Baseline')
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='models/', help='Output directory')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--n_features', type=int, default=200, help='Number of features to select')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    submissions_dir = Path('submissions')
    submissions_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("NeurIPS 2025 Open Polymer Prediction - Enhanced Baseline")
    print("=" * 60)
    
    # Load data
    train_df, test_df = load_competition_data(args.data_dir)
    if train_df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Define target columns for multi-target regression
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    smiles_col = 'SMILES'
    
    print(f"\\nUsing SMILES column: {smiles_col}")
    print(f"Multi-target columns: {target_columns}")
    
    # Check which targets exist in the data
    available_targets = [col for col in target_columns if col in train_df.columns]
    print(f"Available targets: {available_targets}")
    
    if not available_targets:
        print("No target columns found! Exiting.")
        return
    
    # Initialize model
    model = PolymerPredictor(random_state=args.random_state)
    
    # Generate molecular features
    print("\\n" + "="*50)
    print("FEATURE GENERATION")
    print("="*50)
    
    train_features = model.calculate_molecular_descriptors(train_df[smiles_col].tolist(), 'comprehensive')
    test_features = model.calculate_molecular_descriptors(test_df[smiles_col].tolist(), 'comprehensive')
    
    # Feature engineering
    print("\\nApplying feature engineering...")
    train_features_eng = model.engineer_features(train_features)
    test_features_eng = model.engineer_features(test_features)
    
    # Preprocess features
    print("\\nPreprocessing features...")
    X_train, X_test = model.preprocess_features(train_features_eng, test_features_eng)
    
    # Multi-target training
    print("\\n" + "="*50)
    print("MULTI-TARGET MODEL TRAINING")
    print("="*50)
    
    all_models = {}
    all_predictions = {}
    all_scores = {}
    
    for target in available_targets:
        print(f"\\n{'='*20} Training {target} {'='*20}")
        
        # Get target data for this specific target
        y_target = train_df[target]
        valid_mask = y_target.notna()
        valid_count = valid_mask.sum()
        
        print(f"Target {target}: {valid_count}/{len(y_target)} valid samples ({valid_count/len(y_target)*100:.1f}%)")
        
        if valid_count < 10:
            print(f"Insufficient data for {target}, using mean prediction")
            mean_val = y_target.mean() if valid_count > 0 else 0.0
            all_predictions[target] = np.full(len(X_test), mean_val)
            all_models[target] = None
            all_scores[target] = None
            continue
        
        # Feature selection for this target
        X_train_target, X_test_target = model.select_features(X_train, y_target, X_test, args.n_features)
        
        print(f"Selected features for {target}: {X_train_target.shape[1]}")
        
        # Train model for this target only
        target_model = PolymerPredictor(random_state=args.random_state)
        
        # Filter to only valid samples for training
        X_train_valid = X_train_target[valid_mask]
        y_train_valid = y_target[valid_mask]
        
        # Final cleanup of infinite values before XGBoost
        print(f"  Cleaning infinite values...")
        
        # Check for infinite values in features
        inf_mask = np.isinf(X_train_valid.values).any(axis=1)
        if inf_mask.sum() > 0:
            print(f"  Found {inf_mask.sum()} rows with infinite feature values, removing...")
            X_train_valid = X_train_valid[~inf_mask]
            y_train_valid = y_train_valid[~inf_mask]
        
        # Replace any remaining inf values with large finite numbers
        X_train_valid = X_train_valid.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # Check for infinite values in target
        y_inf_mask = np.isinf(y_train_valid)
        if y_inf_mask.sum() > 0:
            print(f"  Found {y_inf_mask.sum()} infinite target values, removing...")
            X_train_valid = X_train_valid[~y_inf_mask]
            y_train_valid = y_train_valid[~y_inf_mask]
        
        # Final verification
        assert np.isfinite(X_train_valid.values).all(), "Features still contain inf/nan!"
        assert np.isfinite(y_train_valid.values).all(), "Target still contains inf/nan!"
        
        print(f"  Final training data: {X_train_valid.shape[0]} samples")
        
        if len(X_train_valid) < 10:
            print(f"  Too few samples after cleaning, using mean prediction")
            mean_val = y_target[valid_mask].mean()
            all_predictions[target] = np.full(len(X_test), mean_val)
            all_models[target] = None
            all_scores[target] = None
            continue
        
        cv_scores, oof_predictions = target_model.train_cv(X_train_valid, y_train_valid, args.cv_folds)
        
        # Clean test data for predictions
        X_test_clean = X_test_target.replace([np.inf, -np.inf], [1e6, -1e6])
        X_test_clean = X_test_clean.fillna(0)
        
        # Make predictions for this target
        test_pred = target_model.predict(X_test_clean)
        
        # Store results
        all_models[target] = target_model
        all_predictions[target] = test_pred
        all_scores[target] = np.mean(cv_scores)
        
        print(f"Target {target} CV RMSE: {np.mean(cv_scores):.6f}")
    
    # Create combined predictions
    print("\\n" + "="*50)
    print("PREDICTIONS SUMMARY")
    print("="*50)
    
    test_predictions = {}
    for target in available_targets:
        test_predictions[target] = all_predictions[target]
        print(f"{target}: Mean={np.mean(all_predictions[target]):.4f}, Std={np.std(all_predictions[target]):.4f}")
    
    # Save results
    print("\\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all models
    model_artifacts = {
        'models': all_models,
        'target_columns': available_targets,
        'cv_scores': all_scores,
        'timestamp': timestamp,
        'xgb_params': model.xgb_params
    }
    model_path = output_dir / f'multi_target_baseline_{timestamp}.pkl'
    joblib.dump(model_artifacts, model_path)
    print(f"Models saved to {model_path}")
    
    # Create submission with all targets
    submission_df = pd.DataFrame({'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df))})
    for target in target_columns:
        if target in all_predictions:
            submission_df[target] = all_predictions[target]
        else:
            submission_df[target] = 0.0  # Fill missing targets with 0
    
    submission_path = submissions_dir / f'multi_target_submission_{timestamp}.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    # Save training log
    training_log = {
        'timestamp': timestamp,
        'cv_folds': args.cv_folds,
        'available_targets': available_targets,
        'target_scores': all_scores,
        'mean_cv_rmse': np.mean([score for score in all_scores.values() if score is not None]),
        'smiles_column': smiles_col,
        'total_targets': len(target_columns)
    }
    
    log_path = output_dir / f'training_log_{timestamp}.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2, default=str)
    print(f"Training log saved to {log_path}")
    
    print("\\n" + "="*60)
    print("MULTI-TARGET BASELINE TRAINING COMPLETE!")
    print("="*60)
    print("Model performance:")
    for target in available_targets:
        if all_scores[target] is not None:
            print(f"  • {target}: CV RMSE = {all_scores[target]:.6f}")
        else:
            print(f"  • {target}: Used mean prediction")
    print(f"\\nFiles saved:")
    print(f"  • Models: {model_path}")
    print(f"  • Submission: {submission_path}")
    print(f"  • Training log: {log_path}")
    print("="*60)


if __name__ == "__main__":
    main()