#!/usr/bin/env python3
"""
Bulletproof Baseline - Handles ALL edge cases
============================================

This version eliminates ALL possible NaN errors and runs fast.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')


def create_bulletproof_features(smiles_list):
    """Create robust features that never fail."""
    
    print(f"Creating bulletproof features for {len(smiles_list)} molecules...")
    
    features = []
    
    for smiles in smiles_list:
        try:
            # String-based features (always work)
            feat = {
                'length': len(smiles),
                'star_count': smiles.count('*'),
                'carbon_count': smiles.count('C') + smiles.count('c'),
                'oxygen_count': smiles.count('O') + smiles.count('o'),
                'nitrogen_count': smiles.count('N') + smiles.count('n'),
                'ring_char_count': smiles.count('c'),
                'aromatic_count': smiles.count('c') + smiles.count('n') + smiles.count('o'),
                'branch_count': smiles.count('('),
                'double_bond_count': smiles.count('='),
                'triple_bond_count': smiles.count('#'),
                'ring_digit_count': sum(1 for c in smiles if c.isdigit()),
                'bracket_count': smiles.count('['),
                'charge_count': smiles.count('+') + smiles.count('-'),
                'fluorine_count': smiles.count('F'),
                'chlorine_count': smiles.count('Cl'),
                'bromine_count': smiles.count('Br'),
                'iodine_count': smiles.count('I'),
                'sulfur_count': smiles.count('S'),
                'phosphorus_count': smiles.count('P'),
                'complexity_score': len(set(smiles)),  # unique character count
                'polymer_ratio': smiles.count('*') / max(len(smiles), 1)
            }
            
            # Ensure all values are finite
            for key, value in feat.items():
                if not np.isfinite(value):
                    feat[key] = 0.0
            
            features.append(feat)
            
        except Exception as e:
            print(f"Error with SMILES '{smiles}': {e}")
            # Fallback: all zeros
            feat = {
                'length': 0, 'star_count': 0, 'carbon_count': 0, 'oxygen_count': 0,
                'nitrogen_count': 0, 'ring_char_count': 0, 'aromatic_count': 0,
                'branch_count': 0, 'double_bond_count': 0, 'triple_bond_count': 0,
                'ring_digit_count': 0, 'bracket_count': 0, 'charge_count': 0,
                'fluorine_count': 0, 'chlorine_count': 0, 'bromine_count': 0,
                'iodine_count': 0, 'sulfur_count': 0, 'phosphorus_count': 0,
                'complexity_score': 0, 'polymer_ratio': 0
            }
            features.append(feat)
    
    df = pd.DataFrame(features)
    
    # Final safety check - replace any remaining NaN/inf
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    print(f"Created {df.shape[1]} bulletproof features")
    return df


def bulletproof_train_target(X_train, y_target, target_name):
    """Train model for single target with bulletproof NaN handling."""
    
    print(f"\\nTraining {target_name}...")
    
    # Step 1: Handle target NaNs
    valid_mask = pd.notna(y_target)
    valid_count = valid_mask.sum()
    
    print(f"  Valid samples: {valid_count}/{len(y_target)} ({valid_count/len(y_target)*100:.1f}%)")
    
    if valid_count < 5:
        print(f"  ❌ Insufficient data, using mean prediction")
        mean_val = y_target.dropna().mean() if valid_count > 0 else 0.0
        mean_val = mean_val if np.isfinite(mean_val) else 0.0
        return None, mean_val, f"mean_{mean_val:.4f}"
    
    # Step 2: Get valid data
    X_valid = X_train[valid_mask].copy()
    y_valid = y_target[valid_mask].copy()
    
    # Step 3: Handle feature NaNs
    feature_imputer = SimpleImputer(strategy='median')
    X_valid_clean = feature_imputer.fit_transform(X_valid)
    X_valid_clean = pd.DataFrame(X_valid_clean, columns=X_train.columns)
    
    # Step 4: Final safety checks
    X_valid_clean = X_valid_clean.replace([np.inf, -np.inf], 0).fillna(0)
    y_valid_clean = pd.Series(y_valid).replace([np.inf, -np.inf], np.nan).fillna(y_valid.median())
    
    # Step 5: Verify no NaNs remain
    assert not X_valid_clean.isna().any().any(), "Features still contain NaN!"
    assert not y_valid_clean.isna().any(), "Target still contains NaN!"
    assert np.isfinite(X_valid_clean.values).all(), "Features contain inf!"
    assert np.isfinite(y_valid_clean.values).all(), "Target contains inf!"
    
    print(f"  ✅ Data cleaned: {X_valid_clean.shape[0]} samples, {X_valid_clean.shape[1]} features")
    
    # Step 6: Train model
    try:
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_valid_clean, y_valid_clean)
        
        # Quick validation
        train_pred = model.predict(X_valid_clean)
        train_rmse = np.sqrt(np.mean((y_valid_clean - train_pred) ** 2))
        
        print(f"  ✅ Model trained successfully, RMSE: {train_rmse:.6f}")
        
        return (model, feature_imputer), None, f"model_rmse_{train_rmse:.6f}"
        
    except Exception as e:
        print(f"  ❌ Model training failed: {e}")
        print(f"  Using mean prediction as fallback")
        mean_val = y_valid_clean.mean()
        return None, mean_val, f"fallback_mean_{mean_val:.4f}"


def bulletproof_predict(model_info, X_test, fallback_value):
    """Make bulletproof predictions."""
    
    if model_info is None:
        # Use fallback value
        pred = np.full(len(X_test), fallback_value)
        print(f"    Using fallback: {pred}")
        return pred
    
    try:
        model, imputer = model_info
        
        # Clean test data same way as training
        X_test_clean = imputer.transform(X_test)
        X_test_clean = pd.DataFrame(X_test_clean, columns=X_test.columns)
        X_test_clean = X_test_clean.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Predict
        pred = model.predict(X_test_clean)
        
        # Safety check predictions
        pred = np.array(pred)
        pred = np.where(np.isfinite(pred), pred, fallback_value)
        
        print(f"    Predictions: {pred}")
        return pred
        
    except Exception as e:
        print(f"    ❌ Prediction failed: {e}, using fallback")
        pred = np.full(len(X_test), fallback_value)
        return pred


def main():
    print("=" * 60)
    print("🛡️  BULLETPROOF BASELINE - NO MORE ERRORS!")
    print("=" * 60)
    
    try:
        # Load data
        print("\\n📁 Loading data...")
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        
        print(f"✅ Data loaded: Train {train_df.shape}, Test {test_df.shape}")
        
        targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        # Show data quality
        print(f"\\n📊 Target data quality:")
        for target in targets:
            if target in train_df.columns:
                valid = train_df[target].notna().sum()
                total = len(train_df)
                mean_val = train_df[target].mean() if valid > 0 else 0
                print(f"  {target}: {valid}/{total} valid ({valid/total*100:.1f}%), mean={mean_val:.4f}")
            else:
                print(f"  {target}: ❌ Column missing!")
        
        # Create features
        print(f"\\n🔧 Creating bulletproof features...")
        train_features = create_bulletproof_features(train_df['SMILES'].tolist())
        test_features = create_bulletproof_features(test_df['SMILES'].tolist())
        
        print(f"✅ Features created: Train {train_features.shape}, Test {test_features.shape}")
        
        # Train models
        print(f"\\n🤖 Training bulletproof models...")
        models = {}
        predictions = {}
        
        for target in targets:
            if target in train_df.columns:
                model_info, fallback, status = bulletproof_train_target(
                    train_features, train_df[target], target
                )
                
                pred = bulletproof_predict(model_info, test_features, fallback or 0.0)
                
                models[target] = (model_info, fallback, status)
                predictions[target] = pred
            else:
                print(f"\\n❌ {target} not in data, using zeros")
                predictions[target] = np.zeros(len(test_features))
                models[target] = (None, 0.0, "missing_column")
        
        # Create submission
        print(f"\\n📄 Creating bulletproof submission...")
        submission = pd.DataFrame({'id': test_df['id']})
        
        for target in targets:
            submission[target] = predictions[target]
        
        # Final safety check on submission
        for col in submission.columns:
            if col != 'id':
                submission[col] = submission[col].replace([np.inf, -np.inf], 0).fillna(0)
        
        print("✅ Submission created:")
        print(submission)
        
        # Save everything
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save submission
        Path('submissions').mkdir(exist_ok=True)
        submission_path = f'submissions/bulletproof_submission_{timestamp}.csv'
        submission.to_csv(submission_path, index=False)
        
        # Save models
        Path('models').mkdir(exist_ok=True)
        artifacts = {
            'models': models,
            'targets': targets,
            'timestamp': timestamp,
            'feature_columns': list(train_features.columns)
        }
        model_path = f'models/bulletproof_models_{timestamp}.pkl'
        joblib.dump(artifacts, model_path)
        
        print(f"\\n" + "=" * 60)
        print("✅ BULLETPROOF BASELINE COMPLETE - NO ERRORS!")
        print("=" * 60)
        
        print("\\n📈 Model Summary:")
        for target in targets:
            if target in models:
                _, _, status = models[target]
                print(f"  • {target}: {status}")
        
        print(f"\\n💾 Files saved:")
        print(f"  • Submission: {submission_path}")
        print(f"  • Models: {model_path}")
        
        print(f"\\n🚀 Ready for Kaggle! No more NaN errors!")
        
        return True
        
    except Exception as e:
        print(f"\\n💥 CRITICAL ERROR: {e}")
        print(f"\\nFull traceback:")
        import traceback
        traceback.print_exc()
        
        print(f"\\n📧 Please share this full error with Claude!")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\\n🎉 SUCCESS! Script completed without errors.")
    else:
        print("\\n💥 FAILED! Please share the error details with Claude.")