#!/usr/bin/env python3
"""
Quick runner for the NeurIPS 2025 Polymer Prediction baseline model.

This script provides a simple interface to run the baseline model
with default parameters.

Usage:
    python run_baseline.py
"""

import subprocess
import sys
from pathlib import Path
from src.kaggle_utils import setup_kaggle_credentials, download_competition_data


def setup_kaggle_and_data():
    """Setup Kaggle credentials and download competition data."""
    
    print("=" * 60)
    print("KAGGLE SETUP AND DATA DOWNLOAD")
    print("=" * 60)
    
    # Ask for kaggle.json path
    kaggle_json_path = input("Enter path to your kaggle.json file (or press Enter for default ~/.kaggle/kaggle.json): ").strip()
    
    if not kaggle_json_path:
        kaggle_json_path = None
    
    # Setup credentials
    if not setup_kaggle_credentials(kaggle_json_path):
        print("❌ Failed to setup Kaggle credentials")
        return False
    
    # Download competition data
    competition_name = "neurips-open-polymer-prediction-2025"
    data_dir = "data"
    
    if download_competition_data(competition_name, data_dir, kaggle_json_path):
        print("✅ Competition data downloaded successfully!")
        return True
    else:
        print("❌ Failed to download competition data")
        return False


def run_baseline():
    """Run the baseline model with default parameters."""
    
    print("=" * 60)
    print("NeurIPS 2025 Open Polymer Prediction - Baseline Runner")
    print("=" * 60)
    
    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"⚠️  Data directory '{data_dir}' not found.")
        print("Please download the competition data first:")
        print("  uv run kaggle competitions download -c neurips-open-polymer-prediction-2025 -p data/")
        return
    
    # Check if there are any CSV files in data directory
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"⚠️  No CSV files found in '{data_dir}'.")
        print("Please download the competition data first:")
        print("  uv run kaggle competitions download -c neurips-open-polymer-prediction-2025 -p data/")
        print("  cd data && unzip *.zip")
        return
    
    print(f"✅ Found {len(csv_files)} CSV files in data directory")
    for file in csv_files:
        print(f"   • {file.name}")
    
    # Run the baseline model
    print(f"\\n🚀 Running baseline model...")
    
    try:
        cmd = [
            sys.executable, 
            "src/baseline_model.py",
            "--data_dir", "data/",
            "--output_dir", "models/",
            "--cv_folds", "5",
            "--n_features", "200"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        print(f"\\n✅ Baseline model completed successfully!")
        print(f"Check the 'models/' and 'submissions/' directories for outputs.")
        
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Error running baseline model:")
        print(f"Exit code: {e.returncode}")
        print(f"Try running manually: uv run python src/baseline_model.py")
    
    except FileNotFoundError:
        print(f"\\n❌ Could not find baseline_model.py")
        print(f"Make sure you're running this from the project root directory.")


def run_optimization():
    """Run hyperparameter optimization."""
    
    print("\\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    model_type = input("Enter model type (xgboost/lightgbm/catboost) [xgboost]: ").strip().lower()
    if not model_type:
        model_type = "xgboost"
    
    if model_type not in ["xgboost", "lightgbm", "catboost"]:
        print(f"Invalid model type: {model_type}")
        return
    
    n_trials = input("Enter number of trials [50]: ").strip()
    if not n_trials:
        n_trials = "50"
    
    try:
        n_trials = int(n_trials)
    except ValueError:
        print(f"Invalid number of trials: {n_trials}")
        return
    
    print(f"\\n🚀 Running {model_type} optimization with {n_trials} trials...")
    
    try:
        cmd = [
            sys.executable,
            "src/hyperparameter_optimization.py",
            "--data_dir", "data/",
            "--output_dir", "models/",
            "--model_type", model_type,
            "--n_trials", str(n_trials),
            "--cv_folds", "5"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        print(f"\\n✅ Optimization completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Error running optimization:")
        print(f"Exit code: {e.returncode}")
    
    except FileNotFoundError:
        print(f"\\n❌ Could not find hyperparameter_optimization.py")


def main():
    """Main menu."""
    
    while True:
        print("\\n" + "=" * 50)
        print("NEURIPS 2025 POLYMER PREDICTION - MAIN MENU")
        print("=" * 50)
        print("1. Setup Kaggle credentials and download data")
        print("2. Run baseline model")
        print("3. Run hyperparameter optimization")
        print("4. Exit")
        print("=" * 50)
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            setup_kaggle_and_data()
        elif choice == "2":
            run_baseline()
        elif choice == "3":
            run_optimization()
        elif choice == "4":
            print("\\n👋 Goodbye!")
            break
        else:
            print("\\n❌ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()