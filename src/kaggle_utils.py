#!/usr/bin/env python3
"""
Kaggle API utilities for NeurIPS 2025 Polymer Prediction
========================================================

This module provides utilities for setting up Kaggle API credentials
and downloading competition data.
"""

import os
import shutil
import subprocess
from pathlib import Path
import json


def setup_kaggle_credentials(kaggle_json_path=None):
    """
    Set up Kaggle API credentials.
    
    Parameters:
    -----------
    kaggle_json_path : str or Path, optional
        Path to kaggle.json file. If None, looks in default locations.
    
    Returns:
    --------
    bool
        True if credentials are set up successfully, False otherwise
    """
    
    # Default Kaggle directory
    kaggle_dir = Path.home() / '.kaggle'
    default_kaggle_json = kaggle_dir / 'kaggle.json'
    
    # If custom path is provided
    if kaggle_json_path:
        kaggle_json_path = Path(kaggle_json_path)
        
        if not kaggle_json_path.exists():
            print(f"❌ Kaggle JSON file not found at: {kaggle_json_path}")
            return False
        
        # Create .kaggle directory if it doesn't exist
        kaggle_dir.mkdir(exist_ok=True)
        
        # Copy the kaggle.json file to the default location
        try:
            shutil.copy2(kaggle_json_path, default_kaggle_json)
            print(f"✅ Copied kaggle.json from {kaggle_json_path} to {default_kaggle_json}")
        except Exception as e:
            print(f"❌ Error copying kaggle.json: {e}")
            return False
    
    # Check if kaggle.json exists
    if not default_kaggle_json.exists():
        print(f"❌ kaggle.json not found at {default_kaggle_json}")
        print(f"Please provide the path to your kaggle.json file or place it in ~/.kaggle/")
        return False
    
    # Set proper permissions (required by Kaggle API)
    try:
        os.chmod(default_kaggle_json, 0o600)
        print(f"✅ Set permissions for {default_kaggle_json}")
    except Exception as e:
        print(f"⚠️  Warning: Could not set permissions for kaggle.json: {e}")
    
    # Validate the JSON file
    try:
        with open(default_kaggle_json, 'r') as f:
            credentials = json.load(f)
        
        required_keys = ['username', 'key']
        if not all(key in credentials for key in required_keys):
            print(f"❌ Invalid kaggle.json format. Required keys: {required_keys}")
            return False
        
        print(f"✅ Kaggle credentials validated for user: {credentials['username']}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format in kaggle.json: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading kaggle.json: {e}")
        return False


def test_kaggle_api():
    """
    Test if Kaggle API is working properly.
    
    Returns:
    --------
    bool
        True if API is working, False otherwise
    """
    try:
        result = subprocess.run(
            ['kaggle', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Kaggle API working: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Kaggle API error: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Kaggle CLI not found. Install with: pip install kaggle")
        return False


def download_competition_data(competition_name, data_dir, kaggle_json_path=None):
    """
    Download competition data using Kaggle API.
    
    Parameters:
    -----------
    competition_name : str
        Name of the Kaggle competition
    data_dir : str or Path
        Directory to download data to
    kaggle_json_path : str or Path, optional
        Path to kaggle.json file
    
    Returns:
    --------
    bool
        True if download successful, False otherwise
    """
    
    # Setup credentials
    if not setup_kaggle_credentials(kaggle_json_path):
        return False
    
    # Test API
    if not test_kaggle_api():
        return False
    
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    print(f"📥 Downloading competition data for: {competition_name}")
    print(f"📁 Download directory: {data_dir}")
    
    try:
        # Download competition files
        result = subprocess.run([
            'kaggle', 'competitions', 'download',
            '-c', competition_name,
            '-p', str(data_dir)
        ], capture_output=True, text=True, check=True)
        
        print(f"✅ Download completed successfully")
        
        # List downloaded files
        zip_files = list(data_dir.glob('*.zip'))
        if zip_files:
            print(f"📦 Downloaded files:")
            for zip_file in zip_files:
                print(f"  • {zip_file.name}")
            
            # Ask if user wants to extract
            extract = input("Extract ZIP files? (y/n) [y]: ").strip().lower()
            if extract in ['', 'y', 'yes']:
                extract_zip_files(data_dir)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False


def extract_zip_files(data_dir):
    """
    Extract all ZIP files in the data directory.
    
    Parameters:
    -----------
    data_dir : str or Path
        Directory containing ZIP files
    """
    import zipfile
    
    data_dir = Path(data_dir)
    zip_files = list(data_dir.glob('*.zip'))
    
    if not zip_files:
        print("No ZIP files found to extract")
        return
    
    for zip_file in zip_files:
        print(f"📦 Extracting {zip_file.name}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"✅ Extracted {zip_file.name}")
        except Exception as e:
            print(f"❌ Error extracting {zip_file.name}: {e}")
    
    # List extracted files
    csv_files = list(data_dir.glob('*.csv'))
    if csv_files:
        print(f"\\n📄 CSV files available:")
        for csv_file in csv_files:
            print(f"  • {csv_file.name}")


def main():
    """Command line interface for Kaggle utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kaggle API utilities')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup credentials command
    setup_parser = subparsers.add_parser('setup', help='Setup Kaggle credentials')
    setup_parser.add_argument('--kaggle-json', type=str, help='Path to kaggle.json file')
    
    # Download data command
    download_parser = subparsers.add_parser('download', help='Download competition data')
    download_parser.add_argument('--competition', type=str, required=True, help='Competition name')
    download_parser.add_argument('--data-dir', type=str, default='data/', help='Data directory')
    download_parser.add_argument('--kaggle-json', type=str, help='Path to kaggle.json file')
    
    # Test API command
    test_parser = subparsers.add_parser('test', help='Test Kaggle API')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_kaggle_credentials(args.kaggle_json)
    elif args.command == 'download':
        download_competition_data(args.competition, args.data_dir, args.kaggle_json)
    elif args.command == 'test':
        test_kaggle_api()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()