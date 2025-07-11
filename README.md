# NeurIPS Open Polymer Prediction 2025

[![Competition](https://img.shields.io/badge/Kaggle-NeurIPS%202025-blue)](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Machine learning solution for predicting polymer properties using molecular descriptors and advanced ML techniques.

## 🧪 Competition Overview

The NeurIPS Open Polymer Prediction 2025 challenge focuses on predicting five key polymer properties:
- **Tg**: Glass transition temperature
- **FFV**: Fractional free volume  
- **Tc**: Critical temperature
- **Density**: Material density
- **Rg**: Radius of gyration

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- UV package manager (recommended) or pip

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/neurips_opp2025.git
cd neurips_opp2025

# Install dependencies using UV (recommended)
uv sync

# Or install with pip
pip install -r requirements.txt
```

### Download Competition Data
```bash
# Set up Kaggle API credentials first
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Download data
uv run kaggle competitions download -c neurips-open-polymer-prediction-2025 -p data/
cd data && unzip neurips-open-polymer-prediction-2025.zip
```

### Run Baseline Models
```bash
# Quick working baseline (fast)
uv run python working_baseline.py

# Enhanced RDKit baseline (comprehensive)
uv run python src/baseline_model.py --data_dir data/

# Optuna-optimized baseline
uv run python optuna_baseline.py
```

## 📁 Project Structure

```
neurips_opp2025/
├── README.md                 # Project documentation
├── CLAUDE.md                 # Claude Code instructions
├── pyproject.toml           # UV/Python dependencies
├── requirements.txt         # Pip requirements
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT license
│
├── data/                   # Competition data
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── baseline_model.py   # Enhanced RDKit baseline
│   ├── kaggle_utils.py     # Kaggle API utilities
│   └── features/           # Feature engineering
│       ├── __init__.py
│       ├── molecular.py    # RDKit descriptors
│       └── engineering.py  # Feature transformations
│
├── notebooks/              # Jupyter notebooks
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_comparison.ipynb
│
├── baselines/              # Baseline implementations
│   ├── working_baseline.py     # Simple reliable baseline
│   ├── optuna_baseline.py      # Hyperparameter optimized
│   ├── fixed_baseline_simple.py
│   └── bulletproof_baseline.py
│
├── models/                 # Saved models
│   └── .gitkeep
│
├── submissions/            # Kaggle submissions
│   └── .gitkeep
│
└── scripts/               # Utility scripts
    ├── download_data.py
    ├── generate_submission.py
    └── validate_format.py
```

## 🤖 Available Models

### 1. Working Baseline (`working_baseline.py`)
- **Speed**: ⚡ Very fast (< 1 minute)
- **Features**: Simple string-based molecular features
- **Models**: Mean prediction strategy
- **Use case**: Quick validation and baseline submission

### 2. Enhanced RDKit Baseline (`src/baseline_model.py`)
- **Speed**: 🔄 Slower (10-30 minutes)
- **Features**: Comprehensive RDKit molecular descriptors
- **Models**: XGBoost with RandomForest/Linear fallback
- **Use case**: High-quality baseline with molecular chemistry

### 3. Optuna-Optimized Baseline (`optuna_baseline.py`)
- **Speed**: ⏱️ Medium (5-15 minutes)
- **Features**: Robust features with hyperparameter optimization
- **Models**: LightGBM with Optuna tuning
- **Use case**: Optimized performance with automated tuning

## 📊 Results

| Model | CV Score | Public LB | Description |
|-------|----------|-----------|-------------|
| Working Baseline | - | - | Simple mean predictions |
| RDKit Enhanced | ~0.08 | - | Molecular descriptors + XGBoost |
| Optuna Optimized | ~0.07 | - | Hyperparameter tuned LightGBM |

## ⚙️ Key Features

- **Robust Error Handling**: All models handle NaN/infinite values gracefully
- **Multi-target Support**: Handles sparse target data across all 5 properties
- **Molecular Chemistry**: RDKit integration for chemical feature engineering
- **Hyperparameter Optimization**: Optuna integration for automated tuning
- **Production Ready**: Comprehensive logging, validation, and error recovery

## 🧬 Molecular Features

The project leverages several types of molecular descriptors:

- **Basic Descriptors**: Molecular weight, LogP, TPSA, atom counts
- **Topological**: Connectivity indices, Kappa descriptors
- **Electronic**: PEOE charges, VSA descriptors  
- **Geometric**: Radius of gyration, surface area
- **Custom**: Polymer-specific features (star count, branching)

## 🔄 Development Workflow

1. **Data Exploration**: Use notebooks for EDA and feature analysis
2. **Baseline Development**: Start with `working_baseline.py` for validation
3. **Feature Engineering**: Enhance with molecular descriptors
4. **Model Optimization**: Use Optuna for hyperparameter tuning
5. **Ensemble**: Combine multiple approaches for final submission

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📅 Competition Timeline

- **Start**: June 2025
- **Final Submission**: September 15, 2025
- **Prize Pool**: $50,000 USD

## 📚 References

- [Competition Page](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)
- [RDKit Documentation](https://rdkit.org/docs/)
- [Polymer Chemistry Handbook](https://link.springer.com/book/10.1007/978-3-540-29878-0)

## 🙏 Acknowledgments

- NeurIPS 2025 competition organizers
- RDKit community for molecular descriptors
- Kaggle for hosting the competition
- Claude Code for development assistance