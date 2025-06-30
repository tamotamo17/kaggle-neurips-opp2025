# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains code for the NeurIPS Open Polymer Prediction 2025 competition on Kaggle. The goal is to predict polymer properties using machine learning techniques.

## Environment Setup

This project requires Python with machine learning libraries. Key dependencies include:
- Core ML: scikit-learn, xgboost, lightgbm, catboost
- Deep learning: torch, tensorflow
- Chemistry/molecular: rdkit-pypi, mordred
- Data science: pandas, numpy, matplotlib, seaborn
- Kaggle API for data download

## Common Commands

Since this is a Kaggle competition project, these commands will be frequently used:

```bash
# Download competition data
kaggle competitions download -c neurips-open-polymer-prediction-2025

# Extract data
unzip neurips-open-polymer-prediction-2025.zip

# Submit predictions
kaggle competitions submit -c neurips-open-polymer-prediction-2025 -f submission.csv -m "submission message"
```

## Development Workflow

1. Download and explore competition data
2. Perform exploratory data analysis in notebooks
3. Feature engineering with molecular descriptors
4. Model training and validation
5. Generate predictions and submissions

## Key Considerations

- This is a polymer prediction task, so molecular chemistry knowledge is important
- RDKit and Mordred are essential for molecular descriptor generation
- Cross-validation strategy should account for potential molecular similarity
- Ensemble methods often perform well in chemistry competitions