"""
EDA Utilities for NeurIPS Open Polymer Prediction 2025
======================================================

This module contains utility functions for exploratory data analysis
specifically designed for polymer prediction tasks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False


class PolymerEDA:
    """
    Comprehensive EDA class for polymer prediction competitions.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the EDA class with dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to analyze
        """
        self.data = data.copy()
        self.numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        self.target_cols = []
        self.smiles_cols = []
        
    def identify_columns(self, target_hints: List[str] = None, smiles_hints: List[str] = None):
        """
        Automatically identify target and SMILES columns.
        
        Parameters:
        -----------
        target_hints : List[str], optional
            Column name patterns that might indicate target variables
        smiles_hints : List[str], optional
            Column name patterns that might indicate SMILES columns
        """
        if target_hints is None:
            target_hints = ['target', 'property', 'value', 'measurement', 'y', 'label']
        
        if smiles_hints is None:
            smiles_hints = ['smiles', 'molecule', 'structure', 'mol', 'compound']
        
        # Identify target columns
        self.target_cols = [col for col in self.numeric_cols 
                           if any(hint in col.lower() for hint in target_hints)]
        
        # Identify SMILES columns
        self.smiles_cols = [col for col in self.data.columns 
                           if any(hint in col.lower() for hint in smiles_hints)]
        
        print(f"Identified target columns: {self.target_cols}")
        print(f"Identified SMILES columns: {self.smiles_cols}")
    
    def basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing basic dataset information
        """
        info = {
            'shape': self.data.shape,
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': self.data.isnull().sum().sum(),
            'missing_percentage': (self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])) * 100,
            'duplicated_rows': self.data.duplicated().sum(),
            'numeric_columns': len(self.numeric_cols),
            'categorical_columns': len(self.categorical_cols),
            'data_types': self.data.dtypes.value_counts().to_dict()
        }
        return info
    
    def missing_data_analysis(self) -> pd.DataFrame:
        """
        Analyze missing data patterns.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with missing data statistics
        """
        missing_data = []
        
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            missing_pct = (missing_count / len(self.data)) * 100
            
            missing_data.append({
                'column': col,
                'missing_count': missing_count,
                'missing_percentage': missing_pct,
                'data_type': str(self.data[col].dtype),
                'unique_values': self.data[col].nunique()
            })
        
        missing_df = pd.DataFrame(missing_data)
        return missing_df.sort_values('missing_count', ascending=False)
    
    def target_analysis(self, target_col: str = None) -> Dict[str, Any]:
        """
        Analyze target variable distribution.
        
        Parameters:
        -----------
        target_col : str, optional
            Target column name. If None, uses first identified target column.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing target variable statistics
        """
        if target_col is None:
            if not self.target_cols:
                raise ValueError("No target columns identified. Please specify target_col parameter.")
            target_col = self.target_cols[0]
        
        target_data = self.data[target_col].dropna()
        
        stats = {
            'column_name': target_col,
            'count': len(target_data),
            'mean': target_data.mean(),
            'median': target_data.median(),
            'std': target_data.std(),
            'min': target_data.min(),
            'max': target_data.max(),
            'range': target_data.max() - target_data.min(),
            'skewness': target_data.skew(),
            'kurtosis': target_data.kurtosis(),
            'cv': target_data.std() / target_data.mean() if target_data.mean() != 0 else np.inf,
            'outliers_iqr': self._count_outliers_iqr(target_data),
            'outliers_zscore': self._count_outliers_zscore(target_data)
        }
        
        return stats
    
    def plot_target_distribution(self, target_col: str = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot target variable distribution.
        
        Parameters:
        -----------
        target_col : str, optional
            Target column name
        figsize : Tuple[int, int]
            Figure size
        """
        if target_col is None:
            if not self.target_cols:
                raise ValueError("No target columns identified.")
            target_col = self.target_cols[0]
        
        target_data = self.data[target_col].dropna()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Distribution Analysis: {target_col}', fontsize=16)
        
        # Histogram
        axes[0, 0].hist(target_data, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Histogram')
        axes[0, 0].set_xlabel(target_col)
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(target_data)
        axes[0, 1].set_title('Box Plot')
        axes[0, 1].set_ylabel(target_col)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(target_data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal)')
        
        # Density plot
        axes[1, 1].hist(target_data, bins=50, alpha=0.7, density=True, edgecolor='black')
        axes[1, 1].set_title('Density Plot')
        axes[1, 1].set_xlabel(target_col)
        axes[1, 1].set_ylabel('Density')
        
        plt.tight_layout()
        plt.show()
    
    def molecular_analysis(self, smiles_col: str = None, sample_size: int = 100) -> Dict[str, Any]:
        """
        Analyze molecular structures from SMILES.
        
        Parameters:
        -----------
        smiles_col : str, optional
            SMILES column name
        sample_size : int
            Number of molecules to analyze
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing molecular analysis results
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular analysis")
        
        if smiles_col is None:
            if not self.smiles_cols:
                raise ValueError("No SMILES columns identified.")
            smiles_col = self.smiles_cols[0]
        
        smiles_data = self.data[smiles_col].dropna().unique()[:sample_size]
        
        valid_molecules = []
        invalid_count = 0
        
        for smiles in smiles_data:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_molecules.append(smiles)
                else:
                    invalid_count += 1
            except:
                invalid_count += 1
        
        # Calculate molecular properties
        mol_properties = []
        for smiles in valid_molecules[:50]:  # Limit to 50 for speed
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                props = {
                    'SMILES': smiles,
                    'MolWt': Descriptors.MolWt(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'NumHDonors': Descriptors.NumHDonors(mol),
                    'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                    'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                    'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'NumAtoms': mol.GetNumAtoms(),
                    'NumBonds': mol.GetNumBonds()
                }
                mol_properties.append(props)
        
        analysis_results = {
            'total_smiles': len(smiles_data),
            'valid_smiles': len(valid_molecules),
            'invalid_smiles': invalid_count,
            'validity_rate': len(valid_molecules) / len(smiles_data) * 100,
            'molecular_properties': pd.DataFrame(mol_properties) if mol_properties else None
        }
        
        return analysis_results
    
    def correlation_analysis(self, method: str = 'pearson', threshold: float = 0.7) -> Dict[str, Any]:
        """
        Perform correlation analysis on numeric variables.
        
        Parameters:
        -----------
        method : str
            Correlation method ('pearson', 'spearman', 'kendall')
        threshold : float
            Threshold for identifying high correlations
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing correlation analysis results
        """
        if len(self.numeric_cols) < 2:
            return {'error': 'Insufficient numeric columns for correlation analysis'}
        
        corr_matrix = self.data[self.numeric_cols].corr(method=method)
        
        # Find high correlation pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        results = {
            'correlation_matrix': corr_matrix,
            'high_correlations': sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True),
            'method': method,
            'threshold': threshold
        }
        
        return results
    
    def plot_correlation_heatmap(self, method: str = 'pearson', figsize: Tuple[int, int] = (12, 10)):
        """
        Plot correlation heatmap.
        
        Parameters:
        -----------
        method : str
            Correlation method
        figsize : Tuple[int, int]
            Figure size
        """
        if len(self.numeric_cols) < 2:
            print("Insufficient numeric columns for correlation heatmap")
            return
        
        corr_matrix = self.data[self.numeric_cols].corr(method=method)
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(f'Correlation Matrix ({method.capitalize()})')
        plt.tight_layout()
        plt.show()
    
    def outlier_detection(self, methods: List[str] = None) -> pd.DataFrame:
        """
        Detect outliers using multiple methods.
        
        Parameters:
        -----------
        methods : List[str], optional
            List of outlier detection methods ('iqr', 'zscore', 'isolation_forest')
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with outlier statistics
        """
        if methods is None:
            methods = ['iqr', 'zscore']
        
        outlier_stats = []
        
        for col in self.numeric_cols:
            data = self.data[col].dropna()
            if len(data) == 0:
                continue
            
            stats = {'column': col, 'total_points': len(data)}
            
            if 'iqr' in methods:
                iqr_outliers = self._count_outliers_iqr(data)
                stats.update({
                    'iqr_mild_outliers': iqr_outliers['mild'],
                    'iqr_extreme_outliers': iqr_outliers['extreme'],
                    'iqr_outlier_pct': (iqr_outliers['mild'] + iqr_outliers['extreme']) / len(data) * 100
                })
            
            if 'zscore' in methods:
                zscore_outliers = self._count_outliers_zscore(data)
                stats.update({
                    'zscore_2std_outliers': zscore_outliers['2std'],
                    'zscore_3std_outliers': zscore_outliers['3std'],
                    'zscore_outlier_pct': zscore_outliers['2std'] / len(data) * 100
                })
            
            outlier_stats.append(stats)
        
        return pd.DataFrame(outlier_stats)
    
    def _count_outliers_iqr(self, data: pd.Series) -> Dict[str, int]:
        """Count outliers using IQR method."""
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        extreme_lower = q1 - 3 * iqr
        extreme_upper = q3 + 3 * iqr
        
        mild = ((data < lower_bound) & (data >= extreme_lower)).sum() + \
               ((data > upper_bound) & (data <= extreme_upper)).sum()
        extreme = (data < extreme_lower).sum() + (data > extreme_upper).sum()
        
        return {'mild': mild, 'extreme': extreme}
    
    def _count_outliers_zscore(self, data: pd.Series) -> Dict[str, int]:
        """Count outliers using Z-score method."""
        z_scores = np.abs((data - data.mean()) / data.std())
        return {
            '2std': (z_scores > 2).sum(),
            '3std': (z_scores > 3).sum()
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Generate a comprehensive EDA report.
        
        Parameters:
        -----------
        output_file : str, optional
            Path to save the report
        
        Returns:
        --------
        str
            Report content
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("COMPREHENSIVE EDA REPORT")
        report_lines.append("NeurIPS Open Polymer Prediction 2025")
        report_lines.append("=" * 60)
        
        # Basic info
        basic_info = self.basic_info()
        report_lines.append("\\n📊 DATASET OVERVIEW:")
        report_lines.append(f"   • Shape: {basic_info['shape']}")
        report_lines.append(f"   • Memory usage: {basic_info['memory_usage_mb']:.2f} MB")
        report_lines.append(f"   • Missing values: {basic_info['missing_values']} ({basic_info['missing_percentage']:.2f}%)")
        report_lines.append(f"   • Duplicated rows: {basic_info['duplicated_rows']}")
        report_lines.append(f"   • Numeric columns: {basic_info['numeric_columns']}")
        report_lines.append(f"   • Categorical columns: {basic_info['categorical_columns']}")
        
        # Target analysis
        if self.target_cols:
            report_lines.append("\\n🎯 TARGET ANALYSIS:")
            for target_col in self.target_cols:
                try:
                    target_stats = self.target_analysis(target_col)
                    report_lines.append(f"   • {target_col}:")
                    report_lines.append(f"     - Mean: {target_stats['mean']:.4f}")
                    report_lines.append(f"     - Std: {target_stats['std']:.4f}")
                    report_lines.append(f"     - Skewness: {target_stats['skewness']:.4f}")
                    report_lines.append(f"     - Outliers (IQR): {target_stats['outliers_iqr']['mild'] + target_stats['outliers_iqr']['extreme']}")
                except Exception as e:
                    report_lines.append(f"   • {target_col}: Analysis failed - {str(e)}")
        
        # Molecular analysis
        if self.smiles_cols and RDKIT_AVAILABLE:
            report_lines.append("\\n🧪 MOLECULAR ANALYSIS:")
            try:
                mol_analysis = self.molecular_analysis()
                report_lines.append(f"   • Total SMILES: {mol_analysis['total_smiles']}")
                report_lines.append(f"   • Valid SMILES: {mol_analysis['valid_smiles']}")
                report_lines.append(f"   • Validity rate: {mol_analysis['validity_rate']:.1f}%")
            except Exception as e:
                report_lines.append(f"   • Analysis failed: {str(e)}")
        
        # Correlation analysis
        try:
            corr_analysis = self.correlation_analysis()
            if 'error' not in corr_analysis:
                report_lines.append("\\n🔍 CORRELATION ANALYSIS:")
                report_lines.append(f"   • High correlations (>0.7): {len(corr_analysis['high_correlations'])}")
                for pair in corr_analysis['high_correlations'][:5]:  # Top 5
                    report_lines.append(f"     - {pair['var1']} ↔ {pair['var2']}: {pair['correlation']:.3f}")
        except Exception as e:
            report_lines.append(f"\\n🔍 CORRELATION ANALYSIS: Failed - {str(e)}")
        
        report_content = "\\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            print(f"Report saved to {output_file}")
        
        return report_content


def create_molecular_features(smiles_list: List[str], descriptor_set: str = 'basic') -> pd.DataFrame:
    """
    Create molecular features from SMILES strings.
    
    Parameters:
    -----------
    smiles_list : List[str]
        List of SMILES strings
    descriptor_set : str
        Set of descriptors to calculate ('basic', 'extended', 'all')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with molecular features
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecular feature generation")
    
    features = []
    
    basic_descriptors = [
        ('MolWt', Descriptors.MolWt),
        ('LogP', Descriptors.MolLogP),
        ('NumHDonors', Descriptors.NumHDonors),
        ('NumHAcceptors', Descriptors.NumHAcceptors),
        ('NumRotatableBonds', Descriptors.NumRotatableBonds),
        ('NumAromaticRings', Descriptors.NumAromaticRings),
        ('TPSA', Descriptors.TPSA),
        ('BertzCT', Descriptors.BertzCT),
        ('Ipc', Descriptors.Ipc),
        ('HeavyAtomCount', Descriptors.HeavyAtomCount)
    ]
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                feature_dict = {'SMILES': smiles}
                
                # Calculate basic descriptors
                for name, func in basic_descriptors:
                    try:
                        feature_dict[name] = func(mol)
                    except:
                        feature_dict[name] = np.nan
                
                # Add more descriptors based on descriptor_set
                if descriptor_set in ['extended', 'all']:
                    try:
                        feature_dict['Chi0v'] = Descriptors.Chi0v(mol)
                        feature_dict['Chi1v'] = Descriptors.Chi1v(mol)
                        feature_dict['BalabanJ'] = Descriptors.BalabanJ(mol)
                        feature_dict['Kappa1'] = Descriptors.Kappa1(mol)
                        feature_dict['Kappa2'] = Descriptors.Kappa2(mol)
                    except:
                        pass
                
                features.append(feature_dict)
            else:
                # Invalid SMILES
                feature_dict = {'SMILES': smiles}
                for name, _ in basic_descriptors:
                    feature_dict[name] = np.nan
                features.append(feature_dict)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            continue
    
    return pd.DataFrame(features)


def plot_molecular_diversity(mol_features: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)):
    """
    Plot molecular diversity analysis.
    
    Parameters:
    -----------
    mol_features : pd.DataFrame
        DataFrame with molecular features
    figsize : Tuple[int, int]
        Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Molecular Diversity Analysis', fontsize=16)
    
    # Molecular weight distribution
    axes[0, 0].hist(mol_features['MolWt'].dropna(), bins=30, alpha=0.7)
    axes[0, 0].set_title('Molecular Weight Distribution')
    axes[0, 0].set_xlabel('Molecular Weight (Da)')
    
    # LogP distribution
    axes[0, 1].hist(mol_features['LogP'].dropna(), bins=30, alpha=0.7)
    axes[0, 1].set_title('LogP Distribution')
    axes[0, 1].set_xlabel('LogP')
    
    # TPSA distribution
    axes[0, 2].hist(mol_features['TPSA'].dropna(), bins=30, alpha=0.7)
    axes[0, 2].set_title('TPSA Distribution')
    axes[0, 2].set_xlabel('TPSA (Ų)')
    
    # Molecular weight vs LogP
    axes[1, 0].scatter(mol_features['MolWt'], mol_features['LogP'], alpha=0.6)
    axes[1, 0].set_title('Molecular Weight vs LogP')
    axes[1, 0].set_xlabel('Molecular Weight (Da)')
    axes[1, 0].set_ylabel('LogP')
    
    # Heavy atom count distribution
    axes[1, 1].hist(mol_features['HeavyAtomCount'].dropna(), bins=20, alpha=0.7)
    axes[1, 1].set_title('Heavy Atom Count Distribution')
    axes[1, 1].set_xlabel('Heavy Atom Count')
    
    # Number of aromatic rings
    if 'NumAromaticRings' in mol_features.columns:
        ring_counts = mol_features['NumAromaticRings'].value_counts().sort_index()
        axes[1, 2].bar(ring_counts.index, ring_counts.values, alpha=0.7)
        axes[1, 2].set_title('Aromatic Rings Distribution')
        axes[1, 2].set_xlabel('Number of Aromatic Rings')
        axes[1, 2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()