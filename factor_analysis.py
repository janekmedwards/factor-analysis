"""
Factor Analysis for Asset Pricing Models

This module implements time series and cross-sectional regression analysis
for various asset pricing models including CAPM, Fama-French 3-factor, and
Fama-French 5-factor models with momentum.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Constants
START_DATE = datetime(1963, 7, 1)
END_DATE = datetime(2024, 12, 31)


def load_data(factors_file='Data_assignmentAAM2025.xlsx', portfolios_file='25_Portfolios_ME_INV_5x5_Excel.xlsx'):
    """
    Load and process data from Excel files.
    
    Parameters:
    - factors_file: Path to factor data Excel file
    - portfolios_file: Path to portfolios Excel file
    
    Returns:
    - Tuple of (factors_df, portfolios_df)
    """
    factors_df = pd.read_excel(factors_file)
    portfolios_df = pd.read_excel(portfolios_file)
    
    factors_df['Date'] = pd.to_datetime(factors_df['Date'])
    portfolios_df['Date'] = pd.to_datetime(portfolios_df['Date'])
    
    # Filter data for the required period
    factors_df = factors_df[(factors_df['Date'] >= START_DATE) & (factors_df['Date'] <= END_DATE)]
    portfolios_df = portfolios_df[(portfolios_df['Date'] >= START_DATE) & (portfolios_df['Date'] <= END_DATE)]
    
    # Convert returns to decimals if they're in percentages
    for col in portfolios_df.columns:
        if col != 'Date' and portfolios_df[col].max() > 1:
            portfolios_df[col] = portfolios_df[col] / 100
    
    return factors_df, portfolios_df


def calculate_excess_returns(portfolios_df, factors_df):
    """
    Calculate excess returns by subtracting risk-free rate.
    
    Parameters:
    - portfolios_df: DataFrame containing portfolio returns
    - factors_df: DataFrame containing risk-free rate
    
    Returns:
    - DataFrame of excess returns
    """
    rf = factors_df['RF']
    excess_returns = portfolios_df.copy()
    
    for col in excess_returns.columns:
        if col != 'Date':
            excess_returns[col] = excess_returns[col] - rf
    
    return excess_returns


def time_series_regression(factors_df, excess_returns, model_type='base'):
    """
    Perform time series regressions for each portfolio.
    
    Parameters:
    - factors_df: DataFrame containing factor returns
    - excess_returns: DataFrame containing portfolio excess returns
    - model_type: Type of model ('base', 'capm', 'ff3', 'ff3_mom')
    
    Returns:
    - Dictionary of regression results for each portfolio
    """
    model_factors = {
        'base': ['Mkt-RF', 'SMB', 'HML', 'CMA'],
        'capm': ['Mkt-RF'],
        'ff3': ['Mkt-RF', 'SMB', 'HML'],
        'ff3_mom': ['Mkt-RF', 'SMB', 'HML', 'MOM']
    }
    
    factors = model_factors.get(model_type, model_factors['base'])
    results = {}
    
    for col in excess_returns.columns:
        if col != 'Date':
            X = sm.add_constant(factors_df[factors])
            y = excess_returns[col]
            
            model = OLS(y, X)
            results[col] = model.fit()
    
    return results


def fama_macbeth_regression(factors_df, excess_returns, model_type='base'):
    """
    Perform Fama-MacBeth two-stage regressions.
    
    Parameters:
    - factors_df: DataFrame containing factor returns
    - excess_returns: DataFrame containing portfolio excess returns
    - model_type: Type of model ('base', 'capm', 'ff3', 'ff3_mom')
    
    Returns:
    - OLS regression results from second stage
    """
    # First stage: estimate betas using time series regressions
    ts_results = time_series_regression(factors_df, excess_returns, model_type)
    
    # Extract betas for each portfolio
    betas = {}
    for portfolio, result in ts_results.items():
        betas[portfolio] = result.params[1:]  # Exclude constant
    
    # Second stage: cross-sectional regression
    avg_returns = excess_returns.mean()
    
    # Prepare data for cross-sectional regression
    X = sm.add_constant(pd.DataFrame(betas).T)
    y = avg_returns
    
    # Run cross-sectional regression
    model = OLS(y, X)
    results = model.fit()
    
    return results


def generate_summary_statistics(factors_df, excess_returns):
    """
    Generate summary statistics for factors and portfolio returns.
    
    Parameters:
    - factors_df: DataFrame containing factor returns
    - excess_returns: DataFrame containing portfolio excess returns
    
    Returns:
    - Tuple of (factor_stats, portfolio_stats, correlation_matrix)
    """
    factor_stats = factors_df.describe()
    portfolio_stats = excess_returns.describe()
    correlation_matrix = pd.concat([factors_df, excess_returns], axis=1).corr()
    
    return factor_stats, portfolio_stats, correlation_matrix


def save_results(factor_stats, portfolio_stats, correlation_matrix, base_fm, capm_fm, ff3_fm, ff3_mom_fm):
    """
    Save analysis results to Excel file.
    
    Parameters:
    - factor_stats: Summary statistics for factors
    - portfolio_stats: Summary statistics for portfolios
    - correlation_matrix: Correlation matrix
    - base_fm, capm_fm, ff3_fm, ff3_mom_fm: Fama-MacBeth regression results
    
    Returns:
    - None (saves to Excel file)
    """
    with pd.ExcelWriter('analysis_results.xlsx') as writer:
        factor_stats.to_excel(writer, sheet_name='Factor_Stats')
        portfolio_stats.to_excel(writer, sheet_name='Portfolio_Stats')
        correlation_matrix.to_excel(writer, sheet_name='Correlations')
        
        pd.DataFrame({
            'Model': ['Base', 'CAPM', 'FF3', 'FF3+MOM'],
            'R-squared': [
                base_fm.rsquared,
                capm_fm.rsquared,
                ff3_fm.rsquared,
                ff3_mom_fm.rsquared
            ]
        }).to_excel(writer, sheet_name='Model_Comparison')


def print_results(factor_stats, portfolio_stats, correlation_matrix, base_fm):
    """Print analysis results to console."""
    print("\nFactor Summary Statistics:")
    print(factor_stats)
    
    print("\nPortfolio Return Summary Statistics:")
    print(portfolio_stats)
    
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    print("\nFama-MacBeth Regression Results (Base Model):")
    print(base_fm.summary())


def main():
    """Main execution function."""
    # Load and process data
    factors_df, portfolios_df = load_data()
    
    # Calculate excess returns
    excess_returns = calculate_excess_returns(portfolios_df, factors_df)
    
    # Generate summary statistics
    factor_stats, portfolio_stats, correlation_matrix = generate_summary_statistics(
        factors_df, excess_returns
    )
    
    # Perform time series regressions
    base_model_results = time_series_regression(factors_df, excess_returns, 'base')
    capm_results = time_series_regression(factors_df, excess_returns, 'capm')
    ff3_results = time_series_regression(factors_df, excess_returns, 'ff3')
    ff3_mom_results = time_series_regression(factors_df, excess_returns, 'ff3_mom')
    
    # Perform Fama-MacBeth regressions
    base_fm_results = fama_macbeth_regression(factors_df, excess_returns, 'base')
    capm_fm_results = fama_macbeth_regression(factors_df, excess_returns, 'capm')
    ff3_fm_results = fama_macbeth_regression(factors_df, excess_returns, 'ff3')
    ff3_mom_fm_results = fama_macbeth_regression(factors_df, excess_returns, 'ff3_mom')
    
    # Print results
    print_results(factor_stats, portfolio_stats, correlation_matrix, base_fm_results)
    
    # Save results to Excel
    save_results(
        factor_stats, portfolio_stats, correlation_matrix,
        base_fm_results, capm_fm_results, ff3_fm_results, ff3_mom_fm_results
    )


if __name__ == "__main__":
    main()