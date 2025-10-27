# Factor Analysis for Asset Pricing Models

This project implements comprehensive factor analysis for various asset pricing models including CAPM, Fama-French 3-factor, and Fama-French 5-factor models with momentum using time series and Fama-MacBeth two-stage regression methodologies.

## Overview

This analysis implements academic asset pricing models to evaluate portfolio returns across different factor structures. The project includes:

1. **Time Series Regressions**: Estimates factor loadings (betas) for each portfolio
2. **Fama-MacBeth Two-Stage Regressions**: Implements the classic Fama-MacBeth methodology
3. **Multiple Model Comparison**: Evaluates CAPM, FF3, FF3+MOM, and baseline models
4. **Statistical Analysis**: Summary statistics, correlations, and model diagnostics

## Project Structure

```
Advanced Asset Management/
├── factor_analysis.py    # Main analysis script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── Data_assignmentAAM2025.xlsx           # Factor data (input)
├── 25_Portfolios_ME_INV_5x5_Excel.xlsx   # Portfolio data (input)
└── analysis_results.xlsx                  # Analysis results (output)
```

## Features

### Asset Pricing Models

1. **CAPM (Capital Asset Pricing Model)**
   - Single factor: Market risk premium (Mkt-RF)
   - Classic Sharpe-Lintner model

2. **Fama-French 3-Factor Model (FF3)**
   - Market factor (Mkt-RF)
   - Size factor (SMB - Small Minus Big)
   - Value factor (HML - High Minus Low)

3. **Fama-French 3-Factor with Momentum (FF3+MOM)**
   - All FF3 factors plus
   - Momentum factor (MOM)

4. **Baseline Model**
   - Market factor (Mkt-RF)
   - Size factor (SMB)
   - Value factor (HML)
   - Investment factor (CMA - Conservative Minus Aggressive)

### Analysis Components

- **Time Series Regression**: Estimates factor loadings for each portfolio
- **Fama-MacBeth Two-Stage**: Cross-sectional analysis of risk premiums
- **Summary Statistics**: Descriptive statistics for factors and portfolios
- **Correlation Analysis**: Relationship between factors and portfolio returns
- **Model Comparison**: R-squared comparison across models

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Navigate to the project directory:
```bash
cd "MIF 2024/Advanced Asset Management"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis

Run the complete factor analysis:

```bash
python factor_analysis.py
```

This will:
- Load factor and portfolio data from Excel files
- Calculate excess returns (returns minus risk-free rate)
- Perform time series regressions for each model
- Perform Fama-MacBeth two-stage regressions
- Generate summary statistics and correlations
- Save results to `analysis_results.xlsx`
- Display results in the console

### Input Files

Place the following Excel files in the project directory:

- `Data_assignmentAAM2025.xlsx`: Contains factor returns including:
  - Risk-free rate (RF)
  - Market factor (Mkt-RF)
  - Size factor (SMB)
  - Value factor (HML)
  - Investment factor (CMA)
  - Momentum factor (MOM)

- `25_Portfolios_ME_INV_5x5_Excel.xlsx`: Contains 25 portfolio returns formed by:
  - 5 Size groups (based on market equity)
  - 5 Investment groups (based on investment ratio)

### Output Files

The analysis generates `analysis_results.xlsx` with multiple sheets:

1. **Factor_Stats**: Summary statistics for factors
2. **Portfolio_Stats**: Summary statistics for portfolio returns
3. **Correlations**: Correlation matrix between factors and portfolios
4. **Model_Comparison**: R-squared values for each model

## Configuration

### Date Range

The analysis covers the period from July 1963 to December 2024:

```python
START_DATE = datetime(1963, 7, 1)
END_DATE = datetime(2024, 12, 31)
```

Modify these constants in `factor_analysis.py` to change the analysis period.

### Model Selection

Available models:
- `'base'`: Market, SMB, HML, CMA factors
- `'capm'`: Market factor only
- `'ff3'`: Market, SMB, HML factors
- `'ff3_mom'`: Market, SMB, HML, MOM factors

## Key Functions

### Data Loading and Processing

- `load_data()`: Loads and filters data from Excel files
- `calculate_excess_returns()`: Subtracts risk-free rate from returns

### Regression Analysis

- `time_series_regression()`: Performs time series regressions for each portfolio
- `fama_macbeth_regression()`: Implements Fama-MacBeth two-stage methodology

### Statistics and Output

- `generate_summary_statistics()`: Creates summary statistics and correlations
- `save_results()`: Saves results to Excel file
- `print_results()`: Displays results in console

## Understanding the Results

### Time Series Regressions

For each portfolio, the time series regression estimates:
- **Alpha (α)**: Abnormal return not explained by factors
- **Beta (β)**: Sensitivity to each factor
- **R-squared**: Proportion of variance explained by factors

### Fama-MacBeth Methodology

**Stage 1**: Time series regressions estimate betas for each portfolio
**Stage 2**: Cross-sectional regression of average returns on betas to estimate risk premiums

The second stage answers: "Do factor risks earn premiums?"

### Interpreting Results

- **Significant factors**: Factors with statistically significant coefficients
- **R-squared**: How well the model explains portfolio returns
- **Alpha**: Whether portfolios earn abnormal returns beyond factor risks
- **Risk premiums**: Compensation for bearing systematic risks

## Customization

### Adding New Models

To add a new model, update the `model_factors` dictionary:

```python
model_factors = {
    'your_model': ['factor1', 'factor2', 'factor3'],
    ...
}
```

### Modifying Analysis Period

Change the date constants:

```python
START_DATE = datetime(YYYY, M, D)
END_DATE = datetime(YYYY, M, D)
```

### Customizing Output

Modify the `save_results()` function to add or remove output sheets.

## Theory Background

### Fama-French Models

The Fama-French models extend CAPM by adding factors that explain stock returns:

**Fama-French 3-Factor Model:**
```
R_i - R_f = α_i + β₁(R_m - R_f) + β₂SMB + β₃HML + ε_i
```

**Fama-French 5-Factor Model:**
Adds profitability (RMW) and investment (CMA) factors

**With Momentum:**
Adds momentum (MOM) factor to capture past performance effects

### Fama-MacBeth Procedure

1. **Stage 1**: Estimate betas for each asset using time series data
2. **Stage 2**: Run cross-sectional regressions of average returns on betas
3. **Stage 3**: Average coefficients across time to get risk premiums

This addresses errors-in-variables issues in cross-sectional regressions.

## Requirements

See `requirements.txt` for complete list of dependencies. Key packages:
- pandas: Data manipulation
- numpy: Numerical computations
- statsmodels: Regression analysis
- matplotlib/seaborn: Visualizations
- openpyxl: Excel file handling

## Troubleshooting

### Common Issues

1. **File not found error**:
   - Ensure Excel files are in the correct directory
   - Check file names match exactly

2. **Date parsing errors**:
   - Verify date formats in Excel files
   - Ensure dates are in correct column

3. **Memory errors**:
   - Reduce analysis period
   - Use fewer portfolios or factors

4. **NaN values**:
   - Check for missing data in input files
   - Some models may not work with incomplete data

## Author

Janek Masojada Edwards

## Course

MIF 2024 - Advanced Asset Management
