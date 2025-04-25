# Surrogate-Assisted Feature Extraction (SAFE)

A Python implementation of surrogate-assisted feature extraction and analysis for complex datasets. This tool helps in understanding feature importance, interactions, and their impact on target variables through surrogate modeling techniques.

## Features

- Surrogate model-based feature importance analysis
- Feature interaction detection and quantification
- Granger causality testing for time series features
- LOFO (Leave One Feature Out) importance calculation
- Comprehensive report generation
- Support for various model types (Random Forest, XGBoost)
- Time series analysis capabilities
- Interactive visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/surrogate-assisted-feature-extraction-gemini.git
cd surrogate-assisted-feature-extraction-gemini
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from safe_analysis import ReportGenerator

# Initialize the report generator
report_gen = ReportGenerator(
    output_dir='safe_analysis',
    surrogate_model_type='random_forest'
)

# Generate analysis report
report_gen.generate_report()
```

### Configuration

The main configuration parameters can be set in the script:

- `DATA_PATH`: Path to the input data file
- `TARGET_VARIABLE`: Name of the target variable
- `DATE_COLUMN`: Name of the date column for time series analysis
- `GROUPING_COLUMN`: Optional column for filtering data
- `TEST_MONTHS`: Number of months for test set
- `SURROGATE_MODEL_TYPE`: Type of surrogate model ('random_forest' or 'xgboost')
- `N_TOP_FEATURES`: Number of top features to include in summaries
- `EXOGENOUS_VARIABLES`: List of variables to include in analysis (default: all)

## Project Structure

```
.
├── safe_analysis.py       # Main implementation
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
├── _data/                # Data directory
└── safe_analysis/        # Output directory for analysis results
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on surrogate modeling techniques for feature importance analysis
- Inspired by SHAP (SHapley Additive exPlanations) values
- Utilizes scikit-learn and XGBoost for model implementation
