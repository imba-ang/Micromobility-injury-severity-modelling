# Micromobility injury severity modelling

Python code for the 2022–2024 bicycle and e-scooter injury severity analysis. The models estimate the probability that a casualty sustained a fatal or serious injury. The analysis compares how risk factors vary across years and how well a model fitted to one mode or year predicts injury severity in another.

## Run the analysis

Use Python 3.11 and run these commands from the directory containing `run_analysis.py`. Replace the example path with the directory holding the Excel files.

```bash
python -m pip install -r requirements.txt
python run_analysis.py "/path/to/workbooks"
```

## Model specifications

`specifications.json` contains the selected terms, random parameters, cross-year coefficient constraints, and coefficient estimates. `model_design.py` constructs the design matrices, `model_estimation.py` implements estimation and prediction using xlogit, and `fit_models.py` fits each specification. Mixed logit estimation uses normally distributed random coefficients and 1,000 Halton draws after dropping the first 100.
