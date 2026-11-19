# Micromobility injury severity modelling

Python code for the 2022–2024 bicycle and e-scooter injury severity analysis in manuscript v14. The models estimate the probability that a casualty sustained a fatal or serious injury. The analysis compares how risk factors vary across years and how well a model fitted to one mode or year predicts injury severity in another.

## Run the analysis

Use Python 3.11 and run these commands from the directory containing `run_analysis.py`. Replace the example path with the directory holding the six study workbooks.

```bash
python -m pip install -r requirements.txt
python run_analysis.py "/path/to/workbooks"
```

The runner prepares the data, fits the models, runs the temporal and transferability comparisons, and then runs the cross-mode likelihood-ratio test. It writes analysis samples to `prepared/` and outputs to `results/`, both alongside the scripts. Rerunning the analysis overwrites its generated files.

## Input data

Supply the six study workbooks separately. The preparation script reads the first worksheet of each file and expects the study's linked collision, vehicle and casualty fields and pre-coded indicators. The table gives the expected sample sizes after filtering.

| Year | Bicycle workbook | Retained records | E-scooter workbook | Retained records |
| --- | --- | ---: | --- | ---: |
| 2022 | `cyclist2022.xlsx` | 11,484 | `escooter2022.xlsx` | 1,337 |
| 2023 | `cyclist2023.xlsx` | 6,723 | `escooter2023.xlsx` | 834 |
| 2024 | `cyclist2024.xlsx` | 6,458 | `escooter2024.xlsx` | 900 |

A record enters the analysis when its three collision identifiers agree, its three year fields match the workbook year, and its casualty severity is coded 1, 2 or 3. The binary outcome is **1 for fatal or serious injury** (source codes 1 and 2) and **0 for slight injury** (source code 3). The six estimation samples contain 27,736 records in total. Each prepared CSV retains the original Excel row number in `excel_row`.

## Model specifications

`specifications.json` contains the selected terms, random parameters, cross-year coefficient constraints and archived coefficient estimates used to initialise the refits. `model_design.py` constructs the design matrices, `model_estimation.py` implements estimation and prediction using xlogit, and `fit_models.py` fits each specification. Mixed logit estimation uses normally distributed random coefficients and 1,000 unshuffled Halton draws after dropping the first 100.

| Specification suffix | Role in the manuscript |
| --- | --- |
| `_annual` | Six mode- and year-specific models for Tables A1–A6, the Table 3 comparisons and the Figure 3 predictions. |
| `_partial` | One model per mode pooling the three years, with selected coefficients shared across years. These produce Tables 4 and 5. |
| `_unrestricted` | Year-specific models with the corresponding partially constrained model's annual terms. Equation 7 uses these fits to test the cross-year equality constraints. |

## Manuscript outputs

All paths in the table are relative to `results/`.

| Manuscript result | Output | Script |
| --- | --- | --- |
| Sample counts and input file hashes | `sample_flow.csv`, `source_manifest.json` | `prepare_data.py` |
| Table 2 descriptive statistics | `table2.csv` | `prepare_data.py` |
| Table 3 directional model transfers | `table3.csv` | `statistical_tests.py` |
| Tables 4 and 5 coefficients and marginal effects | `table4.csv` (bicycle), `table5.csv` (e-scooter) | `statistical_tests.py` |
| Tables A1–A6 annual estimates | `{mode}{year}_appendix.csv` | `statistical_tests.py` |
| Equation 7 temporal constraint tests | `cyclist_temporal_test.json`, `escooter_temporal_test.json` | `statistical_tests.py` |
| Figure 3 prediction differences | `figure3_observation_differences.csv`, `figure3_summary.json` | `statistical_tests.py` |
| Combined versus mode-specific likelihood-ratio test | `combined_test/summary.json` | `cross_mode_test.py` |

In Table 3, `source` identifies the fitted model and `target` identifies the sample used to evaluate it. The `confidence` field is a descriptive chi-square percentile for comparisons between independently specified models. For Figure 3, a positive difference means the transferred model predicts a higher fatal-or-serious injury probability than the target sample's own model.

## Reproducibility

The verification run reproduced the reported log likelihoods for the six appendix models and both partially constrained models at manuscript precision. The combined versus mode-specific test gave a likelihood-ratio statistic of **235.14** with **76 degrees of freedom**. Two Table 3 statistics differed from v14 by 0.01 after rounding, both for transfers to the 2022 bicycle sample.
