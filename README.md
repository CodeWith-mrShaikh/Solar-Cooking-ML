# Solar Cooking Machine Learning Prediction System (PART 6)

A professional Machine Learning analytics and real-time prediction dashboard for solar thermal cooking experiments. The system predicts transient food cooking temperatures ($T_{food}$) from solar irradiance, ambient temperature, absorber plate temperature, cooking time, food mass, food matrix, and solar cooker geometry.

---

## 1. Project Overview
Solar cooking provides a sustainable, carbon-free alternative to fossil fuels and biomass cooking. However, solar cooking processes exhibit high non-linearity due to fluctuating solar flux, ambient convective losses, and thermal inertia. This BTech/EPICS academic project implements a supervised machine learning model (Extra Trees Regressor) achieving an $R^2$ of **99.74%** on validation data to accurately predict food temperatures without requiring complex computational fluid dynamics (CFD) simulations.

---

## 2. Dataset Description
- **Source:** Experimental solar cooking campaigns from multiple experimental runs.
- **Canonical Dataset Location:** `data/final_training_dataset.csv`
- **Total Resampled Observations:** 5,803 rows
- **Columns (11):**
  1. `time_min` (Integer, minutes from start)
  2. `run_id` (String, experimental campaign identifier)
  3. `solar_irradiance_W_m2` (Float, solar radiation flux)
  4. `ambient_temp_C` (Float, atmospheric temperature)
  5. `absorber_temp_C` (Float, collector plate temperature)
  6. `food_temp_C` (Float, target variable)
  7. `food_matrix` (Categorical, food medium)
  8. `cooker_type` (Categorical, cooker geometry)
  9. `mass_kg` (Float, cooking mass in kg)
  10. `source_file` (Metadata, provenance citation)
  11. `Unnamed: 10` (Dropped metadata)

---

## 3. Resampling Methodology
Raw experimental recordings featured irregular sensor sampling rates and communication drops. The data preparation pipeline:
1. Segregates records by `run_id`.
2. Reindexes each run to a continuous 1-minute temporal grid: $\text{arange}(\min(t), \max(t) + 1)$.
3. Forward-fills and backward-fills invariant metadata (`run_id`, `food_matrix`, `cooker_type`, `mass_kg`, `source_file`).
4. Linearly interpolates continuous physical sensor telemetry (`solar_irradiance_W_m2`, `ambient_temp_C`, `absorber_temp_C`, `food_temp_C`).
5. Prunes rows where target values remain unresolvable, yielding the final 5,803-row matrix.

---

## 4. Model Features (7 Canonical Inputs)
The production model accepts exactly 7 ordered inputs:
1. `time_min` — Elapsed cooking time (min)
2. `solar_irradiance_W_m2` — Incident solar flux (W/m²)
3. `ambient_temp_C` — Ambient air temperature (°C)
4. `absorber_temp_C` — Absorber plate temperature (°C)
5. `mass_kg` — Net food / water mass (kg)
6. `food_matrix` — Food substrate category (e.g., Water, Potato chips, Rice)
7. `cooker_type` — Solar cooker geometry (e.g., box, panel, parabolic)

**Target Variable:** `food_temp_C` (Instantaneous food temperature in °C)

The canonical feature ordering is stored in `models/feature_config.json`.

---

## 5. Preprocessing Pipeline
- **Categorical Factors:** Encoded using fitted `LabelEncoder` objects (`food_matrix_encoder.joblib`, `cooker_type_encoder.joblib`).
- **Missing Value Handling:** Handled via fitted `SimpleImputer(strategy='mean')` (`imputer.joblib`).
- **Scaling:** Continuous variables scaled via fitted `StandardScaler()` (`scaler.joblib`).
- **Zero Runtime Re-fitting:** The GUI strictly uses `.transform()` on pre-fitted artifacts and never calls `.fit()` or `.fit_transform()`.

---

## 6. Models Evaluated
Six regression algorithms were benchmarked across two independent train/test splits:
1. Linear Regression (OLS)
2. Ridge Regression ($\text{L}_2$ regularization)
3. Random Forest Regressor ($n=100$)
4. Gradient Boosting Regressor
5. Extra Trees Regressor ($n=100$)
6. Extreme Gradient Boosting (XGBoost Regressor)

---

## 7. Model Evaluation Results
Results obtained from the baseline Google Colab pipeline:

### 80:20 Train/Test Split
| Model | Test $R^2$ | Test MSE | Overfitting Diff |
| :--- | :---: | :---: | :---: |
| **Extra Trees** | **0.997352** | **1.362492** | **0.002648** |
| **XGBoost** | 0.996554 | 1.772795 | 0.002767 |
| **Random Forest** | 0.995648 | 2.239038 | 0.003773 |
| **Gradient Boosting** | 0.980221 | 10.176398 | 0.008601 |
| **Linear Regression** | 0.847437 | 78.493719 | -0.025582 |
| **Ridge** | 0.847434 | 78.495313 | -0.025579 |

### 70:30 Train/Test Split
| Model | Test $R^2$ | Test MSE | Overfitting Diff |
| :--- | :---: | :---: | :---: |
| **Extra Trees** | **0.997067** | **1.505324** | **0.002933** |
| **Random Forest** | 0.996744 | 1.671054 | 0.002467 |
| **XGBoost** | 0.996314 | 1.891805 | 0.003473 |
| **Gradient Boosting** | 0.984489 | 7.961237 | 0.004442 |
| **Linear Regression** | 0.852089 | 75.916210 | -0.036210 |
| **Ridge** | 0.852075 | 75.923429 | -0.036196 |

> [!NOTE]
> $R^2$ represents the coefficient of determination (explained variance ratio) and is **never** labeled as "accuracy" or "model confidence".

---

## 8. Selected Production Model
- **Algorithm:** Extra Trees Regressor (`sklearn.ensemble.ExtraTreesRegressor`)
- **Hyperparameters:** `n_estimators=100`, `random_state=42`
- **Evaluation Split:** 80:20 ($test\_size=0.20$)
- **Validation Metrics:** $R^2 = 0.997352$ ($99.74\%$), $\text{MSE} = 1.362492$

---

## 9. Model Export Workflow
To export the model artifacts:
- **Local Generation:** Run `python scripts/export_model_artifacts.py`
- **From Google Colab:** Execute the cell snippet provided in `colab/colab_export_snippet.py`. Download `models.zip` and extract into `models/`.

Required files in `models/`:
- `extra_trees_model.joblib`
- `imputer.joblib`
- `scaler.joblib`
- `food_matrix_encoder.joblib`
- `cooker_type_encoder.joblib`
- `feature_config.json`
- `model_metadata.json`

---

## 10. Folder Structure
```
pro1/
│
├── app.py                             # Main Streamlit web application
│
├── models/                            # Production artifacts (pre-fitted)
│   ├── extra_trees_model.joblib
│   ├── imputer.joblib
│   ├── scaler.joblib
│   ├── food_matrix_encoder.joblib
│   ├── cooker_type_encoder.joblib
│   ├── feature_config.json
│   └── model_metadata.json
│
├── data/
│   └── final_training_dataset.csv     # Canonical resampled dataset
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Constants, features, benchmark metrics
│   ├── data_loader.py                 # Cached dataset and stats loader
│   ├── model_loader.py                # Cached artifact loading & health status
│   ├── predictor.py                   # Single source of truth for inference
│   └── validation.py                  # Numerical & categorical boundary checks
│
├── pages/
│   ├── __init__.py
│   ├── dashboard_view.py              # Operational KPIs & model overview
│   ├── prediction_view.py             # 2-column input UI, warnings & history
│   ├── eda_view.py                    # Plotly charts: heatmaps, runs, distributions
│   ├── model_performance_view.py      # Benchmark comparisons & feature importance
│   └── about_view.py                  # Thermodynamic background & science notes
│
├── scripts/
│   ├── export_model_artifacts.py      # Artifact generator from canonical data
│   └── validate_model_artifacts.py    # Automated production readiness tests
│
├── colab/
│   └── colab_export_snippet.py        # Drop-in Colab cell export script
│
├── requirements.txt                   # Application dependencies
└── README.md                          # Full system documentation
```

---

## 11. Installation
1. Clone or navigate to the project directory:
   ```bash
   cd "d:/College/Epic proj/pro1"
   ```
2. Activate your virtual environment (or create one):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # On Windows
   # source .venv/bin/activate # On Linux/macOS
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 12. Running the GUI
Launch the Streamlit web application:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

---

## 13. Prediction Workflow
1. Navigate to **🔮 Prediction** in the sidebar.
2. Enter values in the two columns:
   - **Solar / Environmental Conditions:** Time (min), Solar Irradiance (W/m²), Ambient Temperature (°C).
   - **Thermal / Cooking Parameters:** Absorber Plate Temperature (°C), Mass (kg), Food Matrix (dropdown), Cooker Type (dropdown).
3. If an input falls outside the historical training range, a non-blocking warning alert appears.
4. Click **🚀 Predict Food Temperature**.
5. The predicted temperature appears in a prominent card along with validation $R^2$ ($99.74\%$) and inference latency in milliseconds.
6. Review prediction history or download session logs as CSV.

---

## 14. EDA Functionality
Navigate to **📈 EDA Dashboard** to interact with:
- **Dataset Overview:** Tabular preview, data types, null count diagnostics.
- **Target & Feature Distributions:** Interactive histograms, boxplots, and statistical summaries.
- **Correlation Heatmap:** Strictly numerical correlation matrix ($r$).
- **Run-Level Time Series:** Dynamic selection of any of the 16 experimental runs, displaying synchronized temperature and irradiance profiles.
- **Food & Cooker Analysis:** Categorical frequency and temperature breakdown.

---

## 15. Troubleshooting & FAQ
- **Missing Artifacts Warning:** If the sidebar displays `Model ✗ Missing`, run:
  ```bash
  python scripts/export_model_artifacts.py
  ```
- **Scikit-learn Feature Name Warnings:** Eliminated by passing a named `pandas.DataFrame` matching `models/feature_config.json` directly into `imputer.transform()`.
- **Validation Script:** Run automated sanity checks at any time:
  ```bash
  python scripts/validate_model_artifacts.py
  ```
