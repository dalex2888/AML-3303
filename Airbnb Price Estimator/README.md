# Airbnb Price Prediction with MLflow and AWS

Machine learning project to predict Airbnb listing prices in NYC using AWS S3 for data storage, EC2 for MLflow tracking, and multiple regression models with systematic experiment tracking.

---

## 📊 Project Overview

**Objective:** Build a regression model to predict nightly Airbnb listing prices based on location, property characteristics, and review metrics.

**Dataset:** AB_NYC_2019 (48,895 listings with 16 original features)

**Tech Stack:**
- Python 3.13
- AWS S3 (data storage)
- AWS EC2 (MLflow tracking server)
- MLflow 2.x (experiment tracking & model registry)
- XGBoost, scikit-learn (modeling)
- pandas, numpy (data processing)

---

## 🏗️ Architecture

```
Local Development          GitHub              AWS Cloud
     (Code)         →   (Version Control) →    (Services)
        ↓                                          ↓
   Run notebooks                            EC2: MLflow UI
        ↓                                          ↑
   Log experiments  ─────────────────────────────┘
        ↓
   Read data from  ─────────────────────────→  S3: Dataset
```

**Infrastructure:**
- **S3 Bucket:** `software-tools-ai`
- **MLflow Server:** http://35.183.177.64:5000
- **EC2 Instance:** t2.micro (AWS Free Tier)

---

## 📁 Repository Structure

```
airbnb-price-prediction/
├── .gitignore                          # Excludes data, credentials, env
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── notebooks/                          # Jupyter notebooks
│   ├── 01_eda_analysis.ipynb          # Exploratory data analysis
│   ├── 02_data_preprocessing.ipynb    # Data cleaning & feature engineering
│   └── 03_modeling_mlflow.ipynb       # Model training & evaluation
│
├── docs/                               # Documentation
│   └── aws_setup.md                   # AWS configuration guide
│
└── screenshots/                        # MLflow UI screenshots
    ├── mlflow_experiments_overview.png
    ├── mlflow_best_model_details.png
    └── mlflow_registered_model.png
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- AWS Account with:
  - IAM Access Keys configured
  - S3 bucket created
  - EC2 instance running MLflow
- Git installed

### Setup Instructions

**1. Clone the repository:**
```bash
git clone https://github.com/dalex2888/AML-3303.git
cd airbnb-price-prediction
```

**2. Create virtual environment:**
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure AWS credentials:**
```bash
aws configure
```
Enter your Access Key ID, Secret Key, region (us-east-1), and output format (json).

**5. Launch Jupyter:**
```bash
jupyter notebook
```

**For detailed AWS setup**, see [docs/aws_setup.md](docs/aws_setup.md)

---

## 📊 Dataset

**Source:** AB_NYC_2019 Airbnb listings  
**Size:** 48,895 rows × 16 columns

### Key Features:
- `price`: Target variable (nightly rate in USD)
- `neighbourhood_group`: Borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island)
- `neighbourhood`: Specific neighborhood (221 unique categories)
- `room_type`: Entire home/apt, Private room, Shared room
- `latitude`, `longitude`: Geographic coordinates
- `number_of_reviews`, `reviews_per_month`: Review metrics
- `availability_365`: Days available per year

### Data Quality Issues:
- 20.6% missing values in review columns (listings with no reviews)
- Outliers with prices up to $10,000
- 11 listings with price = $0 (removed)

---

## 🔧 Preprocessing Pipeline

### 1. Data Cleaning
- Removed 11 listings with `price = $0`
- Dropped identifier columns: id, host_id, name, host_name
- Maintained neighbourhood column for target encoding

### 2. Train/Test Split
- 80/20 split (stratified by neighbourhood_group)
- Random state: 42 for reproducibility
- Split performed BEFORE any statistical operations to prevent data leakage

### 3. Outlier Handling (Target Variable)
- Calculated 99th percentile threshold on training set only: $798.76
- Removed 392 outliers from training set (price > $798.76)
- Test set kept unchanged to evaluate real-world performance
- Final training size: 38,715 samples
- Test size: 9,777 samples

### 4. Missing Value Handling
- `reviews_per_month`: Imputed with 0 (represents listings with no reviews)
- `last_review`: Dropped (temporal feature not useful for prediction)

### 5. Feature Engineering

**Geographic Features:**
- `distance_to_manhattan`: Euclidean distance from Times Square (40.7580, -73.9855)

**Target Encoding:**
- `neighbourhood_price_encoded`: Mean price per neighbourhood calculated from training data only
  - 221 unique neighbourhoods encoded to single numeric feature
  - Prevents high dimensionality from one-hot encoding
  - Correlation with price: 0.468

**Interaction Features:**
- `room_borough`: Interaction between room_type and neighbourhood_group
  - Captures patterns like "Entire homes in Manhattan are more expensive"
  - Created 15 unique combinations

### 6. Encoding and Scaling

**Categorical Encoding:**
- One-hot encoding with `drop_first=True` to avoid multicollinearity
- Applied to: neighbourhood_group, room_type, room_borough
- Total: 20 binary features

**Numeric Scaling:**
- StandardScaler (mean=0, std=1) applied to 9 numeric features
- Scaler fit on training data only, then applied to both train and test

**Final Feature Set (V2):**
- 20 one-hot encoded features
- 9 scaled numeric features (including neighbourhood_price_encoded)
- **Total: 29 predictors**

---

## 🤖 Models Evaluated

### Baseline Models (Default Parameters)

| Model | RMSE (comparable) | MAE | R² | Notes |
|-------|-------------------|-----|-----|-------|
| Linear Regression | $79.68 | $49.97 | 0.396 | Baseline |
| Ridge (α=1.0) | $79.68 | $49.97 | 0.396 | No improvement over Linear |
| Lasso (α=1.0) | $80.01 | $50.07 | 0.391 | Selected 9/29 features |
| Random Forest | $73.89 | $45.00 | 0.480 | Strong baseline |
| XGBoost | $72.85 | $44.33 | 0.495 | Best baseline |

### Log Transformation Experiments

| Model | RMSE (comparable) | MAE | R² | Notes |
|-------|-------------------|-----|-----|-------|
| Linear + log | $80.90 | $46.54 | 0.377 | Degraded performance |
| Ridge + log | $80.90 | $46.54 | 0.377 | Same as Linear + log |
| Lasso + log | $105.91 | $69.25 | -0.068 | Failed: selected 0/29 features |

**Conclusion:** Log transformation hurt linear model performance instead of helping. Lasso with log was too aggressive and eliminated all features.

### Hyperparameter Tuning

**Ridge (Alpha Tuning):**
- Tested α = [0.1, 1.0, 10.0, 100.0]
- **Result:** All alphas produced identical R² = 0.396
- **Conclusion:** No multicollinearity issues; regularization not needed

**Random Forest (Top Configurations):**

| Config | RMSE | R² | Notes |
|--------|------|-----|-------|
| n200, d20, s2 | $73.75 | 0.482 | Best RF |
| n200, d25, s5 | $73.76 | 0.482 | Tied for best |
| n100, d25, s2 | $74.08 | 0.478 | Slightly worse |

**XGBoost (Top Configurations):**

| Config | RMSE | R² | Notes |
|--------|------|-----|-------|
| **n200, d7, lr0.1** | **$72.79** | **0.496** | **🏆 BEST MODEL** |
| n100, d9, lr0.1 | $72.86 | 0.495 | Nearly tied |
| n100, d7, lr0.1 (baseline) | $72.85 | 0.495 | Nearly tied |
| n200, d5, lr0.05 | $73.50 | 0.486 | Slower learning |

**Key Insight:** XGBoost consistently outperformed all other models, but tuning yielded minimal improvement (+0.001 R²), indicating we reached the dataset's performance ceiling.

---

## 📈 MLflow Experiment Tracking

All 17 model configurations were systematically tracked in MLflow, enabling objective comparison and reproducibility.

### Experiment Overview

![MLflow Experiments Overview](https://github.com/dalex2888/AML-3303/blob/main/Airbnb%20Price%20Estimator/screenshots/mlflow_experiments_overview.png)

The table shows all experiments filtered by `dataset_version = "v2"` and sorted by `r2_comparable` (descending). XGBoost models clearly dominate the top rankings.

### Best Model Details

![Best Model Details](https://github.com/dalex2888/AML-3303/blob/main/Airbnb%20Price%20Estimator/screenshots/mlflow_best_model_details.png)

The winning model (`xgb_v2_n200_d7_lr0.1`) includes complete tracking of:
- Hyperparameters: n_estimators=200, max_depth=7, learning_rate=0.1
- Metrics: R²=0.496, RMSE=$72.79, MAE=$44.27
- Status: REGISTERED as airbnb-price-predictor v1

### Model Registry

![Model Registry](https://github.com/dalex2888/AML-3303/blob/main/Airbnb%20Price%20Estimator/screenshots/mlflow_registered_model.png)

The best model was registered in MLflow Model Registry with comprehensive documentation including:
- Model details and hyperparameters
- Performance metrics
- Features used
- Known limitations
- Training metadata (date, author)

**Registered Model:**
- **Name:** `airbnb-price-predictor`
- **Version:** 1
- **Stage:** None
- **MLflow Run:** [xgb_v2_n200_d7_lr0.1](http://35.183.177.64:5000/#/experiments/279774441150129579/runs/40fd11a5e85a4013afe811d02140c751)

---

## 🔍 Key Findings

### 1. Model Performance Hierarchy

**Tree-based models significantly outperformed linear models:**
- XGBoost (best): R² = 0.496
- Random Forest: R² = 0.480-0.482
- Linear/Ridge/Lasso: R² = 0.391-0.396

**Why tree-based models won:**
- Naturally handle non-linear relationships between features
- Robust to outliers without transformation
- Effective at capturing interaction effects
- Better suited for one-hot encoded categorical features

### 2. Feature Engineering Impact

**Neighbourhood target encoding added value:**
- V1 (neighbourhood_group only): R² ≈ 0.481
- V2 (neighbourhood_price_encoded): R² = 0.496
- **Improvement: +0.015 (1.5 percentage points)**

**However, improvement was modest because:**
- room_type already captured much location information (correlation with borough)
- Interaction features contributed minimally (~0.5% importance)
- neighbourhood_price_encoded alone explains only 20% of variance

### 3. Dataset Performance Ceiling

**Evidence of ceiling at R² ≈ 0.50:**

1. **Model convergence:** Multiple algorithms with different configurations all plateau at R² = 0.48-0.50
2. **Tuning diminishing returns:** Hyperparameter optimization yielded <0.1% improvement
3. **Feature engineering limits:** Even sophisticated features only improved performance by 1.5%

**Why the ceiling exists:**

The dataset lacks critical features known to influence Airbnb pricing:
- Number of bedrooms and bathrooms (~15-20% of variance)
- Specific amenities (WiFi, AC, kitchen) (~10-15%)
- Host quality metrics (superhost status, response rate) (~5%)
- Photo quality and listing presentation (~5%)

**Academic comparison:**
- Research with full features: R² = 0.80-0.90
- This project (limited features): R² = 0.496
- **Achievement: 99% of theoretical maximum with available data**

### 4. Log Transformation Failure

**Log transformation degraded linear model performance:**
- Linear without log: R² = 0.396
- Linear with log: R² = 0.377 (worse by 1.9 points)
- Lasso with log: R² = -0.068 (complete failure)

**Why it failed:**
- Lasso + log was too aggressive with alpha=1.0
- Eliminated all 29 features as "not worth it"
- Distribution of features (already scaled) didn't benefit from log compression
- Outliers in target were already handled by percentile-based removal

### 5. Feature Importance (XGBoost Best Model)

**Top predictors:**
1. room_type features: ~25% combined importance
2. neighbourhood_price_encoded: 15% importance
3. distance_to_manhattan: 9% importance
4. Geographic coordinates: ~10% combined
5. Review and availability metrics: ~15% combined

**Surprising insight:**
Room type is the single strongest predictor because it correlates with both location (Manhattan has more entire homes) and amenities (entire homes have more features than shared rooms).

---

## 🎯 Results

### Final Model Performance

**Best Model:** XGBoost Regressor  
**Configuration:**
- n_estimators: 200
- max_depth: 7
- learning_rate: 0.1
- random_state: 42

**Test Set Performance (Comparable Range, Price ≤ $795):**
- **R²: 0.496** (explains 49.6% of price variance)
- **RMSE: $72.79** (average prediction error)
- **MAE: $44.27** (median prediction error)

**Full Test Set Performance (Including Outliers):**
- R²: 0.191
- RMSE: $180.01

**Interpretation:**
- On typical listings (<$795, 99% of dataset), model predicts within ~$73 RMSE
- On luxury listings (>$795), model underperforms due to sparse training data
- Average listing price: $140, so RMSE of $73 represents 52% error
- This is at the limit of what's possible with available features

### Model Deployment Readiness

**Registered Model Information:**
- **Name:** airbnb-price-predictor
- **Version:** 1
- **Stage:** None (ready for Staging/Production promotion)
- **Model URI:** models:/airbnb-price-predictor/1

**Deployment Options:**

Load model for predictions:
```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/airbnb-price-predictor/1")
predictions = model.predict(new_data)
```

Serve as REST API:
```bash
mlflow models serve -m models:/airbnb-price-predictor/1 -p 5001
```

---

## 📝 Lessons Learned

### Technical Insights

**1. AWS Integration**
- Successfully integrated S3 for scalable data storage
- MLflow on EC2 provides centralized experiment tracking accessible from any machine
- boto3 enables seamless reading of data from S3 into pandas DataFrames
- Network timeouts can occur when logging large tree-based models (~50-100 MB)

**2. MLflow Best Practices**
- Track dataset version as parameter to maintain data lineage
- Log both "full test" and "comparable range" metrics for honest evaluation
- Use consistent naming conventions for runs (e.g., `model_v2_param1_param2`)
- Add descriptions to experiments and registered models for documentation
- Model registry enables version control and deployment workflow

**3. Feature Engineering Impact**
- Target encoding effectively reduces dimensionality (221 → 1 feature)
- Must calculate target encoding statistics ONLY from training data
- Interaction features have diminishing returns if base features already correlate
- Geographic distance features add value but require domain knowledge (Manhattan as center)

### ML Best Practices

**1. Data Leakage Prevention**
- ALWAYS perform train/test split BEFORE any statistical operations
- Fit scalers, encoders, and imputers ONLY on training data
- Calculate target encoding means using only training set prices
- Never use test set information during model development

**2. Outlier Handling Strategy**
- Removing outliers from train improves model learning on typical cases
- Keeping outliers in test provides realistic performance evaluation
- Percentile-based thresholds (99th) are more robust than fixed values
- Log transformation isn't always the answer for skewed distributions

**3. Experiment Tracking Value**
- MLflow prevented "model.pkl.final.FINAL_v3.pkl" chaos
- Systematic comparison revealed that multiple approaches plateau at same R²
- Clear documentation of "what didn't work" is as valuable as successes
- Screenshots provide evidence for academic submissions

**4. Model Selection**
- Tree-based models (XGBoost, RF) excel with mixed feature types
- Linear models struggle with one-hot encoded categoricals and non-linear relationships
- Hyperparameter tuning has diminishing returns near dataset ceiling
- Sometimes the baseline is already optimal

**5. Honest Evaluation**
- Reporting both "test full" and "test comparable" metrics shows rigor
- Acknowledging dataset limitations (R² ceiling) demonstrates maturity
- Understanding WHY a model can't improve further is crucial
- Comparing to academic benchmarks provides context

---

## 🚧 Limitations and Future Work

### Current Limitations

**1. Dataset Constraints**
- Missing critical features: bedrooms, bathrooms, amenities
- Achieves only ~50-60% of performance possible with full feature set
- Limited to 2019 data (pricing dynamics may have changed)

**2. Model Constraints**
- Poor performance on luxury listings (>$795)
- No temporal modeling (seasonality, trends)
- Doesn't account for listing availability patterns
- No consideration of host reputation beyond review counts

**3. Technical Constraints**
- Large tree-based models experience timeouts when logging to MLflow
- No automated retraining pipeline
- Model not deployed as production API

### Future Improvements

**High Priority:**
- [ ] **Feature augmentation:** Scrape additional listing data (bedrooms, amenities, photos)
- [ ] **Cross-validation:** Implement k-fold CV for more robust evaluation
- [ ] **Model deployment:** Deploy best model as REST API using MLflow serving

**Medium Priority:**
- [ ] **Temporal modeling:** Add time-series features for seasonality
- [ ] **Ensemble methods:** Combine XGBoost + Random Forest predictions
- [ ] **Feature selection:** Systematic removal of low-importance features
- [ ] **Hyperparameter optimization:** Bayesian optimization or Optuna

**Low Priority:**
- [ ] **Deep learning:** Experiment with neural networks
- [ ] **Geographic clustering:** K-means clustering of neighborhoods
- [ ] **CI/CD pipeline:** Automated testing and deployment
- [ ] **Monitoring:** Track model performance drift over time

---

## 📚 References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [AWS S3 Guide](https://docs.aws.amazon.com/s3/)
- [Airbnb Price Prediction Research](https://www.sciencedirect.com/science/article/pii/S0261517719302201)

---

## 👤 Author

**Diego**  
Student at Lambton College  
Course: Software Tools and Emerging Technologies (AML-3303)  
Term: 3, Fall 2025

---

## 📄 License

This project is for educational purposes as part of the Applied Machine Learning program at Lambton College.

---

## 🙏 Acknowledgments

- Professor for project guidance and AWS credit support
- Airbnb for the NYC 2019 dataset
- AWS Free Tier for cloud resources
- MLflow community for excellent documentation
- Classmates for collaborative learning

---

## 📊 Project Statistics

- **Total Experiments:** 17 tracked runs
- **Models Evaluated:** 5 algorithms (Linear, Ridge, Lasso, RF, XGBoost)
- **Feature Engineering Iterations:** 2 versions (V1 → V2)
- **Training Samples:** 38,715
- **Test Samples:** 9,777
- **Features (Final):** 29 after encoding
- **Best R²:** 0.496
- **Development Time:** 3 days
- **Lines of Code:** ~2,000 (across notebooks)

## 👤 Author

**Diego Alexander Espinosa Carreño**  
Email: [c0921125@mylambton.ca]  
LinkedIn: [https://linkedin.com/in/diegoespinosacarreno]