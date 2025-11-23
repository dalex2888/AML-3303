# MLflow Quick Reference Guide

## What is MLflow?

MLflow is an experiment tracking system that automatically logs:
- **Parameters:** Configuration choices you make (model type, hyperparameters)
- **Metrics:** Performance results (RMSE, MAE, R²)
- **Artifacts:** Files (trained models, plots, data)
- **Models:** Complete trained models ready for deployment

Think of it as a digital lab notebook that never loses your experiments.

---

## Your MLflow Setup

**Server location:** EC2 instance at http://35.183.177.64:5000
**How it works:** Your Python code sends logs → EC2 stores them → You view in browser UI

---

## Basic Usage Pattern

### One-time setup (beginning of notebook)
```python
import mlflow

# Connect to your EC2 server
mlflow.set_tracking_uri("http://35.183.177.64:5000")

# Create/use an experiment (like a project folder)
mlflow.set_experiment("your-experiment-name")
```

### For each model you train
```python
with mlflow.start_run(run_name="descriptive_name"):
    
    # 1. Train your model (normal code)
    model = SomeModel()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # 2. Calculate metrics (normal code)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # 3. LOG everything to MLflow
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")
```

---

## Common MLflow Commands

### Logging Parameters
```python
# Single parameter
mlflow.log_param("learning_rate", 0.01)

# Multiple parameters
mlflow.log_params({
    "alpha": 0.5,
    "max_iter": 1000,
    "solver": "lbfgs"
})
```

### Logging Metrics
```python
# Single metric
mlflow.log_metric("rmse", 45.23)

# Multiple metrics
mlflow.log_metrics({
    "rmse": 45.23,
    "mae": 32.10,
    "r2": 0.75
})
```

### Logging Models
```python
# Scikit-learn models
mlflow.sklearn.log_model(model, "model")

# XGBoost models
mlflow.xgboost.log_model(model, "model")

# Generic Python models
mlflow.pyfunc.log_model("model", python_model=model)
```

### Logging Artifacts (files)
```python
# Save a plot
plt.savefig("residual_plot.png")
mlflow.log_artifact("residual_plot.png")

# Save a CSV
df.to_csv("predictions.csv")
mlflow.log_artifact("predictions.csv")

# Save entire folder
mlflow.log_artifacts("outputs/")
```

---

## Viewing Results in MLflow UI

1. Open browser: http://35.183.177.64:5000
2. Click on your experiment name
3. See all runs in a table
4. Click on a run to see details
5. Compare runs side-by-side

**UI Features:**
- Sort by metrics (find best model)
- Filter runs by parameters
- Download logged artifacts
- Compare multiple runs visually

---

## Model Registry (Production Models)

### Register a model
```python
# During training
mlflow.sklearn.log_model(model, "model", registered_model_name="AirbnbPricePredictor")

# Or after training from UI:
# Click run → Artifacts → model → Register Model
```

### Load a registered model
```python
model_uri = "models:/AirbnbPricePredictor/1"  # version 1
loaded_model = mlflow.sklearn.load_model(model_uri)
predictions = loaded_model.predict(X_new)
```

---

## Complete Example
```python
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Setup (once)
mlflow.set_tracking_uri("http://35.183.177.64:5000")
mlflow.set_experiment("airbnb-price-prediction")

# Train and log
with mlflow.start_run(run_name="random_forest_v1"):
    
    # Model configuration
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    
    # Train
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    
    # Evaluate
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    
    # Log everything
    mlflow.log_params(params)
    mlflow.log_metrics({"rmse": rmse, "r2": r2})
    mlflow.sklearn.log_model(rf, "model")
    
    print(f"Logged run with RMSE: {rmse:.2f}")
```

---

## Troubleshooting

**Problem:** `ConnectionError: cannot connect to http://35.183.177.64:5000`
**Solution:** 
- Check EC2 instance is running
- Verify MLflow server is running on EC2: `ps aux | grep mlflow`
- Check security group allows port 5000

**Problem:** `MlflowException: Run with UUID ... not found`
**Solution:** 
- The run was deleted or experiment was cleaned
- Start a new run

**Problem:** Model not appearing in UI
**Solution:**
- Check you're looking at the correct experiment
- Refresh the page
- Verify the run completed without errors

**Problem:** Artifacts not uploading
**Solution:**
- Check file exists before logging
- Verify file path is correct
- Check EC2 has disk space

---

## Best Practices

1. **Use descriptive run names:** `"linear_regression_baseline"` not `"run1"`
2. **Log everything that matters:** Params, metrics, model, important plots
3. **Don't log too much:** Avoid logging every intermediate step
4. **Organize with experiments:** One experiment per project/dataset
5. **Add tags for filtering:** `mlflow.set_tag("version", "v2")`
6. **Document params:** Use clear param names like `"learning_rate"` not `"lr"`

---

## Quick Commands Reference
```python
# Setup
mlflow.set_tracking_uri("http://IP:5000")
mlflow.set_experiment("name")

# Run context
with mlflow.start_run(run_name="name"):
    mlflow.log_param("key", value)
    mlflow.log_metric("key", value)
    mlflow.log_artifact("file.png")
    mlflow.sklearn.log_model(model, "model")
    mlflow.set_tag("tag_key", "tag_value")

# Search runs (programmatically)
runs = mlflow.search_runs(experiment_names=["exp_name"])
best_run = runs.sort_values("metrics.rmse").iloc[0]
```

---

## Project Workflow

1. **Training phase** (03_modeling_mlflow.ipynb):
   - Set tracking URI and experiment
   - Loop through models
   - Each model = one `mlflow.start_run()`
   - Log params, metrics, model

2. **Comparison phase** (MLflow UI):
   - Open http://35.183.177.64:5000
   - Compare metrics across runs
   - Identify best model

3. **Registration phase**:
   - Register best model in Model Registry
   - Screenshot for documentation

4. **Documentation**:
   - Add screenshots to `screenshots/` folder
   - Update README with results

---

## Resources

- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- MLflow UI: http://35.183.177.64:5000
- EC2 Server: 35.183.177.64