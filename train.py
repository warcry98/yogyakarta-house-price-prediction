
import pandas as pd
import re
import joblib
from pathlib import Path
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

data_path = Path("data/rumah123_yogya_unfiltered.csv")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

def train_and_evaluate(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return {
        "model": model,
        "metrics": {
            "mae": mean_absolute_error(y_val, y_pred),
            "rmse": root_mean_squared_error(y_val, y_pred),
            "r2": r2_score(y_val, y_pred),
        },
        "residuals": pd.DataFrame({
            "y_true": y_val,
            "y_pred": y_pred,
            "residual": y_val - y_pred
        })
    }

def tune_xgboost(xgb_pipeline, X_train, y_train):
    param_distributions = {
        "regressor__n_estimators": [200, 300, 500],
        "regressor__max_depth": [3, 5, 7],
        "regressor__learning_rate": [0.01, 0.05, 0.1],
        "regressor__subsample": [0.7, 0.8, 1.0],
        "regressor__colsample_bytree": [0.7, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=param_distributions,
        n_iter=15,
        scoring="neg_mean_absolute_error",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def cross_validate(model, X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    mae = -cross_val_score(
        model, X, y,
        scoring="neg_mean_absolute_error",
        cv=cv
    ).mean()

    rmse = np.sqrt(-cross_val_score(
        model, X, y,
        scoring="neg_mean_squared_error",
        cv=cv
    ).mean())

    r2 = cross_val_score(
        model, X, y,
        scoring="neg_mean_squared_error",
        cv=cv
    ).mean()

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

def parse_price(price: str):
    if pd.isna(price):
        return None
    
    price = price.replace("Rp", "").replace(" ", "").replace(",", ".")

    if "Miliar" in price:
        value = float(price.replace("Miliar", ""))
        return int(value * 1_000_000_000)
    
    if "Juta" in price:
        value = float(price.replace("Juta", ""))
        return int(value * 1_000_000)
    
def parse_area(area: str):
    if pd.isna(area):
        return None
    return int(re.sub(r"[^\d]", "", area))

def main():
    df = pd.read_csv(data_path).dropna()
    
    # drop unused column
    df = df.drop(columns=["nav-link", "description"])

    # convert price and area
    df["price"] = df["price"].apply(parse_price)
    df["surface_area"] = df["surface_area"].apply(parse_area)
    df["building_area"] = df["building_area"].apply(parse_area)

    target = "price"

    X = df.drop(columns=[target])
    y = df[target]

    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object"]).columns

    # Preprocessors
    pre_lr = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ])

    pre_xgb = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    # Models
    lr_model = Pipeline([
        ("preprocessor", pre_lr),
        ("regressor", LinearRegression())
    ])

    xgb_model = Pipeline([
        ("preprocessor", pre_xgb),
        ("regressor", XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        ))
    ])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr_cv = cross_validate(lr_model, X_train, y_train)

    lr_art = train_and_evaluate(lr_model, X_train, X_val, y_train, y_val)

    best_xgb_model, best_xgb_params = tune_xgboost(xgb_model, X_train, y_train)

    xgb_cv = cross_validate(best_xgb_model, X_train, y_train)

    xgb_art = train_and_evaluate(best_xgb_model, X_train, X_val, y_train, y_val)

    metadata = {
        "features": {
            "numerical": list(num_features),
            "categorical": list(cat_features)
        }
    }

    joblib.dump(
        {
            **lr_art, 
            "cv_metrics": lr_cv,
            **metadata
        }, 
        MODEL_DIR / "linear_model.pkl"
    )
    joblib.dump(
        {
            **xgb_art,
            "cv_metrics": xgb_cv,
            "best_params": best_xgb_params,
            **metadata
        }, 
        MODEL_DIR / "xgb_model.pkl"
    )

if __name__ == "__main__":
    main()