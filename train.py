
import pandas as pd
import re
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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
    df = df.drop(columns=["nav-link"])

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
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        ))
    ])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr_art = train_and_evaluate(lr_model, X_train, X_val, y_train, y_val)
    xgb_art = train_and_evaluate(xgb_model, X_train, X_val, y_train, y_val)

    metadata = {
        "features": {
            "numerical": list(num_features),
            "categorical": list(cat_features)
        }
    }

    joblib.dump({**lr_art, **metadata}, MODEL_DIR / "linear_model.pkl")
    joblib.dump({**xgb_art, **metadata}, MODEL_DIR / "xgb_model.pkl")

if __name__ == "__main__":
    main()