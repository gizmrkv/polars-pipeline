import numpy as np
import polars as pl
from polars_pipeline.model import LightGBM, Stacker
from sklearn.datasets import (
    make_circles,
    make_classification,
    make_friedman1,
    make_friedman3,
    make_gaussian_quantiles,
    make_moons,
    make_s_curve,
    make_swiss_roll,
)
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def binary_valid_accuracy(X_np: np.ndarray, y_np: np.ndarray) -> float:
    X = pl.from_numpy(X_np, schema=[f"feature_{i}" for i in range(X_np.shape[1])])
    y = pl.from_numpy(y_np, schema=["target"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Stacker(
        LightGBM({"objective": "binary", "metric": "binary_logloss", "verbosity": -1}),
        fold=StratifiedKFold(n_splits=5),
    )
    model.fit(X_train, y_train)
    y_pred = model.transform(X_test)
    y_pred = y_pred.select(pl.all().gt(0.5))

    accuracy = accuracy_score(y_test, y_pred)
    return float(accuracy)


def multiclass_valid_accuracy(X_np: np.ndarray, y_np: np.ndarray) -> float:
    X = pl.from_numpy(X_np, schema=[f"feature_{i}" for i in range(X_np.shape[1])])
    y = pl.from_numpy(y_np, schema=["target"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Stacker(
        LightGBM(
            {
                "objective": "multiclass",
                "num_class": len(np.unique(y_np)),
                "metric": "multi_logloss",
                "verbosity": -1,
            },
        ),
        fold=StratifiedKFold(n_splits=5),
    )
    model.fit(X_train, y_train)
    y_pred = model.transform(X_test)
    y_pred = y_pred.transpose().select(pl.all().arg_max()).transpose()

    accuracy = accuracy_score(y_test, y_pred)
    return float(accuracy)


def regression_valid_mse(X_np: np.ndarray, y_np: np.ndarray) -> float:
    X = pl.from_numpy(X_np, schema=[f"feature_{i}" for i in range(X_np.shape[1])])
    y = pl.from_numpy(y_np, schema=["target"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Stacker(
        LightGBM(
            {
                "objective": "regression",
                "metric": "rmse",
                "verbosity": -1,
            },
        ),
        fold=KFold(n_splits=5),
    )
    model.fit(X_train, y_train)
    y_pred = model.transform(X_test)

    mse = mean_squared_error(y_test, y_pred)
    return float(mse)


def test_binary_circles():
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
    accuracy = binary_valid_accuracy(X, y)
    assert accuracy > 0.95, f"Accuracy is too low: {accuracy}"


def test_binary_classification():
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    accuracy = binary_valid_accuracy(X, y)
    assert accuracy > 0.89, f"Accuracy is too low: {accuracy}"


def test_binary_gaussian_quantiles():
    X, y = make_gaussian_quantiles(n_samples=1000, n_classes=2, random_state=42)
    accuracy = binary_valid_accuracy(X, y)
    assert accuracy > 0.96, f"Accuracy is too low: {accuracy}"


def test_binary_moons():
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    accuracy = binary_valid_accuracy(X, y)
    assert accuracy > 0.99, f"Accuracy is too low: {accuracy}"


def test_multiclass_classification():
    X, y = make_classification(
        n_samples=1000, n_classes=3, n_informative=3, random_state=42
    )
    accuracy = multiclass_valid_accuracy(X, y)
    assert accuracy > 0.89, f"Accuracy is too low: {accuracy}"


def test_multiclass_gaussian_quantiles():
    X, y = make_gaussian_quantiles(n_samples=1000, n_classes=3, random_state=42)
    accuracy = multiclass_valid_accuracy(X, y)
    assert accuracy > 0.96, f"Accuracy is too low: {accuracy}"


def test_regression_friedman1():
    X, y = make_friedman1(n_samples=1000, noise=0.1, random_state=42)
    mse = regression_valid_mse(X, y)
    assert mse < 0.72, f"MSE for friedman1 is too high: {mse}"


def test_regression_friedman3():
    X, y = make_friedman3(n_samples=1000, noise=0.1, random_state=42)
    mse = regression_valid_mse(X, y)
    assert mse < 0.02, f"MSE for friedman3 is too high: {mse}"


def test_regression_s_curve():
    X, y = make_s_curve(n_samples=1000, noise=0.1, random_state=42)
    mse = regression_valid_mse(X, y)
    assert mse < 0.04, f"MSE for s_curve is too high: {mse}"


def test_regression_swiss_roll():
    X, y = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
    mse = regression_valid_mse(X, y)
    assert mse < 0.03, f"MSE for swiss_roll is too high: {mse}"
