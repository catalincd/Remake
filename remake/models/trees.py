"""Gradient-boosted / bagged tree models on engineered features.

These are the cheapest, strongest-per-FLOP models for this problem: entropy and
the compression/histogram fingerprint separate the archive cluster almost
linearly, and trees mix those features with no GPU. CPU-only by default (ROCm
GPU tree support is unreliable). Each builder returns an estimator exposing
.fit(X, y) and .predict_proba(X); tree_trainer.py drives them.
"""
from __future__ import annotations

import numpy as np

from ..registry import register


# --- LightGBM ---------------------------------------------------------------
@register("lgbm", kind="tree", input="features",
          description="LightGBM gradient-boosted trees on engineered features")
def build_lgbm(num_classes, n_estimators=800, num_leaves=255, learning_rate=0.05,
               feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
               min_child_samples=50, max_depth=-1, n_jobs=-1, **kw):
    import lightgbm as lgb
    # num_class is auto-derived from y by the sklearn wrapper; don't set it.
    return lgb.LGBMClassifier(
        objective="multiclass", n_estimators=n_estimators,
        num_leaves=num_leaves, learning_rate=learning_rate,
        feature_fraction=feature_fraction, bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq, min_child_samples=min_child_samples,
        max_depth=max_depth, n_jobs=n_jobs, verbose=-1, **kw)


# --- XGBoost ----------------------------------------------------------------
@register("xgb", kind="tree", input="features",
          description="XGBoost gradient-boosted trees on engineered features")
def build_xgb(num_classes, n_estimators=700, max_depth=10, learning_rate=0.08,
              subsample=0.8, colsample_bytree=0.7, tree_method="hist",
              n_jobs=-1, **kw):
    import xgboost as xgb
    # num_class/objective are auto-derived from y by the sklearn wrapper.
    return xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=subsample, colsample_bytree=colsample_bytree,
        tree_method=tree_method, n_jobs=n_jobs, eval_metric="mlogloss", **kw)


# --- Random Forest / Extra Trees (sklearn baselines) ------------------------
@register("rf", kind="tree", input="features",
          description="Random Forest (sklearn) baseline")
def build_rf(num_classes, n_estimators=400, max_depth=None, n_jobs=-1,
             min_samples_leaf=2, **kw):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs,
        min_samples_leaf=min_samples_leaf, **kw)


@register("extratrees", kind="tree", input="features",
          description="Extremely Randomized Trees (sklearn) baseline")
def build_extra(num_classes, n_estimators=500, n_jobs=-1, min_samples_leaf=2, **kw):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=n_jobs,
                                min_samples_leaf=min_samples_leaf, **kw)


# --- CatBoost (optional) ----------------------------------------------------
@register("catboost", kind="tree", input="features",
          description="CatBoost gradient boosting (optional dependency)")
def build_catboost(num_classes, iterations=800, depth=8, learning_rate=0.06,
                   **kw):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=iterations, depth=depth,
                              learning_rate=learning_rate, loss_function="MultiClass",
                              verbose=False, **kw)
