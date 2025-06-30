import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from functools import partial
from scipy.special import expit
from scipy.optimize import minimize

# The Utility code is adopted from the Search Arena project https://github.com/lmarena/search-arena/tree/main
def get_matchups_models(df: pd.DataFrame) -> Tuple[np.ndarray, List[Any]]:
    """Get matchup pairs and model list from battle data."""
    n_rows = len(df)
    model_indices, models = pd.factorize(pd.concat([df["model_a"], df["model_b"]]))
    matchups = np.column_stack([model_indices[:n_rows], model_indices[n_rows:]])
    return matchups, models.to_list()


def preprocess_for_elo(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """Preprocess data for Elo calculation.
    
    Returns:
        matchups: (N,2) array with model ids for competitors
        outcomes: (N,) array with 1.0/0.5/0.0 for win/tie/loss of model_a
        models: list of model names
    """
    matchups, models = get_matchups_models(df)
    outcomes = np.full(len(df), 0.5)
    outcomes[df["winner"] == "model_a"] = 1.0
    outcomes[df["winner"] == "model_b"] = 0.0
    return matchups, outcomes, models


def preprocess_for_bt(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Any], np.ndarray]:
    """Preprocess data for Bradley-Terry model.
    
    Returns unique (matchup, outcome) sets with occurrence weights.
    """
    n_rows = len(df)
    schedule = np.full((n_rows, 3), fill_value=1, dtype=np.int32)
    schedule[:, [0, 1]], models = get_matchups_models(df)
    schedule[df["winner"] == "model_a", 2] = 2
    schedule[df["winner"] == "model_b", 2] = 0
    matchups_outcomes, weights = np.unique(schedule, return_counts=True, axis=0)
    matchups = matchups_outcomes[:, [0, 1]]
    outcomes = matchups_outcomes[:, 2].astype(np.float64) / 2.0
    weights = weights.astype(np.float64)
    return matchups, outcomes, models, weights


def preprocess_for_style(
    df: pd.DataFrame,
    style_elements: Sequence[str],
    add_one: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
    """Preprocess data for style-controlled analysis."""
    apply_ratio = list(np.ones(len(style_elements) // 2))
    matchups, outcomes, models = preprocess_for_elo(df)
    n = matchups.shape[0]
    k = int(len(style_elements) / 2)

    def extract_style_feature(x: Dict[str, Any], feature: str) -> float:
        val = x[feature]
        if isinstance(val, (int, float)):
            return val
        else:
            return sum(val.values())

    style_vector = np.zeros(shape=(2 * k, n), dtype=np.int32)
    for idx, element in enumerate(style_elements):
        style_vector[idx, :] = df.conv_metadata.map(
            partial(extract_style_feature, feature=element)
        ).values
    style_vector = np.ascontiguousarray(style_vector)

    style_diff = (style_vector[:k] - style_vector[k:]).astype(float)
    style_sum = (style_vector[:k] + style_vector[k:]).astype(float)

    if add_one:
        style_sum = style_sum + np.ones(style_diff.shape)

    apply_ratio_idx = np.flatnonzero(apply_ratio)
    style_diff[apply_ratio_idx] /= style_sum[apply_ratio_idx]

    style_mean = np.mean(style_diff, axis=1)
    style_std = np.std(style_diff, axis=1)
    features = ((style_diff - style_mean[:, np.newaxis]) / style_std[:, np.newaxis]).T

    return matchups, features, outcomes, models


def bt_loss_and_grad(
    ratings: np.ndarray,
    matchups: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
    alpha: float = 1.0
) -> Tuple[float, np.ndarray]:
    """Bradley-Terry loss function and gradient."""
    matchup_ratings = ratings[matchups]
    logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    probs = expit(logits)
    loss = -(
        (np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes)) * weights
    ).sum()
    matchups_grads = -alpha * (outcomes - probs) * weights
    model_grad = np.zeros_like(ratings)
    np.add.at(
        model_grad,
        matchups[:, [0, 1]],
        matchups_grads[:, None] * np.array([1.0, -1.0], dtype=np.float64),
    )
    return loss, model_grad


def fit_bt(
    matchups: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
    n_models: int,
    alpha: float,
    tol: float = 1e-6
) -> np.ndarray:
    """Fit Bradley-Terry model using L-BFGS-B optimization."""
    initial_ratings = np.zeros(n_models, dtype=np.float64)
    result = minimize(
        fun=bt_loss_and_grad,
        x0=initial_ratings,
        args=(matchups, outcomes, weights, alpha),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 100, "gtol": tol},
    )
    return result["x"]


def scale_and_offset(
    ratings: np.ndarray,
    models: List[Any],
    scale: float,
    init_rating: float,
    anchor_model_and_rating: Optional[Tuple[Any, float]] = None,
) -> np.ndarray:
    """Convert ratings to Elo scale with optional anchor model."""
    scaled_ratings = (ratings * scale) + init_rating
    if anchor_model_and_rating is not None:
        anchor_model, anchor_rating = anchor_model_and_rating
        baseline_idx = models.index(anchor_model)
        scaled_ratings += anchor_rating - scaled_ratings[..., [baseline_idx]]
    return scaled_ratings


def compute_bt(
    df: pd.DataFrame,
    base: float = 10.0,
    scale: float = 400.0,
    init_rating: float = 1000,
    tol: float = 1e-6,
    anchor_model_and_rating: Optional[Tuple[Any, float]] = None,
) -> pd.Series:
    """Compute Bradley-Terry ratings from battle data."""
    matchups, outcomes, models, weights = preprocess_for_bt(df)
    ratings = fit_bt(matchups, outcomes, weights, len(models), math.log(base), tol)
    scaled_ratings = scale_and_offset(
        ratings, models, scale, init_rating, anchor_model_and_rating
    )
    return pd.Series(scaled_ratings, index=models).sort_values(ascending=False)


def compute_bootstrap_bt(
    battles: pd.DataFrame,
    num_round: int,
    base: float = 10.0,
    scale: float = 400.0,
    init_rating: float = 1000.0,
    tol: float = 1e-6,
    num_cpu: Optional[int] = None,
    anchor_model_and_rating: Optional[Tuple[Any, float]] = None,
    offset: float = 0.0,
) -> pd.DataFrame:
    """Compute bootstrap Bradley-Terry ratings."""
    matchups, outcomes, models, weights = preprocess_for_bt(battles)
    rng = np.random.default_rng(seed=0)
    idxs = rng.multinomial(
        n=len(battles), pvals=weights / weights.sum(), size=(num_round)
    )
    boot_weights = idxs.astype(np.float64) / len(battles)
    bt_fn = partial(
        fit_bt, matchups, outcomes, n_models=len(models), alpha=np.log(base), tol=tol
    )
    results = []
    for weights_ in boot_weights:
        results.append(bt_fn(weights_))
    ratings = np.array(results)
    scaled_ratings = scale_and_offset(
        ratings, models, scale, init_rating + offset, anchor_model_and_rating
    )
    df = pd.DataFrame(scaled_ratings, columns=models)
    return df[df.median().sort_values(ascending=False).index]


# Global constant for contextual BT optimization
DIFF_MASK: np.ndarray = np.array([1.0, -1.0], dtype=np.float64)


def contextual_bt_loss_and_grad(
    params: np.ndarray,
    n_competitors: int,
    matchups: np.ndarray,
    features: np.ndarray,
    outcomes: np.ndarray,
    alpha: float = 1.0,
    reg: float = 1.0,
    half_reg: float = 0.5,
) -> Tuple[float, np.ndarray]:
    """Contextual Bradley-Terry loss function and gradient."""
    reg_loss = half_reg * np.inner(params, params)
    ratings = params[:n_competitors]
    feature_params = params[n_competitors:]
    matchup_ratings = ratings[matchups]
    bt_logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    context_logits = np.dot(features, feature_params)
    probs = expit(bt_logits + context_logits)
    loss = (
        -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes))).sum()
        + reg_loss
    )
    error = outcomes - probs
    grad = reg * params
    matchups_grads = -alpha * error
    np.add.at(
        grad[:n_competitors], matchups[:, [0, 1]], matchups_grads[:, None] * DIFF_MASK
    )
    grad[n_competitors:] -= np.dot(features.T, error)
    return loss, grad


def fit_contextual_bt(
    matchups: np.ndarray,
    features: np.ndarray,
    outcomes: np.ndarray,
    models: List[Any],
    idxs: Optional[np.ndarray] = None,
    alpha: float = math.log(10.0),
    reg: float = 0.5,
    tol: float = 1e-6,
) -> np.ndarray:
    """Fit contextual Bradley-Terry model with style features."""
    n_features = features.shape[1]
    n_models = len(models)
    initial_params = np.zeros(n_models + n_features, dtype=np.float64)
    half_reg = reg / 2.0
    if idxs is not None:
        matchups, features, outcomes = matchups[idxs], features[idxs], outcomes[idxs]
    result = minimize(
        fun=contextual_bt_loss_and_grad,
        x0=initial_params,
        args=(n_models, matchups, features, outcomes, alpha, reg, half_reg),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 100, "gtol": tol},
    )
    return result["x"]


def compute_style_control(
    df: pd.DataFrame,
    style_elements: Sequence[str],
    alpha: float = math.log(10.0),
    reg: float = 0.5,
    init_rating: float = 1000.0,
    scale: float = 400.0,
    tol: float = 1e-6,
    anchor_model_and_rating: Optional[Tuple[Any, float]] = None,
) -> Tuple[pd.Series, np.ndarray]:
    """Compute Bradley-Terry ratings with style control."""
    matchups, features, outcomes, models = preprocess_for_style(df, style_elements=style_elements)
    ratings_params = fit_contextual_bt(
        matchups,
        features,
        outcomes,
        models=models,
        alpha=alpha,
        reg=reg,
        tol=tol,
    )
    ratings = ratings_params[: len(models)]
    params = ratings_params[len(models):]
    scaled_ratings = scale_and_offset(
        ratings, models, scale, init_rating, anchor_model_and_rating
    )
    scaled_ratings = pd.Series(scaled_ratings, index=models).sort_values(
        ascending=False
    )
    return scaled_ratings, params


def compute_bootstrap_style_control(
    df: pd.DataFrame,
    style_elements: Sequence[str],
    num_round: int,
    alpha: float = math.log(10.0),
    reg: float = 0.5,
    init_rating: float = 1000.0,
    scale: float = 400.0,
    tol: float = 1e-6,
    num_cpu: Optional[int] = None,
    offset: float = 0.0,
    anchor_model_and_rating: Optional[Tuple[Any, float]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Compute bootstrap Bradley-Terry ratings with style control."""
    matchups, features, outcomes, models = preprocess_for_style(df, style_elements=style_elements)
    contextual_bt_fn = partial(
        fit_contextual_bt,
        matchups,
        features,
        outcomes,
        models,
        alpha=alpha,
        reg=reg,
        tol=tol,
    )
    np.random.seed(0)
    boot_idxs = np.random.randint(
        low=0, high=matchups.shape[0], size=(num_round, matchups.shape[0])
    )
    results = []
    for idx in boot_idxs:
        results.append(contextual_bt_fn(idx))
    ratings_params = np.array(results)
    ratings = ratings_params[:, : len(models)]
    params = ratings_params[:, len(models):]
    scaled_ratings = scale_and_offset(
        ratings, models, scale, init_rating + offset, anchor_model_and_rating
    )
    df_out = pd.DataFrame(scaled_ratings, columns=models)
    return df_out[df_out.median().sort_values(ascending=False).index], params
