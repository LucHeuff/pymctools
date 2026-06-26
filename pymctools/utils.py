from typing import Literal

import arviz_stats as azs
import numpy as np
import polars as pl
import xarray as xr
from scipy import stats

from pymctools.exceptions import (
    CoordinateNotFoundError,
    GroupNotFoundError,
    LogLikelihoodNotFoundError,
    ModelNotFoundError,
)

# TODO over het algemeen kijken of dingen niet te ingewikkeld in elkaar zitten

# ---- Statistical calculations over pl.Series


def standardise(s: pl.Expr) -> pl.Expr:
    """Standardise a series, moving its mean to 0 and standard deviation to 1."""
    return (s - s.mean()) / s.std()


def scale(s: pl.Expr) -> pl.Expr:
    """Scale a series, by dividing the series by its mean."""
    return s / s.mean()


def normalise(s: pl.Expr) -> pl.Expr:
    """Normalise a series, by squeezing it to [0, 1]."""
    return (s - s.min()) / (s.max() - s.min())


def center(s: pl.Expr) -> pl.Expr:
    """Center a series, by subtracting its mean."""
    return s - s.mean()


def maximise(s: pl.Expr) -> pl.Expr:
    """Maximise a series to 1, by dividing by the maximum value."""
    return s / s.max()


def index(s: pl.Expr) -> pl.Expr:
    """Convert nominal values into an index."""
    return s.rank("dense").cast(pl.Int64) - 1


# --- Conversion functions


def to_df(data: xr.Dataset | xr.DataArray) -> pl.DataFrame:
    """Convert an xr.Dataset or xr.DataArray to a pl.DataFrame."""
    return pl.from_pandas(data.to_dataframe(), include_index=True)


# ---- Predictive Summaries

PredictiveGroups = Literal["prior_predictive", "posterior_predictive"]


def check_group(dt: xr.DataTree, group: PredictiveGroups) -> None:
    """Check if the desired group appears in the DataTree, and return it."""
    if f"/{group}" not in dt.groups:
        msg = f"{group} does not appear in the provided DataTree."
        raise GroupNotFoundError(msg)


def check_coordinates(dt: xr.DataTree, data: pl.DataFrame) -> list[str]:
    """Check if the constant data coordinates appear in the data and return them."""
    coords = [str(c) for c in list(dt["constant_data"].coords)]

    if miss := set(coords) - set(data.columns):
        msg = f"""
        Coordinates {miss} don't appear in data.
        Did you forget to define dims somewhere in your model definition?
        """
        raise CoordinateNotFoundError(msg)

    return coords


def _add_constant_data(dt: xr.DataTree, data: pl.DataFrame) -> pl.DataFrame:
    """Add constant data to the DataFrame, if it appears."""
    if "/constant_data" not in dt.groups:
        return data
    coords = check_coordinates(dt, data)
    constant = to_df(dt.constant_data.to_dataset())
    return data.join(constant, on=coords)


def get_predictive_summary(dt: xr.DataTree, group: PredictiveGroups) -> pl.DataFrame:
    """Convert prior or posterior predictive inference data to pl.DataFrame.

    Adds mean, median, max, min and 89% HDI for each variable in the inference data.
    If constant data appears, these will be joined to the predictions.

    Args:
        dt: xarray.DataTree, output from pymc models.
        group: 'posterior_predictive' or 'prior_predictive', indicating which
            data should be converted

    Returns:
        polars.DataFrame with mean, median, HDI high, HDI low, max and min,
        combined with constant data if available

    Raises:
        GroupNotFoundError: if the desired group does not exist
        CoordinateNotFoundError: if a coordinate from constant data is missing

    """
    check_group(dt, group)

    def get_df(var: str) -> pl.DataFrame:
        """Convert a variable into a Dataset with its mean and HDI values."""
        mean = dt[group][var].mean(("chain", "draw")).rename("mean")
        median = dt[group][var].median(("chain", "draw")).rename("median")
        max_ = dt[group][var].max(("chain", "draw")).rename("max")
        min_ = dt[group][var].min(("chain", "draw")).rename("min")
        hdi = azs.hdi(dt[group][var], prob=0.89)

        higher = hdi.sel(ci_bound="upper").drop_vars("ci_bound").rename("higher")
        lower = hdi.sel(ci_bound="lower").drop_vars("ci_bound").rename("lower")

        return to_df(
            xr.Dataset(
                {
                    "mean": mean,
                    "median": median,
                    "higher": higher,
                    "lower": lower,
                    "max": max_,
                    "min": min_,
                }
            )
        ).with_columns(variable=pl.lit(var))

    variables = list(dt[group])
    data = pl.concat([get_df(var) for var in variables], how="vertical_relaxed")

    # Adding constant data if it appears.
    return _add_constant_data(dt, data)


def get_predictive_counts(
    dt: xr.DataTree, group: PredictiveGroups, over: str = "obs"
) -> pl.DataFrame:
    """Convert prior or posterior predictive counts to pl.DataFrame.

    Calculates frequency counts for each variable.
    Designed to be used for count models, e.g. Binomial or Poisson

    Args:
        dt: xarray.DataTree, output from pymc models.
        group: 'posterior_predictive' or 'prior_predictive', indicating which
            data should be converted
        over (Optional): name of coordinate to summarise over. Defaults to 'obs'

    Returns:
        polars.DataFrame with frequency per group for each variable,
        combined with constant data if available

    Raises:
        GroupNotFoundError: if the desired group does not exist
        CoordinateNotFoundError: if a coordinate from constant data is missing

    """
    check_group(dt, group)

    def get_df(var: str) -> pl.DataFrame:
        """Convert a variable to a Dataset with its counts and values."""
        var = str(var)
        return (
            to_df(dt[group].to_dataset()[var])
            .group_by(over)
            .agg(pl.col(var).value_counts())
            .explode(var)
            .unnest(var)
            .with_columns(
                frequency=(pl.col("count") / pl.col("count").sum()).over(over),
                variable=pl.lit(var),
            )
            .rename({var: "value"})
        )

    variables = list(dt[group])
    data = pl.concat([get_df(var) for var in variables], how="align")

    # Adding constant data if it appears.
    return _add_constant_data(dt, data)


def get_predictive_model(
    dt: xr.DataTree, group: PredictiveGroups, model_name: str = "model"
) -> pl.DataFrame:
    """Convert prior or predictive model to dataframe.

    Args:
        dt: xarray.DataTree, output from pymc models.
        group: 'posterior_predictive' or 'prior_predictive', indicating which
            data should be converted
        model_name (Optional): name of model Dataset.

    Returns:
        polars.DataFrame with draws from model predictions,
        combined with constant data if available

    Raises:
        GroupNotFoundError: if the desired group does not exist
        CoordinateNotFoundError: if a coordinate from constant data is missing
        ModelNotFoundError: if the model named in model_name does not exist
    """
    check_group(dt, group)

    if model_name not in list(dt[group]):
        msg = f"xarray.DataTree.{group} does not contain {model_name}"
        raise ModelNotFoundError(msg)

    data = to_df(dt[group].to_dataset()[model_name].mean("chain"))

    # Adding constant data if it appears.
    return _add_constant_data(dt, data)


# --- Adding statistics


def outlier_indicators(
    dt: xr.DataTree,
) -> pl.DataFrame:
    """Create a dataframe with outlier indicators for the constant data.

    NOTE: requires log_likelihood to be calculated on the xarray.DataTree!

    Args:
        dt: xarray.DataTree, output from pymc models.

    Returns:
        polars.DataFrame with outlier indicators.

    Raises:
        LogLikelihoodNotFoundError: if log_likelihood is unavailable.

    """
    if "/constant_data" not in dt.groups:
        msg = """
        Provided xarray.DataTree has no constant_data.
        This makes it rather difficult to append outlier indicators
        """
        raise GroupNotFoundError(msg)

    if "/log_likelihood" not in dt.groups:
        msg = """
        Provided xarray.DataTree has no log_likelihood group.
        Use pm.compute_log_likelihood to compute these.
        """
        raise LogLikelihoodNotFoundError(msg)

    constant = to_df(dt.constant_data.to_dataset())
    loo = azs.loo(dt, pointwise=True)  # ty:ignore[call-non-callable]
    pareto_k = to_df(loo.pareto_k.rename("pareto_k"))

    # finding columns the two DataFrames have in common
    on = list(set(constant.columns) & set(pareto_k.columns))

    return constant.join(pareto_k, on=on)


# --- Calculations for drawing covariance ellipses


def get_covariance_matrix(rho: float, sigma_x: float, sigma_y: float) -> np.ndarray:
    """Calculate 2D covariance matrix.

    Args:
        rho: correlation
        sigma_x: standard deviation on x
        sigma_y: standard deviation on y

    Returns:
        2D np.ndarray containing covariance matrix

    """
    cov_xy = sigma_x * sigma_y * rho
    sigmas = np.asarray([sigma_x, sigma_y])
    rho_matrix = np.asarray([[1, cov_xy], [cov_xy, 1]])
    return np.diag(sigmas) @ rho_matrix @ np.diag(sigmas)


def get_ellipse(
    covariance_matrix: np.ndarray, confidence_interval: float, steps: int = 100
) -> pl.DataFrame:
    """Get data for plotting one ellipse for this covariance matrix and interval.

    Args:
        covariance_matrix: np.ndarray with 2D covariance matrix (see also get_covariance_matrix())
        confidence_interval: interval to draw ellipse at on domain [0, 1]
        steps (Optional): resolution for drawing the ellipse.

    Returns:
        pl.DataFrame with x and y positions, ci level and ordering for drawing
        the covariance ellipse

    """  # noqa: E501
    ppf = stats.chi2.ppf(confidence_interval, 2)
    t = np.linspace(0, 2 * np.pi, num=steps)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    rotation = np.column_stack([np.cos(t), np.sin(t)])
    positions = eigenvectors @ (np.sqrt(ppf * eigenvalues) * rotation).T

    return (
        pl.DataFrame({"x": positions[0, :], "y": positions[1, :]})
        .with_columns(confidence_interval=pl.lit(f"{confidence_interval:.0%}"))
        .with_row_index("order")
    )


def get_ellipses(
    covariance_matrix: np.ndarray,
    confidence_intervals: list[float] | None = None,
    mean_x: float = 0.0,
    mean_y: float = 0.0,
    steps: int = 100,
) -> pl.DataFrame:
    """Get data for multiple covariance ellipses for this covariance matrix.

    Args:
        covariance_matrix: np.ndarray with 2D covariance matrix (see also get_covariance_matrix())
        confidence_intervals: list of intervals to draw ellipse at on domain [0, 1]
        mean_x (Optional): x-offset for center of ellipse
        mean_y (Optional): y-offset for center of ellipse
        steps (Optional): resolution for drawing the ellipse.

    Returns:
        pl.DataFrame with x and y positions, ci level and ordering for drawing
        the covariance ellipse
    """  # noqa: E501
    if confidence_intervals is None:
        confidence_intervals = [0.1, 0.5, 0.8, 0.95]
    return pl.concat(
        [get_ellipse(covariance_matrix, ci, steps) for ci in confidence_intervals]
    ).with_columns(x=pl.col("x") + mean_x, y=pl.col("y") + mean_y)
