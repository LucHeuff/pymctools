from collections.abc import Hashable
from typing import Literal

import arviz as az
import polars as pl
import xarray as xr

from pymctools.exceptions import CoordinateNotFoundError, GroupNotFoundError

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


# TODO test coverage fixen met custom inference data dingen
# (Checken of dat handig werkt met fixtures voor de andere functies die mss hetzelfde doen)
# https://python.arviz.org/en/stable/api/generated/arviz.InferenceData.html#arviz.InferenceData


def check_group(idata: az.InferenceData, group: PredictiveGroups) -> None:
    """Check if the desired group appears in the InferenceData."""
    if group not in idata.groups():
        msg = f"{group} does not appear in the provided InferenceData."
        raise GroupNotFoundError(msg)


def check_coordinates(idata: az.InferenceData, data: pl.DataFrame) -> list[str]:
    """Check if the constant data coordinates appear in the data and return them."""
    coords = [str(c) for c in list(idata["constant_data"].coords)]

    if miss := set(coords) - set(data.columns):
        msg = f"""
        Coordinates {miss} don't appear in data.
        Did you forget to define dims somewhere in your model definition?
        """
        raise CoordinateNotFoundError(msg)

    return coords


def get_predictive_summary(
    idata: az.InferenceData, group: PredictiveGroups
) -> pl.DataFrame:
    """Convert prior or posterior predictive inference data to pl.DataFrame.

    Adds mean, median, max, min and 89% HDI for each variable in the inference data.
    If constant data appears, these will be joined to the predictions.

    Args:
        idata: InferenceData, output from pymc models.
        group: 'posterior_predictive' or 'prior_predictive', indicating which
            data should be converted

    Returns:
        polars.DataFrame with mean, median, HDI high, HDI low, max and min,
        combined with constant data if available

    Raises:
        GroupNotFoundError: if the desired group does not exist
        CoordinateNotFoundError: if a coordinate from constant data is missing

    """
    check_group(idata, group)

    def get_df(var: Hashable) -> pl.DataFrame:
        """Convert a variable into a Dataset with its mean and HDI values."""
        mean = idata[group][var].mean(("chain", "draw")).rename("mean")
        median = idata[group][var].median(("chain", "draw")).rename("median")
        max_ = idata[group][var].max(("chain", "draw")).rename("max")
        min_ = idata[group][var].min(("chain", "draw")).rename("min")
        hdi = az.hdi(idata[group][var], hdi_prob=0.89)
        higher = hdi.sel(hdi="higher")[var].drop_vars("hdi").rename("higher")  # pyright: ignore[reportAttributeAccessIssue]
        lower = hdi.sel(hdi="lower")[var].drop_vars("hdi").rename("lower")  # pyright: ignore[reportAttributeAccessIssue]

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

    variables = list(idata[group])
    data = pl.concat([get_df(var) for var in variables], how="vertical_relaxed")

    # Adding constant data if it appears.
    if "constant_data" in idata.groups():
        coords = check_coordinates(idata, data)
        constant = to_df(idata["constant_data"])
        data = data.join(constant, on=coords)
    return data


def get_predictive_counts(
    idata: az.InferenceData, group: PredictiveGroups, over: str = "obs"
) -> pl.DataFrame:
    """Convert prior or posterior predictive counts to pl.DataFrame.

    Calculates frequency counts for each variable.
    Designed to be used for count models, e.g. Binomial or Poisson

    Args:
        idata: InferenceData, output from pymc models.
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
    check_group(idata, group)

    def get_df(var: Hashable) -> pl.DataFrame:
        """Convert a variable to a Dataset with its counts and values."""
        var = str(var)
        return (
            to_df(idata[group][var])
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

    variables = list(idata[group])
    data = pl.concat([get_df(var) for var in variables], how="align")

    # Adding constant data if it appears.
    if "constant_data" in idata.groups():
        coords = check_coordinates(idata, data)
        constant = to_df(idata["constant_data"])
        data = data.join(constant, on=coords)
    return data


# TODO get_predictive_model -> Add new fixture?
# TODO add_outlier_indicators
# TODO get_covariance_matrix
# TODO get_ellipse
# TODO get_ellipse_data
