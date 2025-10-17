from typing import Callable

import arviz as az
import numpy as np
import polars as pl
import pytest
import xarray as xr
from numpy.testing import assert_almost_equal
from pymctools.exceptions import (
    CoordinateNotFoundError,
    GroupNotFoundError,
    ModelNotFoundError,
)
from pymctools.utils import (
    center,
    check_coordinates,
    check_group,
    get_predictive_counts,
    get_predictive_model,
    get_predictive_summary,
    index,
    maximise,
    normalise,
    scale,
    standardise,
    to_df,
)


def test_calculation() -> None:
    """Basic test of calculation functions."""
    df = pl.DataFrame({"nums": [1, 2, 3, 4, 5], "cats": ["A", "B", "C", "D", "E"]})

    nums = pl.col("nums")
    cats = pl.col("cats")

    df = df.with_columns(
        standardised=standardise(nums),
        scaled=scale(nums),
        normalised=normalise(nums),
        centered=center(nums),
        maximised=maximise(nums),
        indices=index(cats),
    )

    assert len(df) == 5, "Length of dataframe is incorrect."  # noqa: PLR2004

    # --- Standardise
    assert_almost_equal(
        df.select(pl.col("standardised").mean()).item(),
        0,
        err_msg="Incorrect mean of standardise",
    )
    assert_almost_equal(
        df.select(pl.col("standardised").std()).item(),
        1,
        err_msg="Incorrect standard deviaton of standardise",
    )

    # --- Scale
    assert_almost_equal(
        df.select(pl.col("scaled").mean()).item(),
        1,
        err_msg="Incorrect mean of scale",
    )

    # --- Normalise
    assert_almost_equal(
        df.select(pl.col("normalised").min()).item(),
        0,
        err_msg="Incorrect min of normalise",
    )
    assert_almost_equal(
        df.select(pl.col("normalised").max()).item(),
        1,
        err_msg="Incorrect max of normalise",
    )

    # --- Center
    assert_almost_equal(
        df.select(pl.col("centered").mean()).item(),
        0,
        err_msg="Incorrect mean of center",
    )

    # --- Maximise
    assert_almost_equal(
        df.select(pl.col("maximised").max()).item(),
        1,
        err_msg="Incorrect max of maximise",
    )

    # --- Index
    assert isinstance(df["indices"].dtype, pl.Int64)
    assert df.select(pl.col("indices").n_unique()).item() == 5  # noqa: PLR2004
    assert df.select(pl.col("indices").min()).item() == 0


def test_dataset_to_df() -> None:
    """Test converting an xr.Dataset to a pl.DataFrame."""
    ds = xr.tutorial.load_dataset("basin_mask")
    df = to_df(ds)

    assert df.columns == ["Z", "Y", "X", "basin"], "Column names are incorrect."
    assert df.shape == (2_138_400, 4)


def test_dataarray_to_df() -> None:
    """Test converting an xr.DataArray to a pl.DataFrame."""
    da = xr.tutorial.load_dataset("basin_mask")["basin"]
    df = to_df(da)

    assert df.columns == ["Z", "Y", "X", "basin"], "Column names are incorrect."
    assert df.shape == (2_138_400, 4)


# --- Testing helper functions


def test_check_group_raises() -> None:
    """Test whether check_group() raises the correct exception."""
    idata = az.InferenceData()
    group = "prior_predictive"

    with pytest.raises(GroupNotFoundError):
        check_group(idata, group)


def test_check_coordinates_raises() -> None:
    """Test whether check_coordinates() raises the correct exception."""
    idata = az.load_arviz_data("centered_eight")
    data = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(CoordinateNotFoundError):
        check_coordinates(idata, data)  # pyright: ignore[reportArgumentType]


# --- Testing summary functions

N = 50


@pytest.fixture
def continuous_idata() -> az.InferenceData:
    """Generate random InferenceData with continuous values."""
    rng = np.random.default_rng(12345)
    return az.InferenceData(
        posterior_predictive=xr.Dataset(
            {
                "y": xr.DataArray(
                    rng.uniform(size=(4, 100, N)),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "obs": np.arange(N),
                    },
                )
            }
        ),
        prior_predictive=xr.Dataset(
            {
                "y": xr.DataArray(
                    rng.uniform(size=(4, 100, N)),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "obs": np.arange(N),
                    },
                )
            }
        ),
        constant_data=xr.Dataset(
            {"x": xr.DataArray(rng.uniform(size=N), coords={"obs": np.arange(N)})}
        ),
    )


@pytest.fixture
def counts_idata() -> az.InferenceData:
    """Generate random InferenceData with counts values."""
    rng = np.random.default_rng(12345)
    return az.InferenceData(
        posterior_predictive=xr.Dataset(
            {
                "y": xr.DataArray(
                    rng.binomial(5, 0.3, size=(4, 100, N)),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "obs": np.arange(N),
                    },
                )
            }
        ),
        prior_predictive=xr.Dataset(
            {
                "y": xr.DataArray(
                    rng.binomial(5, 0.3, size=(4, 100, N)),
                    coords={
                        "chain": np.arange(4),
                        "draw": np.arange(100),
                        "obs": np.arange(N),
                    },
                )
            }
        ),
        constant_data=xr.Dataset(
            {"x": xr.DataArray(rng.uniform(size=N), coords={"obs": np.arange(N)})}
        ),
    )


def test_get_predictive_summary(continuous_idata: az.InferenceData) -> None:
    """Test get_predictive_summary()."""
    data = continuous_idata
    columns = [
        "obs",
        "mean",
        "median",
        "higher",
        "lower",
        "max",
        "min",
        "variable",
        "x",
    ]

    posterior_summary = get_predictive_summary(data, group="posterior_predictive")  # pyright: ignore[reportArgumentType]

    assert isinstance(posterior_summary, pl.DataFrame), (
        "Posterior summary is not a dataframe"
    )
    assert posterior_summary.shape == (N, 9), "Posterior summary has wrong shape"
    assert posterior_summary.columns == columns, (
        "Posterior summary has incorrect columns"
    )
    assert posterior_summary.select(pl.col("variable").unique()).item() == "y", (
        "Posterior summary has incorrect variable"
    )

    prior_summary = get_predictive_summary(data, group="prior_predictive")  # pyright: ignore[reportArgumentType]

    assert isinstance(prior_summary, pl.DataFrame), (
        "Prior summary is not a dataframe"
    )
    assert prior_summary.shape == (N, 9), "Prior summary has the wrong shape"
    assert prior_summary.columns == columns, "Prior summary has incorrect columns"
    assert prior_summary.select(pl.col("variable").unique()).item() == "y", (
        "Prior summary has incorrect variable"
    )


def test_get_predictive_counts(counts_idata: az.InferenceData) -> None:
    """Test get_predictive_counts()."""
    data = counts_idata
    columns = ["obs", "value", "count", "frequency", "variable", "x"]

    posterior_counts = get_predictive_counts(data, group="posterior_predictive")

    assert isinstance(posterior_counts, pl.DataFrame), (
        "Posterior counts not is not a dataframe."
    )
    assert posterior_counts.shape == (276, 6), (
        "Posterior counts have incorrect shape"
    )
    assert posterior_counts.columns == columns, (
        "Posterior counts has incorrect columns"
    )
    assert posterior_counts.select(pl.col("variable").unique()).item() == "y", (
        "Posterior counts has incorrect variable"
    )

    prior_counts = get_predictive_counts(data, group="prior_predictive")

    assert isinstance(prior_counts, pl.DataFrame), (
        "Prior counts not is not a dataframe."
    )
    assert prior_counts.shape == (289, 6), "Prior counts have incorrect shape"
    assert prior_counts.columns == columns, "Prior counts has incorrect columns"
    assert prior_counts.select(pl.col("variable").unique()).item() == "y", (
        "Prior counts has incorrect variable"
    )


def test_get_predictive_model(continuous_idata: az.InferenceData) -> None:
    """Test get_predictive_model()."""
    data = continuous_idata
    columns = ["draw", "obs", "y", "x"]

    posterior_model = get_predictive_model(
        data, group="posterior_predictive", model_name="y"
    )
    assert isinstance(posterior_model, pl.DataFrame), (
        "Posterior model is not a dataframe"
    )
    assert posterior_model.shape == (5000, 4), "Posterior model has incorrect shape"
    assert posterior_model.columns == columns, (
        "Posterior model has incorrect columns"
    )

    prior_model = get_predictive_model(
        data, group="prior_predictive", model_name="y"
    )
    assert isinstance(prior_model, pl.DataFrame), "Prior model is not a dataframe"
    assert prior_model.shape == (5000, 4), "Prior model has incorrect shape"
    assert prior_model.columns == columns, "Prior model has incorrect columns"


def test_get_predictive_model_raises(continuous_idata: az.InferenceData) -> None:
    """Test if get_predictive_model raises the correct exception."""
    data = continuous_idata

    with pytest.raises(ModelNotFoundError):
        get_predictive_model(data, group="posterior_predictive", model_name="fiets")
