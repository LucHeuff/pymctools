import altair as alt
import arviz_base as az
import numpy as np
import polars as pl
import pytest
import xarray as xr
from pymctools.exceptions import (
    NoPosteriorError,
    VariablesNotFoundError,
)
from pymctools.graphs import (
    check_group,
    check_variables,
    default_chart_config,
    diagnostic_plots,
    distribution_plots,
    process_dataset,
)

# ---- Testing defaults


@pytest.fixture
def basic_chart() -> alt.Chart:
    """Generate a simple altair chart for basic testing purposes."""
    rng = np.random.default_rng(12345)

    df = pl.DataFrame(
        {"x": rng.uniform(-1, 1, size=50), "y": rng.uniform(-1, 1, size=50)}
    )

    return alt.Chart(df).mark_point().encode(alt.X("x"), alt.Y("y"))


def test_default_chart_config(basic_chart: alt.Chart) -> None:
    """Test default_chart_config()."""
    default_chart_config(basic_chart)


# ---- Testing check functions


def test_check_group_raises() -> None:
    """Test whether check_posterior() raises the correct exception."""
    data = xr.DataTree()
    with pytest.raises(NoPosteriorError):
        check_group(data)
    with pytest.raises(NoPosteriorError):
        check_group(data, "posterior_predictive")
    with pytest.raises(NoPosteriorError):
        check_group(data, "prior_predictive")


def test_check_variables_raises() -> None:
    """Test whether check_variables() raises the correct exception."""
    dataset = xr.Dataset()
    with pytest.raises(VariablesNotFoundError):
        check_variables(dataset, ["x", "y"])


def test_check_variables() -> None:
    """Test check_variables()."""
    dataset = xr.Dataset(
        {"x": ("obs", [7, 8, 9]), "y": ("obs", [4, 5, 6])}, coords={"obs": [0, 1, 2]}
    )
    variables = check_variables(dataset, ["x", "y"])
    assert variables == ["x", "y"]

    one_variable = check_variables(dataset, ["x"])
    assert one_variable == ["x"]

    no_variables = check_variables(dataset, None)
    assert no_variables == ["x", "y"]


# ---- Testing processing function


@pytest.fixture
def dataset() -> xr.Dataset:
    """xr.Dataset for testing purposes."""
    rng = np.random.default_rng(12345)
    c = 3
    d = 10
    chain = np.arange(c)
    draw = np.arange(d)

    return xr.Dataset(
        {
            "a": xr.DataArray(
                rng.normal(0, 1, size=(c, d)), coords={"chain": chain, "draw": draw}
            ),
            "b": xr.DataArray(
                rng.normal(0, 1, size=(c, d, 2)),
                coords={"chain": chain, "draw": draw, "group": ["A", "B"]},
            ),
            "c": xr.DataArray(
                rng.normal(0, 1, size=(c, d, 2, 3)),
                coords={
                    "chain": chain,
                    "draw": draw,
                    "group": ["A", "B"],
                    "shape": ["circle", "square", "triangle"],
                },
            ),
        }
    )


def test_process_dataset(dataset: xr.Dataset) -> None:
    """Test process_dataset()."""
    df = process_dataset(dataset)

    columns = [
        "chain",
        "draw",
        "a",
        "b(group=A)",
        "b(group=B)",
        "c(group=A, shape=circle)",
        "c(group=A, shape=square)",
        "c(group=A, shape=triangle)",
        "c(group=B, shape=circle)",
        "c(group=B, shape=square)",
        "c(group=B, shape=triangle)",
    ]

    assert isinstance(df, pl.DataFrame), "Output is not a pl.DataFrame"
    assert df.shape == (30, 11), "Dataframe has the wrong shape"
    assert len(df.columns) == len(columns), "Incorrect number of columns"
    assert df.columns == columns, "Dataframe has incorrect columns"


# ---- Testing plotting functions


@pytest.fixture
def multilevel_datatree() -> xr.DataTree:
    """xr.DataTree where the variable has more than 1 extra coordinate."""
    rng = np.random.default_rng(12346)
    chain = np.arange(4)
    draw = np.arange(500)
    var_1 = np.arange(3)
    var_2 = np.arange(4)

    posterior = xr.DataTree(
        xr.Dataset(
            {
                "alpha": (
                    ("chain", "draw", "var_1", "var_2"),
                    rng.normal(size=(4, 500, 3, 4)),
                )
            },
            {"chain": chain, "draw": draw, "var_1": var_1, "var_2": var_2},
        )
    )

    return xr.DataTree(children={"posterior": posterior})


def test_diagnostic_plots(multilevel_datatree: xr.DataTree) -> None:
    """Test diagnostic_plots()."""
    data = az.load_arviz_data("centered_eight")

    diagnostic_plots(data)  # pyright: ignore[reportArgumentType]

    diagnostic_plots(multilevel_datatree)


def test_distribution_plots(multilevel_datatree: xr.DataTree) -> None:
    """Test distribution_plots()."""
    data = az.load_arviz_data("centered_eight")

    distribution_plots(data)  # pyright: ignore[reportArgumentType]
    distribution_plots(data, group="posterior_predictive")  # pyright: ignore[reportArgumentType]

    distribution_plots(multilevel_datatree)
