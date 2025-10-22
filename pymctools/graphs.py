import re
from functools import partial
from typing import Literal

import altair as alt
import numpy as np
import polars as pl
from arviz import InferenceData
from scipy.stats import rankdata
from xarray import Dataset

from pymctools.exceptions import (
    NoPosteriorError,
    VariablesNotFoundError,
)
from pymctools.utils import to_df

# ---- Defaults and types
Color = Literal[
    "black",
    "silver",
    "gray",
    "white",
    "maroon",
    "red",
    "purple",
    "fuchsia",
    "green",
    "lime",
    "olive",
    "yellow",
    "navy",
    "blue",
    "teal",
    "aqua",
    "orange",
    "aliceblue",
    "antiquewhite",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "blanchedalmond",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflowerblue",
    "cornsilk",
    "crimson",
    "cyan",
    "darkblue",
    "darkcyan",
    "darkgoldenrod",
    "darkgray",
    "darkgreen",
    "darkgrey",
    "darkkhaki",
    "darkmagenta",
    "darkolivegreen",
    "darkorange",
    "darkorchid",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkslateblue",
    "darkslategray",
    "darkslategrey",
    "darkturquoise",
    "darkviolet",
    "deeppink",
    "deepskyblue",
    "dimgray",
    "dimgrey",
    "dodgerblue",
    "firebrick",
    "floralwhite",
    "forestgreen",
    "gainsboro",
    "ghostwhite",
    "gold",
    "goldenrod",
    "greenyellow",
    "grey",
    "honeydew",
    "hotpink",
    "indianred",
    "indigo",
    "ivory",
    "khaki",
    "lavender",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgreen",
    "lightgrey",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "lightskyblue",
    "lightslategray",
    "lightslategrey",
    "lightsteelblue",
    "lightyellow",
    "limegreen",
    "linen",
    "magenta",
    "mediumaquamarine",
    "mediumblue",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mintcream",
    "mistyrose",
    "moccasin",
    "navajowhite",
    "oldlace",
    "olivedrab",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "rosybrown",
    "royalblue",
    "saddlebrown",
    "salmon",
    "sandybrown",
    "seagreen",
    "seashell",
    "sienna",
    "skyblue",
    "slateblue",
    "slategray",
    "slategrey",
    "snow",
    "springgreen",
    "steelblue",
    "tan",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "whitesmoke",
    "yellowgreen",
    "rebeccapurple",
]

Chart = (
    alt.Chart
    | alt.LayerChart
    | alt.FacetChart
    | alt.VConcatChart
    | alt.HConcatChart
    | alt.RepeatChart
)

AXIS_LABELS = 13
AXIS_TITLE = 15
LEGEND_TITLE = 16
LEGEND_LABELS = 15
TITLE = 18


def default_chart_config(chart: Chart) -> Chart:
    """Apply default configuration to a chart."""
    return (
        chart.configure_axis(grid=False)
        .configure_title(fontSize=TITLE, fontWeight=500)
        .configure_view(stroke=None)
        .configure_legend(titleFontSize=LEGEND_TITLE, labelFontSize=LEGEND_LABELS)
        .configure_axisY(
            titleAngle=0,
            titleAlign="left",
            titleY=-3,
            titleX=3,
            titleFontWeight="normal",
            titleFontStyle="italic",
            titleFontSize=AXIS_TITLE,
            labelFontSize=AXIS_LABELS,
        )
        .configure_axisX(
            labelAngle=0,
            titleFontSize=AXIS_TITLE,
            labelFontSize=AXIS_LABELS,
            titleFontWeight="normal",
            titleFontStyle="italic",
        )
    )


# ---- Check functions
InferenceGroup = Literal["posterior", "posterior_predictive", "prior_predictive"]


def check_group(data: InferenceData, group: InferenceGroup = "posterior") -> None:
    """Check if this InferenceData contains a posterior group."""
    if group not in data:
        msg = f"Provided InferenceData does not contain {group} samples."
        raise NoPosteriorError(msg)


def check_variables(dataset: Dataset, variables: list[str] | None) -> list[str]:
    """Check if the desired variables exist in this dataset.

    If variables is empty, returns the available variables instead.

    Args:
        dataset: to check variables in
        variables: to check/take from the dataset. Can be None.

    Returns:
        List of desired variables, or all variables if None.

    Raises:
        VariablesNotFoundError: if any of the desired variables does not appear.
    """
    available = list(map(str, dataset))
    if variables is None:
        return available
    remainder = set(variables) - set(available)
    if len(remainder) > 0:
        msg = f"Variables: {remainder} not found in Dataset."
        raise VariablesNotFoundError(msg)
    return variables


# ---- Helper functions


def process_dataset(
    dataset: Dataset, variables: list[str] | None = None
) -> pl.DataFrame:
    """Convert PyMC output (xr.Dataset) into a pl.DataFrame for altair to plot.

    Args:
        dataset: xr.Dataset taken from InferenceData
        variables (Optional): List of variables to take from dataset.

    Returns:
        pl.DataFrame in tidy form for altair to plot with.

    Raises:
        VariablesNotFoundError: if any of the desired variables does not appear.

    """
    variables = check_variables(dataset, variables)

    # pivoting out additional dimensions if they are relevant for the variable
    index = {"chain", "draw"}

    dfs = []

    def rename(name: str, pattern: str, replace: str) -> str:
        """Rename with regex."""
        if name in index:
            return name
        return re.sub(pattern, replace, name)

    for var in variables:
        array = dataset[var]
        df = to_df(array)
        if set(array.dims) != index:
            # If there are other dimensions present for this variable,
            # These need to be split out in so they can appear individually
            # in posterior distributions.
            # E.g. Effect(group=1, lang=en) etc.
            # Taking columns in order of discovery -> makes sure that everything
            # lines up with the correct levels in the renaming step
            columns = [str(col) for col in array.dims if col not in index]
            # Polars either just returns the value as the column name from a
            # pivot, or a ste of the values (e.g. {1, 2}) for each of the levels
            # This means having to treat the single variable case differently.
            if len(columns) == 1:
                column = next(iter(columns))
                pattern = r"(\w+)"
                replace = rf"{var}({column}=\1)"
            else:
                # Dynamically creating the regex pattern based on the number of
                # columns
                capture = r'"?([\w\s]+)"?'
                pattern = r"\{" + ",".join([capture] * len(columns)) + r"\}"
                replace = (
                    (var + "(")
                    + ", ".join(
                        [rf"{c}=\{i}" for i, c in enumerate(columns, start=1)]
                    )
                    + ")"
                )
            rename_columns = partial(rename, pattern=pattern, replace=replace)

            df = df.pivot(index=list(index), on=columns, values=var).rename(
                rename_columns
            )

        dfs.append(df)

    return pl.concat(dfs, how="align")


# --- Plot functions


def diagnostic_plots(
    data: InferenceData, variables: list[str] | None = None, bins: int = 50
) -> Chart:
    """Create diagnostics with trace and trank plots for each variable.

    Args:
        data: InferenceData, output from PyMC containing MCMC samples.
        variables (Optional): list of variables to plot. Defaults to all.
        bins (Optional): number of bins to use for trank plot.

    Returns:
        alt.HConcatChart containing each individual chart

    Raises:
        NoPosteriorError: if no posterior distribution is found in InferenceData
        NoPyMCError: if data was not generated from PyMC
        VariablesNotFoundError: if any of the variables are missing from the posterior
    """  # noqa: E501
    check_group(data)
    dataset = data["posterior"]

    plot_data = process_dataset(dataset, variables)
    variables = [var for var in plot_data.columns if var not in ["chain", "draw"]]

    # width and height of plot segments
    width, height = 450, 300

    def _trace(variable: str) -> alt.Chart:
        return (
            alt.Chart(plot_data, title=variable)
            .mark_line()
            .encode(
                alt.X("draw", axis=None),
                alt.Y(variable).scale(zero=False).title(None),
                alt.Color("chain:N", sort="descending", legend=None).scale(
                    scheme="purples"
                ),
            )
            .properties(width=width, height=height)
        )

    def _trank(variable: str) -> alt.Chart:
        ranks = plot_data.select(["chain", variable]).with_columns(
            ranks=pl.col(variable).map_batches(
                partial(rankdata, method="ordinal"), return_dtype=pl.Int64
            )
        )

        def _get_hist_df(chain: int) -> pl.DataFrame:
            hist, pos = np.histogram(
                ranks.filter(pl.col("chain") == chain)["ranks"], bins=bins
            )
            return pl.DataFrame({"chain": chain, "hist": hist, "pos": pos[1:]})

        tranks = pl.concat(_get_hist_df(c) for c in ranks["chain"].unique())
        return (
            alt.Chart(tranks, title=variable)
            .mark_line(interpolate="step")
            .encode(
                alt.X("pos", axis=None),
                alt.Y("hist").scale(zero=False).title(None),
                alt.Color("chain:N", legend=None).scale(scheme="category10"),
            )
            .properties(width=width, height=height)
        )

    trace_plots = alt.vconcat(*[_trace(var) for var in variables]).resolve_scale(
        y="independent", color="independent"
    )
    trank_plots = alt.vconcat(*[_trank(var) for var in variables]).resolve_scale(
        y="independent", color="independent"
    )

    return alt.hconcat(trace_plots, trank_plots)


def distribution_plots(
    data: InferenceData,
    variables: list[str] | None = None,
    group: InferenceGroup = "posterior",
    obs_name: str = "obs",
) -> Chart:
    """Plot distributions of MCMC samples.

    Args:
        data: InferenceData from PyMC containing MCMC samples.
        variables (Optional): list of variables to plot. Defaults to all.
        group: Whether to plot posterior, posterior_predictive or prior_predictive.
                Defaults to posterior
        obs_name (Optional): name of observation indicator. Defaults to 'obs'.
                             Only relevant for prior and posterior predictive.

    Returns:
        alt.HConcatChart containing each individual chart

    Raises:
        NoPosteriorError: if no posterior distribution is found in InferenceData
        NoPyMCError: if data was not generated from PyMC
        VariablesNotFoundError: if any of the variables are missing from the posterior

    """  # noqa: E501
    check_group(data, group)
    dataset = data[group]
    if group in ["posterior_predictive", "prior_predictive"]:
        dataset = dataset.mean(obs_name)

    title_prefix = group.replace("_", " ").capitalize()

    plot_data = process_dataset(dataset, variables)
    plot_variables = [
        var for var in plot_data.columns if var not in ["chain", "draw"]
    ]

    width, height = 600, 400

    def _density(variable: str) -> alt.LayerChart:
        line = pl.DataFrame({"zero": 0})
        rule = alt.Chart(line).mark_rule(strokeDash=[8, 8]).encode(x="zero")
        return (
            alt.Chart(plot_data, title=f"{title_prefix} distribution for {variable}")
            .transform_density(
                variable, groupby=["chain"], as_=[variable, "density"]
            )
            .mark_line()
            .encode(
                alt.X(variable).title(None),
                alt.Y("density:Q").title(None),
                alt.Color("chain", legend=None).scale(scheme="category10"),
            )
            .properties(width=width, height=height)
        ) + rule

    return alt.vconcat(*[_density(var) for var in plot_variables])
