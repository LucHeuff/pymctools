# PyMCTools

Contains tools I've found useful when doing Bayesian Statistics and causal 
using [`pymc`](https://www.pymc.io). 

## Installation
Install using:
```
pip install git+https://www.github.com/LucHeuff/pymctools.git
```
or using a [`uv`](https://docs.astral.sh/uv/) project:
```
uv add git+https://www.github.com/LucHeuff/pymctools.git
```

# Graphs
The package contains the following graph functions:

- `diagnostic_plots`: creates trace and trank plots from `pymc` MCMC samples
- `distribution_plots`: creates posterior distribution plots from `pymc` MCMC samples
- `default_chart_config`: applies a default chart configuration to any [`altair`](https://altair-viz.github.io/) chart

# Utils
The package contains a set of utility functions.

## Data transformations
The following convenience functions for common data transformations using [`polars`](https://pola.rs/) are provided:

- `standardise`: subtracts the mean and divides by the standard deviation of the series, giving it a mean of 0 and a standard deviation of 1
- `scale`: divides the series by its mean
- `normalise`: squeezes the series to [0, 1]
- `center`: subtracts the mean of the series
- `maximise`: divides the series by the maximum value, maximising it to 1
- `index`: creates unique indices for each distinct nominal value in the series

A conversion function from [`xarray`](https://docs.xarray.dev/en/stable/index.html) is also provided:

-  `to_df`: converts `xarray.Dataset` or `xarray.DataArray` to a `polars.DataFrame`

## Predictive summaries
The following functions are provided to calculate over prior or posterior predictive samples.
Each of these functions returns a `polars` DataFrame.

- `get_predictive_summary`: calculates mean, median, min, max and 89% HDI for each variable
- `get_predictive_counts`: calculates frequency counts for each variable
- `get_predictive_model`: extracts prior or predictive models given by `model_name`, if you included one using `pm.Deterministic`.

## Adding outlier indicators
This function calculates pointwise outlier indicators using PSIS (`pareto_k`) and WAIC (`p_waic`),
and adds these to a `polars` DataFrame indexed along observations.

- `outlier_indicators`: calculates outlier indicators using the `log_likelihood` group. Adds variables from `constant_data` group if present. 

## Calculating covariance ellipses
These functions can be used to visualise 2D covariance matrices, when working with multivariate models.


- `get_covariance_matrix`: calculate 2D covariance matrix from correlation `rho` and deviations `sigma_x` and `sigma_y`
- `get_ellipse`: calculate a single covariance ellipse based on covariance matrix and confidence interval. Returns a `polars` DataFrame
- `get_ellipses`: calculate a set of covariance ellipses given a covariance matrix and a set of confidence intervals. Returns a `polars` DataFrame

# Exceptions
All exceptions in this package inhert from the `PyMCToolsError` parent class.




