import plotly.graph_objects as go
from pathlib import Path
from pprint import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import time
from functools import wraps
from typing import Any

from datetime import datetime

def timeit(func):
    """Decorator to measure execution time of a function."""

    @wraps(func)  # Preserves function metadata (name, docstring, etc.)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        time_str = format_time(elapsed)
        print(f"Time taken by {func.__name__}: {time_str}")        
        return result
    return wrapper

@timeit
def make_rel_tenor(
    df: pd.DataFrame,
    rel_tenors: list[str],
    rolling_offset: int,
    value_col: str,
) -> pd.DataFrame:
    df = df.copy()

    trading_days = pd.DatetimeIndex(df["date"].unique()).sort_values()

    # init tenor and ensures type for future data storing
    df["tenor"] = pd.Series(pd.NA, index=df.index, dtype="string")

    for tenor in rel_tenors:
        tenor_periods = map_tenor_period(
            trading_days,
            tenor,
            rolling_offset,
        )
        mask = df["period"].eq(df["date"].map(tenor_periods))
        df.loc[mask, "tenor"] = tenor

    res = (
        df.dropna(subset=["tenor"])
        .drop_duplicates(["date", "tenor"])
        .pivot(index="date", columns="tenor", values=value_col)
        .sort_index()
    )

    return res

def reindex_ffill(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return data.reindex(
        pd.date_range(data.index.min(), data.index.max())
    ).ffill()


def shift_non_nan(s: pd.Series, i: int) -> pd.Series:
    s_clean = s.dropna()
    shifted = s_clean.shift(i)
    return shifted.reindex(s.index)


def shift_non_nan_df(df: pd.DataFrame, i: int) -> pd.DataFrame:
    return df.apply(shift_non_nan, i=i)


def dicts_equal_ignore_keys(d1, d2, ignore_keys):
    f1 = {k: v for k, v in d1.items() if k not in ignore_keys}
    f2 = {k: v for k, v in d2.items() if k not in ignore_keys}
    return f1 == f2


def dicts_equal_spec_keys(d1, d2, keys):
    f1 = {k: v for k, v in d1.items() if k in keys}
    f2 = {k: v for k, v in d2.items() if k in keys}
    return f1 == f2


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric Mean Absolute Percentage Error (sMAPE) to avoid division by zero which can happen with relative error
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return np.abs(y_true - y_pred) / np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, 1e-8) * 100


# def create_new_dir(dir, base_name):
#     i = 0
#     while True:
#         new_path = dir / f"{base_name}_{i}"
#         if not new_path.exists():
#             break
#         i += 1
#         # Check if directory is empty
#         if new_path.exists() and not any(new_path.iterdir()):
#             break  # Directory exists but is empty        
#     Path(new_path).mkdir(exist_ok=True)
#     return new_path


def create_new_dir(dir, base_name, extra="") -> Path:
    dir = Path(dir)
    dir.mkdir(exist_ok=True, parents=True)  # make sure base dir exists
    
    # Find all existing dirs starting with base_name
    existing = [d for d in dir.iterdir() if d.is_dir() and d.name.startswith(base_name + "_")]

    # Extract experiment numbers
    nums = []
    for d in existing:
        try:
            # Expect format: base_name_extra_i or base_name_i
            parts = d.name.split("_")
            num = int(parts[-1])  # last part is the counter
            nums.append(num)
        except ValueError:
            pass  # skip if last part isn't an integer

    i = max(nums) + 1 if nums else 0

    # Build new folder name
    if extra:
        new_name = f"{base_name}_{extra}_exp_{i}"
    else:
        new_name = f"{base_name}_exp_{i}"

    new_path = dir / new_name
    new_path.mkdir(exist_ok=True)
    
    return new_path


def create_new_filename(dir, base_name, extension):
    i = 0
    while True:
        new_path = dir / f"{base_name}_{i}.{extension}"
        if not new_path.exists():
            break
        i += 1
    return new_path


def convert_to_polars_date(date_val):
    """Convert any date type to Polars Date."""
    if isinstance(date_val, pd.Timestamp):
        return date_val.date()  # Returns datetime.date
    elif isinstance(date_val, np.datetime64):
        return pd.Timestamp(date_val).date()
    elif isinstance(date_val, str):
        return pd.to_datetime(date_val).date()
    elif isinstance(date_val, datetime):
        return date_val.date()
    else:
        return date_val
        

def format_time(elapsed) -> str:
    # Format time with appropriate units
    if elapsed < 60:
        time_str = f"{elapsed:.3f} seconds"
    elif elapsed < 3600:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        time_str = f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        time_str = f"{hours}h {minutes}m {seconds:.1f}s"
    return time_str


def map_first_day_tenor(trading_days: pd.DatetimeIndex, tenor: str, margin_days_roll: int) -> pd.Series:
    """
    Rolling with first day of tenor.
    For instance, 
        - M2 on 04 Jan 2000 will have value 01 March 2000, 
        - Q1 on 7th Jan 2026 will give 01 April 2026, 
        - Y1 on 22nd of April 2025 will give 01-01-2026
    """
    # Input validation: tenor has right format
    tenor_type = tenor[0].upper()
    n = int(tenor[1:])
    assert tenor_type in "MQY" and tenor[1:].isdigit()
    
    trading_days_ext = trading_days.append(
        pd.bdate_range(
            trading_days.max() + pd.offsets.BDay(),
            periods=margin_days_roll,
        )
    )
    
    days = trading_days_ext[:-margin_days_roll]
    days_shifted = trading_days_ext[margin_days_roll:]

    dec_cutoff = (
        pd.Series(trading_days_ext, index=trading_days_ext)
        .groupby(trading_days_ext.year)
        .apply(lambda x: x[x <= pd.Timestamp(x.name, 12, 24)].max())
    )

    days_shifted = pd.DatetimeIndex([
        pd.Timestamp(orig.year + 1, 1, 1) if orig.month == 12 and orig >= dec_cutoff.loc[orig.year] else d
        for orig, d in zip(days, days_shifted)
    ])

    if tenor_type == 'M':
        periods = pd.date_range(
            trading_days[0], 
            trading_days[-1] + pd.DateOffset(months=n+1), 
            freq='MS'
        )

    elif tenor_type == 'Q':
        months = pd.date_range(
            trading_days[0], 
            trading_days[-1] + pd.DateOffset(months=3*(n+1)), 
            freq='MS'
        )
        periods = months[months.month.isin([1,4,7,10])] # Keeps only first month of each trimester

    elif tenor_type == 'Y':
        periods = pd.date_range(
            trading_days[0], 
            trading_days[-1] + pd.DateOffset(years=n+1), 
            freq='YS'
        )

    values = []

    for d in days_shifted:
        if n == 0:
            value = pd.Timestamp(d.year, d.month, 1)
        else:
            future_periods = periods[periods > d]
            value = future_periods[n - 1]

        values.append(value)
    
    pd.Series(values, index=days).to_csv("tenor_periods.csv", header=["period"])

    return pd.Series(values, index=days)


def map_tenor_period(trading_days: pd.DatetimeIndex, tenor: str, margin_days_roll: int) -> pd.Series:
    """
    Rolling with first day of tenor.
    For instance, 
        - M2 on 04 Jan 2000 will have value 01 March 2000, 
        - Q1 on 7th Jan 2026 will give 01 April 2026, 
        - Y1 on 22nd of April 2025 will give 01-01-2026
    """
    first_day_tenor = map_first_day_tenor(trading_days, tenor, margin_days_roll)

    # print(tenor)
    # print(first_day_tenor[-50:])

    series=first_day_tenor.copy()

    tenor_type = tenor[0].upper()
    if tenor_type == 'M':
        series = first_day_tenor.dt.strftime('%b %y')
    elif tenor_type == 'Q':
        series = 'Q' + first_day_tenor.dt.quarter.astype(str) + ' ' + first_day_tenor.dt.strftime('%y')
    elif tenor_type == 'Y':
        series = 'Cal ' + first_day_tenor.dt.strftime('%y')

    return series


def period_to_months(tenor: str):
    tenor = tenor.strip()
    
    # Extract year (assume 2-digit -> 20xx)
    year = int(tenor.split()[-1])
    if year < 0 or year > 99:
        raise ValueError(f"Invalid year in tenor: {tenor}")    
    year += 2000

    # Month mapping
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    
    # --- Case 1: Calendar ---
    if tenor.startswith("Cal"):
        return [
            pd.Timestamp(year=year, month=m, day=1).strftime("%b%y")
            for m in range(1, 13)
        ]

    # --- Case 2: Quarter ---
    elif tenor.startswith("Q"):
        q = int(tenor[1])
        start_month = 3 * (q - 1) + 1
        return [
            pd.Timestamp(year=year, month=m, day=1).strftime("%b%y")
            for m in range(start_month, start_month + 3)
        ]

    # --- Case 3: Single month ---
    month_str = tenor[:3]
    if month_str in month_map:
        m = month_map[month_str]
        return [
            pd.Timestamp(year=year, month=m, day=1).strftime("%b%y")
        ]

    raise ValueError(f"Unrecognized period format: {tenor}")


def rolling_frontward_eff_ewm(
        series: pd.Series, 
        span: int, 
        m: int = 0, 
        n: int = 0,
        min_win_ratio: int = 1
    ) -> pd.Series:
    """
    Forward-looking EWM over next n valid points starting from the m-th next,
    with weights decaying according to index distance.
    
    series: pd.Series (can contain NaNs)
    n: number of valid points to include (default 0 = all future points)
    m: start from m-th next valid point (1-based)
    span: EWM span to control decay
    """
    min_win = n // min_win_ratio

    if n == 0:
        n = len(series)

    arr = series.to_numpy(dtype=float)
    result = np.full_like(arr, np.nan)
    alpha = 2 / (span + 1)

    valid_idx = series.index[~series.isna()]

    for t_pos, t in enumerate(series.index):
        # All future valid indices
        future_idx = valid_idx[valid_idx >= t]

        start_idx = m
        if len(future_idx) >= start_idx + 1 + min_win:
            end_idx = min(start_idx + n, len(future_idx))
            selected_idx = future_idx[start_idx:end_idx]
            selected_values = series.loc[selected_idx].to_numpy()

            # Compute distances according to index
            if pd.api.types.is_datetime64_any_dtype(series.index):
                distances = (selected_idx - t) / pd.Timedelta(days=1)
            elif pd.api.types.is_numeric_dtype(series.index):
                distances = selected_idx - t
            else:
                raise TypeError(f"Unsupported index type: {series.index.dtype}")

            # decay weights
            weights = (1 - alpha) ** distances

            result[t_pos] = np.sum(weights * selected_values) / np.sum(weights)

    return pd.Series(result, index=series.index)


def rolling_frontward_ewm(series: pd.Series, span: int) -> pd.Series:
    return series[::-1].ewm(span=span).mean()[::-1]


def rolling_frontward_nan_mean(series: pd.Series, start: int, end: int) -> pd.Series:
    """
    Forward-looking mean for a pandas Series: mean of series[t+start : t+end+1] for each t.
    NaN values are ignored (weight zero). Last indices naturally become NaN.
    """
    arr = series.to_numpy(dtype=float).reshape(-1)
    windows = np.lib.stride_tricks.sliding_window_view(arr[start:], end - start + 1)        
    lookahead_mean = np.full(len(arr), np.nan, dtype=float)
    lookahead_mean[:len(windows)] = np.nanmean(windows, axis=1)  # ignores NaNs
    
    return pd.Series(lookahead_mean, index=series.index)


def rolling_frontward_eff_mean(
        series: pd.Series, 
        n: int, 
        is_ret_log: bool,
        m: int=0,
        min_win_ratio: int = 1,
    ) -> pd.Series:
    """
    For each time t, take the n next available (non-NaN) points
    starting from the m-th next non-NaN point.
    For example: m=1, n=3 on Thursday will return [Friday, Monday, Tuesday]
    
    series: 1D array-like (can contain NaNs)
    n: number of points to average
    m: start from the m-th next non-NaN point
    """
    min_win = n // min_win_ratio

    arr = np.asarray(series, dtype=float).flatten()
    lookahead = np.full_like(arr, np.nan, dtype=float)

    # Indices of non-NaN values
    valid_idx = np.where(~np.isnan(arr))[0]

    for t in range(len(arr)):
        # Indices after current time
        future_idx = valid_idx[valid_idx >= t]

        # Start from m-th future non-NaN point
        start_idx = m
        if len(future_idx) >= start_idx + 1 + min_win:
            end_idx = min(start_idx + n, len(future_idx))
            selected_idx = future_idx[start_idx:end_idx]
            if is_ret_log:
                lookahead[t] = np.nansum(arr[selected_idx])
            else:
                lookahead[t] = np.nanmean(arr[selected_idx])
        else:
            lookahead[t] = np.nan # not enough points

    return pd.Series(lookahead, index=series.index)


def z_score(x):
    return (x - np.mean(x)) / np.std(x)


def normalize_to_one(series):
    return series / series.abs().max()


def rolling_normalize(df, window=252):
    return df / df.abs().rolling(window, min_periods=1).max()


"""
PLOTING FUNCTIONS
"""

def _format_time_axis(ax, maxticks=15):
    """
    Apply common formatting for time series x-axis.
    """
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=maxticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.7)


def _shade_missing_business_days(ax, series, alpha_missing=0.2):
    """
    Shade missing business days (NaNs) on the given axis.
    """
    import pandas as pd
    
    series = series.squeeze()
    cal_date_range = pd.date_range(series.index.min(), series.index.max(), freq='B')
    
    nan_mask = series.isna() & series.index.isin(cal_date_range)
    
    for dt in series.index[nan_mask]:
        ax.axvspan(dt, dt + pd.Timedelta(days=1), color='red', alpha=alpha_missing, zorder=0)


def plot_series(
        series, 
        filename, 
        title=None, 
        is_show_missing=True,
        maxticks=15, 
        marker_size=2, 
        alpha_missing=0.2,
        start="2024-01-01",
    ):
    """
    Plots a single time series (pandas Series) and saves to file.
    Highlights missing business days as a semi-transparent red region.
    """
    series = series.squeeze()
    
    fig, ax = plt.subplots(figsize=(10,5))
    
    series_trunc = series.loc[start:].dropna()
    series_trunc = series_trunc.squeeze()

    ax.plot(series_trunc.index, series_trunc.values, marker='o', linestyle='-', markersize=marker_size)
    
    if is_show_missing:
        _shade_missing_business_days(ax, series_trunc, alpha_missing)
        ax.axvspan(
            series_trunc.index.min(),
            series_trunc.index.min(),
            color="red",
            alpha=alpha_missing,
            label="Missing B-days",
        )

    if title:
        ax.set_title(title)
    
    _format_time_axis(ax, maxticks=maxticks)
    
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_multiple_series(
        series_list, 
        filename, 
        title=None, 
        is_show_missing=True, 
        marker_size=2, 
        alpha_missing=0.2,
        start="2024-01-01",
    ):
    """
    Plots multiple time series (pandas Series) on the same figure and saves to file.
    series_list: list of (series, label) tuples
    """
    
    fig, ax = plt.subplots(figsize=(10,5))

    for i, (series, label) in enumerate(series_list):
        series_trunc = series.loc[start:].dropna()
        series_trunc = series_trunc.squeeze()
        
        ax.plot(series_trunc.index, series_trunc.values, label=label, marker='o', linestyle='-', markersize=marker_size)
        
        if is_show_missing:
            _shade_missing_business_days(ax, series_trunc, alpha_missing)
    
    # Make missing days legend
    if is_show_missing:
        ax.axvspan(
            series_trunc.index.min(),
            series_trunc.index.min(),
            color="red",
            alpha=alpha_missing,
            label="Missing B-days",
        )

    if title:
        ax.set_title(title)
    
    if any(label for _, label in series_list):
        ax.legend()
    
    _format_time_axis(ax)
    
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


# def plot_with_signals(adj_prices, ema, signal, column, output_dir):
#     price = adj_prices[column]
#     ema_col = ema[column]
#     sig = signal[column]

#     plt.figure(figsize=(12, 6))
    
#     # price
#     plt.plot(price.index, price.values, label="Price")

#     # EMA
#     plt.plot(ema_col.index, ema_col.values, linestyle="--", label="EMA (20)")

#     # upward crosses (green X)
#     up = sig == 1
#     plt.scatter(
#         price.index[up],
#         price[up],
#         marker='x',
#         s=100,
#         color='green',
#         label='Bullish cross'
#     )

#     # downward crosses (red X)
#     down = sig == -1
#     plt.scatter(
#         price.index[down],
#         price[down],
#         marker='x',
#         s=100,
#         color='red',
#         label='Bearish cross'
#     )

#     plt.title(f"{column} - Price with EMA Cross Signals")
#     plt.legend()
#     plt.grid(True)

#     file_path = output_dir + f"/figs/{column}_ema.svg"

#     plt.savefig(file_path, bbox_inches="tight")
#     plt.close()


def align_yaxis_zero(ax1, ax2, pad_frac=0.2):
    y1min, y1max = ax1.get_ylim()
    y2min, y2max = ax2.get_ylim()

    def zero_frac(ymin, ymax):
        return (0 - ymin) / (ymax - ymin)

    z1 = zero_frac(y1min, y1max)

    span2 = y2max - y2min
    new_y2min = 0 - z1 * span2
    new_y2max = new_y2min + span2

    # add whitespace padding (key fix)
    pad = pad_frac * (new_y2max - new_y2min)

    ax2.set_ylim(new_y2min - pad, new_y2max + pad)


def save_interactive_plot(
    df,
    cfg,
    out_name,
    x_col,
    y_col,
    z_col,
    color_col = None,
    title="Interactive 3D Plot",
    special_last_y=None,
):
    
    """
    df: pandas DataFrame
    x_col, y_col: categorical or numeric
    z_col: numeric
    """

    xs_raw = df[x_col].tolist()
    ys_raw = df[y_col].tolist()
    zs = df[z_col].tolist()

    # ============================================================
    # SAFE AXIS ENCODER
    # ============================================================
    def encode_axis(vals, special_last=None):
        vals = list(vals)

        numeric = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        categorical = [v for v in vals if v not in numeric]

        numeric_sorted = sorted(set(numeric))
        categorical_sorted = sorted(set(categorical), key=str)

        ordered = numeric_sorted

        if special_last:
            ordered += [v for v in categorical_sorted if v not in special_last]
            ordered += [v for v in special_last if v in categorical_sorted]
        else:
            ordered += categorical_sorted

        mp = {v: i for i, v in enumerate(ordered)}
        return [mp[v] for v in vals], ordered

    # ============================================================
    # ENCODE AXES
    # ============================================================
    xs, x_ticks = encode_axis(xs_raw)
    ys, y_ticks = encode_axis(ys_raw, special_last=special_last_y)

    # ============================================================
    # PLOT
    # ============================================================
    data: list[Any] = []

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    if color_col is not None:
        if isinstance(color_col, list):
            color_vals = df[color_col].astype(str).agg(" | ".join, axis=1)
        else:
            color_vals = df[color_col].astype(str)
        # color_vals = df[color_col].astype(str).tolist()
        unique_vals = sorted(set(color_vals), key=str)

        for v in unique_vals:
            mask = df[color_col].astype(str) == v

            data.append(
                go.Scatter3d(
                    x=xs[mask],
                    y=ys[mask],
                    z=zs[mask],
                    mode="markers",
                    name=str(v),
                    marker=dict(size=5),
                )
            )
    else:
        data = [
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(
                    size=5,
                    color=zs,
                    colorscale="Viridis",
                    opacity=0.8,
                ),
            )
        ]

    # Add 0 flat surface
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    plane = go.Mesh3d(
        x=[x_min, x_max, x_max, x_min],
        y=[y_min, y_min, y_max, y_max],
        z=[0, 0, 0, 0],
        color="lightgray",
        opacity=0.5,
        name="Sharpe = 0",
    )

    data.append(plane)

    fig = go.Figure(data=data)

    # ============================================================
    # AXES
    # ============================================================
    x_label = x_col
    y_label = y_col
    z_label = z_col
    scene = {
        "xaxis": {"title": x_label},
        "yaxis": {"title": y_label},
        "zaxis": {"title": z_label},
    }

    if x_ticks is not None:
        scene["xaxis"].update(
            tickvals=list(range(len(x_ticks))),
            ticktext=x_ticks,
        )

    if y_ticks is not None:
        scene["yaxis"].update(
            tickvals=list(range(len(y_ticks))),
            ticktext=y_ticks,
        )

    fig.update_layout(scene=scene, title=title)

    # ============================================================
    # SAVE
    # ============================================================
    parts = out_name.rsplit(".", 1)
    suffix = f"_{x_col}_{y_col}"
    if color_col is not None:
        suffix += f"_{color_col}"
    base = parts[0]
    ext = parts[1] if len(parts) > 1 else ""
    filename = f"{base}{suffix}.{ext}" if ext else f"{base}{suffix}"

    out_path = cfg.fig_dir / filename
    fig.write_html(out_path)

    print(f"Saved interactive plot: {out_path}")