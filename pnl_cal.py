import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def get_pnl(
    filepath: Path,
    output_dir: Path,
    eods: pd.DataFrame,
    contract: str,
) -> pd.DataFrame:
    """
    Compute daily mark-to-market PnL for multiple contract periods.
    """

    # ------------------------------------------------------------------
    # Load trades
    # ------------------------------------------------------------------

    trades = (
        pd.read_excel(filepath, sheet_name=contract)
        .dropna(how="all")
        .dropna(subset=["Quantity", "Rate"])
    )

    trades["Trade Date"] = pd.to_datetime(trades["Trade Date"])
    trades["Period"] = pd.to_datetime(trades["Period"])
    trades["period"] = trades["Period"].dt.strftime("%b %y")

    trades["commission"] = (
        trades["Rate"].clip(upper=20_000) * 0.001 * trades["Quantity"].abs()
        + 5 * trades["Quantity"].abs()
    )

    # ------------------------------------------------------------------
    # EODs
    # ------------------------------------------------------------------

    eods = eods.copy()
    eods["date"] = pd.to_datetime(eods["date"])

    # Only keep periods we actually traded
    periods = trades["period"].unique()
    eods = eods[eods["period"].isin(periods)]

    # ------------------------------------------------------------------
    # Master calendar
    # ------------------------------------------------------------------

    idx = (
        pd.Index(trades["Trade Date"].unique())
        .union(eods["date"].unique())
        .sort_values()
    )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    trade_qty = trades.pivot_table(
        index="Trade Date",
        columns="period",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    )

    positions = (
        trade_qty
        .reindex(idx)
        .fillna(0)
        .cumsum()
    )

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    prices = (
        eods.pivot_table(
            index="date",
            columns="period",
            values="value",
            aggfunc="last",
        )
        .reindex(idx)
        # .ffill()
    )

    # ------------------------------------------------------------------
    # Cashflows
    # ------------------------------------------------------------------

    cashflow = (
        -(trades["Quantity"] * trades["Rate"])
        .groupby(trades["Trade Date"])
        .sum()
        .reindex(idx, fill_value=0)
    )

    commission = (
        trades.groupby("Trade Date")["commission"]
        .sum()
        .reindex(idx, fill_value=0)
    )

    # ------------------------------------------------------------------
    # Daily dataframe
    # ------------------------------------------------------------------

    daily = pd.DataFrame(index=idx)

    daily["trade_cashflow"] = cashflow
    daily["commission"] = commission

    daily["cum_cashflow"] = cashflow.cumsum()
    daily["cum_commission"] = commission.cumsum()

    daily["mtm"] = (positions * prices).sum(axis=1)

    daily["gross_cumul_pnl"] = (
        daily["cum_cashflow"]
        + daily["mtm"]
    )

    daily["net_cumul_pnl"] = (
        daily["gross_cumul_pnl"]
        - daily["cum_commission"]
    )

    daily["daily_pnl"] = (
        daily["net_cumul_pnl"]
        .diff()
        .fillna(0)
    )

    # ------------------------------------------------------------------
    # Remove days before first position
    # ------------------------------------------------------------------

    has_position = positions.abs().sum(axis=1) > 0

    if has_position.any():
        first = has_position.idxmax()
        daily = daily.loc[first:]
        prices = prices.loc[first:]

    # ------------------------------------------------------------------
    # Sharpe
    # ------------------------------------------------------------------

    mean = daily["daily_pnl"].mean()
    std = daily["daily_pnl"].std()

    sharpe = mean / std if std else np.nan

    print(daily)
    print(f"Sharpe: {sharpe:.2f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    mask = prices.notna().any(axis=1)

    fig, ax = plt.subplots()

    x = daily.index[mask].strftime("%Y-%m-%d")
    y = daily.loc[mask, "net_cumul_pnl"]

    ax.plot(x, y, marker="o", linewidth=1)

    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45, ha="right")

    ax.set_xlabel("Date")

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x / 1000:.0f}")
    )

    ax.set_ylabel("PnL (k)")
    ax.set_title(f"Net Cumulative PnL ({contract})")

    fig.text(
        0.5,
        0.02,
        f"Sharpe: {sharpe:.2f}",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))

    output_path = output_dir / f"day_{mask.sum()}_real_pnl.png"
    fig.savefig(output_path, dpi=150)
    print(f"Real P&L fig saved at {output_path}")
    plt.close(fig)

    return daily